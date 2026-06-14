from __future__ import annotations

from collections import deque

import torch

from microserve.config import ModelConfig


class BlockManager:
    """Paged KV cache: a shared pool of fixed-size blocks.

    Layout: [num_layers, 2 (K|V), num_blocks, block_size, num_kv_heads, head_dim].
    Each sequence is allocated blocks on demand and keeps a per-sequence
    `block_table: list[int]` mapping logical token positions to physical block
    indices. Logical position `p` lives at `block_table[p // block_size]` slot
    `p % block_size`.

    Block size 16 matches vLLM's default. The choice is a tradeoff:
      - Smaller blocks waste less on the per-sequence tail but increase
        per-step indexing overhead.
      - Larger blocks reduce indexing but waste more when sequences finish
        partway through a block.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ):
        self.cfg = cfg
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Zero-init: positions beyond a sequence's context_len contain zeros
        # so the attention mask cleanly suppresses them under sdpa.
        self.kv_cache = torch.zeros(
            cfg.num_layers,
            2,
            num_blocks,
            block_size,
            cfg.num_kv_heads,
            cfg.head_dim,
            dtype=dtype,
            device=device,
        )
        self.free_blocks: deque[int] = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self.free_blocks)

    def allocate(self, n: int) -> list[int]:
        """Allocate n blocks and return their IDs. Raises if not enough free."""
        if not self.can_allocate(n):
            raise RuntimeError(
                f"Block pool exhausted: requested {n}, "
                f"only {self.num_free} of {self.num_blocks} free"
            )
        return [self.free_blocks.popleft() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        """Return blocks to the pool."""
        for b in block_ids:
            self.free_blocks.append(b)

    # ---- Scatter/gather K/V across block boundaries ----

    def write_kv(
        self,
        layer: int,
        write_block_ids: torch.Tensor,
        write_slot_ids: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write per-token K/V to (block, slot) pairs.

        k, v: [B, num_kv_heads, T, head_dim]
        write_block_ids, write_slot_ids: [B, T] long tensors. Token (b, t)
        writes into block write_block_ids[b, t] at slot write_slot_ids[b, t].
        """
        B, num_kv_heads, T, head_dim = k.shape
        flat_block = write_block_ids.reshape(-1)
        flat_slot = write_slot_ids.reshape(-1)
        # k.permute(0, 2, 1, 3): [B, T, num_kv_heads, head_dim]
        k_flat = k.permute(0, 2, 1, 3).reshape(B * T, num_kv_heads, head_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(B * T, num_kv_heads, head_dim)
        self.kv_cache[layer, 0, flat_block, flat_slot] = k_flat
        self.kv_cache[layer, 1, flat_block, flat_slot] = v_flat

    def gather_kv(
        self,
        layer: int,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K, V for sequences indexed by block_tables.

        block_tables: [B, max_blocks_per_seq] long tensor, padded with any
        valid block id (padding entries are masked out at attention time).

        Returns each as [B, num_kv_heads, max_blocks_per_seq * block_size, head_dim].
        Materializes a contiguous tensor — slow under PyTorch but correct;
        a real CUDA kernel reads block_table inline without this materialization.
        """
        # [B, max_blocks, block_size, num_kv_heads, head_dim]
        gathered_k = self.kv_cache[layer, 0, block_tables]
        gathered_v = self.kv_cache[layer, 1, block_tables]
        B, max_blocks, block_size, num_kv_heads, head_dim = gathered_k.shape
        k_full = gathered_k.permute(0, 3, 1, 2, 4).reshape(
            B, num_kv_heads, max_blocks * block_size, head_dim
        )
        v_full = gathered_v.permute(0, 3, 1, 2, 4).reshape(
            B, num_kv_heads, max_blocks * block_size, head_dim
        )
        return k_full, v_full
