from __future__ import annotations

import torch

from microserve.config import ModelConfig


class FlatKVCache:
    """Single-sequence flat KV cache for Phase 1b.

    Layout: [num_layers, 2 (K|V), num_kv_heads, max_seq_len, head_dim].
    The two-axis dimension (K|V) makes one contiguous allocation per sequence;
    BlockManager in Phase 4 replaces this with a paged pool shared across many
    concurrent sequences. The (num_kv_heads, max_seq_len, head_dim) tail matches
    attention's native K/V shape so reads and writes need no transposes.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        max_seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        self.cache = torch.empty(
            cfg.num_layers,
            2,
            cfg.num_kv_heads,
            max_seq_len,
            cfg.head_dim,
            dtype=dtype,
            device=device,
        )
        self.max_seq_len = max_seq_len

    def write(
        self, layer: int, start_pos: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write the post-RoPE K and V tensors into the cache.

        k, v shape: [B=1, num_kv_heads, T, head_dim]. Squeezes the batch dim
        because Phase 1b is single-sequence; Phase 2+ batches will be handled
        by indexing the BlockManager instead.
        """
        T = k.shape[2]
        end_pos = start_pos + T
        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: write would reach position {end_pos} "
                f"but max_seq_len={self.max_seq_len}"
            )
        self.cache[layer, 0, :, start_pos:end_pos, :] = k.squeeze(0)
        self.cache[layer, 1, :, start_pos:end_pos, :] = v.squeeze(0)

    def read(self, layer: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read K and V for positions [0, end_pos).

        Returns each as [B=1, num_kv_heads, end_pos, head_dim].
        """
        k = self.cache[layer, 0, :, :end_pos, :].unsqueeze(0)
        v = self.cache[layer, 1, :, :end_pos, :].unsqueeze(0)
        return k, v
