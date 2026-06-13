from __future__ import annotations

import torch

from microserve.config import ModelConfig


class FlatKVCache:
    """Flat KV cache for a fixed batch of concurrent sequences.

    Layout: [num_layers, 2 (K|V), batch_size, num_kv_heads, max_seq_len, head_dim].
    Each batch element gets its own slot along the batch axis; sequences share
    the model and the cache allocation but not the cache slots. A paged
    allocator that breaks the fixed-batch constraint comes later.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        max_seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device,
        batch_size: int = 1,
    ):
        # Zero-init: continuous batching reads K/V at slot positions that
        # may not have been written yet (the attention mask masks them out
        # but sdpa's FlashAttention backend can still propagate NaN if K is
        # NaN before the mask is applied). Zeros are safe.
        self.cache = torch.zeros(
            cfg.num_layers,
            2,
            batch_size,
            cfg.num_kv_heads,
            max_seq_len,
            cfg.head_dim,
            dtype=dtype,
            device=device,
        )
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def write(
        self, layer: int, start_pos: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write post-RoPE K and V into the cache at positions [start_pos, start_pos + T).

        k, v shape: [batch_size, num_kv_heads, T, head_dim].
        """
        T = k.shape[2]
        end_pos = start_pos + T
        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: write would reach position {end_pos} "
                f"but max_seq_len={self.max_seq_len}"
            )
        self.cache[layer, 0, :, :, start_pos:end_pos, :] = k
        self.cache[layer, 1, :, :, start_pos:end_pos, :] = v

    def read(self, layer: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read K and V for positions [0, end_pos).

        Returns each as [batch_size, num_kv_heads, end_pos, head_dim].
        """
        k = self.cache[layer, 0, :, :, :end_pos, :]
        v = self.cache[layer, 1, :, :, :end_pos, :]
        return k, v

    # ---- Per-sequence scatter/gather (continuous-batching path) ----

    def write_at(
        self,
        layer: int,
        slot_idxs: torch.Tensor,
        start_positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Per-sequence scatter write.

        Writes k[i] (T tokens) into cache slot slot_idxs[i] starting at
        position start_positions[i]. For each i, the write covers slot
        positions [start_positions[i], start_positions[i] + T).

        k, v: [B, num_kv_heads, T, head_dim]
        slot_idxs, start_positions: [B] long tensors
        """
        B, _, T, _ = k.shape
        max_end = int((start_positions + T).max().item())
        if max_end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: writes would reach position {max_end} "
                f"but max_seq_len={self.max_seq_len}"
            )
        if T == 1:
            # Decode: fully vectorized scatter.
            self.cache[layer, 0, slot_idxs, :, start_positions, :] = k.squeeze(2)
            self.cache[layer, 1, slot_idxs, :, start_positions, :] = v.squeeze(2)
        else:
            # Prefill: typically B=1 so this small loop is fine. Multi-sequence
            # prefill (chunked prefill) is not supported here.
            for i in range(B):
                slot = int(slot_idxs[i].item())
                start = int(start_positions[i].item())
                self.cache[layer, 0, slot, :, start:start + T, :] = k[i]
                self.cache[layer, 1, slot, :, start:start + T, :] = v[i]

    def gather_at(
        self,
        layer: int,
        slot_idxs: torch.Tensor,
        max_context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K, V from given slots up to max_context_len positions.

        Returns each as [B, num_kv_heads, max_context_len, head_dim].
        Positions beyond a sequence's own context_len contain zeros (init
        state); the caller's attention mask must mask them out.
        """
        k = self.cache[layer, 0, slot_idxs, :, :max_context_len, :]
        v = self.cache[layer, 1, slot_idxs, :, :max_context_len, :]
        return k, v
