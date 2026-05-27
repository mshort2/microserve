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
        self.cache = torch.empty(
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
