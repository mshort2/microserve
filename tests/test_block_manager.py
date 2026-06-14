"""Unit tests for the BlockManager.

No model required; these run on CUDA only because BlockManager allocates a
GPU tensor. The state-machine tests use a tiny block pool.
"""

import pytest
import torch

from microserve.block_manager import BlockManager
from microserve.config import ModelConfig


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="BlockManager allocates a CUDA tensor"
)


def _bm(num_blocks: int = 4, block_size: int = 16) -> BlockManager:
    cfg = ModelConfig.qwen2_5_0_5b()
    return BlockManager(
        cfg=cfg,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cuda",
    )


def test_block_manager_starts_with_all_blocks_free():
    bm = _bm(num_blocks=8)
    assert bm.num_free == 8
    assert bm.can_allocate(8)
    assert bm.can_allocate(0)
    assert not bm.can_allocate(9)


def test_block_manager_allocate_returns_unique_ids():
    bm = _bm(num_blocks=4)
    ids = bm.allocate(3)
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert bm.num_free == 1


def test_block_manager_allocate_then_free_returns_to_pool():
    bm = _bm(num_blocks=4)
    ids = bm.allocate(3)
    assert bm.num_free == 1
    bm.free(ids)
    assert bm.num_free == 4
    # Can allocate everything again
    ids2 = bm.allocate(4)
    assert len(set(ids2)) == 4


def test_block_manager_exhaustion_raises():
    bm = _bm(num_blocks=4)
    bm.allocate(4)
    assert not bm.can_allocate(1)
    with pytest.raises(RuntimeError, match="exhausted"):
        bm.allocate(1)


def test_block_manager_partial_allocation_does_not_corrupt_pool():
    """can_allocate is checked atomically — allocate either succeeds entirely
    or raises before touching the pool.
    """
    bm = _bm(num_blocks=3)
    with pytest.raises(RuntimeError, match="exhausted"):
        bm.allocate(5)
    # The failed call shouldn't have popped anything
    assert bm.num_free == 3


def test_block_manager_write_then_gather_roundtrip():
    cfg = ModelConfig.qwen2_5_0_5b()
    bm = BlockManager(
        cfg=cfg,
        num_blocks=8,
        block_size=4,  # small block size makes verification easy
        dtype=torch.float32,
        device="cuda",
    )
    # Seq holds blocks [2, 5, 0] (out-of-order on purpose)
    seq_blocks = [2, 5, 0]
    seq_len = 3 * 4  # 12 positions
    layer = 7

    # Write distinct K/V values per token position
    k = torch.randn(1, cfg.num_kv_heads, seq_len, cfg.head_dim, device="cuda")
    v = torch.randn(1, cfg.num_kv_heads, seq_len, cfg.head_dim, device="cuda")
    write_block_ids = torch.tensor(
        [seq_blocks[p // 4] for p in range(seq_len)],
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    write_slot_ids = torch.tensor(
        [p % 4 for p in range(seq_len)], device="cuda", dtype=torch.long
    ).unsqueeze(0)

    bm.write_kv(layer, write_block_ids, write_slot_ids, k, v)

    # Gather back through the same block table; should recover what we wrote
    block_tables = torch.tensor([seq_blocks], device="cuda", dtype=torch.long)
    k_full, v_full = bm.gather_kv(layer, block_tables)
    # k_full shape: [1, num_kv_heads, 3*4=12, head_dim]
    assert k_full.shape == (1, cfg.num_kv_heads, seq_len, cfg.head_dim)
    # k after write has shape [1, num_kv_heads, T, head_dim] — k_full has same shape
    assert torch.equal(k_full, k)
    assert torch.equal(v_full, v)


def test_block_manager_writes_dont_leak_across_layers():
    bm = _bm(num_blocks=4, block_size=4)
    cfg = ModelConfig.qwen2_5_0_5b()
    k = torch.randn(1, cfg.num_kv_heads, 4, cfg.head_dim, device="cuda")
    v = torch.randn(1, cfg.num_kv_heads, 4, cfg.head_dim, device="cuda")
    write_block_ids = torch.zeros((1, 4), device="cuda", dtype=torch.long)
    write_slot_ids = torch.arange(4, device="cuda", dtype=torch.long).unsqueeze(0)

    bm.write_kv(layer=5, write_block_ids=write_block_ids, write_slot_ids=write_slot_ids, k=k, v=v)

    # Layer 5 should have what we wrote
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.long)
    k_back, _ = bm.gather_kv(layer=5, block_tables=block_tables)
    assert torch.equal(k_back, k)

    # Layer 0 should be untouched (still zeros from init)
    k_other, _ = bm.gather_kv(layer=0, block_tables=block_tables)
    assert torch.all(k_other == 0)
