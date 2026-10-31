from types import SimpleNamespace

import torch

from sglang.srt.mem_cache import memory_pool


def test_cached_layer_views_share_storage_and_follow_slot_lifecycle(monkeypatch):
    monkeypatch.setattr(memory_pool, "_cpu_has_amx_support", False)
    params = SimpleNamespace(
        shape=SimpleNamespace(conv=[(4, 3)], temporal=(2, 4, 4)),
        dtype=SimpleNamespace(conv=torch.float32, temporal=torch.float32),
    )
    pool = memory_pool.MambaPool(
        size=4,
        spec_state_size=4,
        cache_params=params,
        mamba_layer_ids=[0, 1, 2],
        device="cpu",
    )

    layer = pool.mamba2_layer_cache(1)
    assert layer is pool.mamba2_layer_cache(1)
    assert layer.temporal._base is None
    assert layer.conv[0]._base is None
    assert layer.temporal.data_ptr() == pool.mamba_cache.temporal[1].data_ptr()
    assert layer.conv[0].data_ptr() == pool.mamba_cache.conv[0][1].data_ptr()

    slot = pool.alloc(1)
    layer.temporal[slot] = 3
    layer.conv[0][slot] = 5
    assert torch.all(pool.mamba_cache.temporal[1, slot] == 3)
    assert torch.all(pool.mamba_cache.conv[0][1, slot] == 5)
    assert torch.all(pool.mamba_cache.temporal[0] == 0)

    copy = pool.fork_from(slot)
    assert torch.all(layer.temporal[copy] == 3)
    assert torch.all(layer.conv[0][copy] == 5)
    pool.free(copy)
    recycled = pool.alloc(3)
    assert copy.item() in recycled.tolist()
    # All later writes and allocator clears must remain visible through the
    # previously constructed view of the original shared storage.
    assert torch.all(layer.temporal[recycled] == 0)
    assert torch.all(layer.conv[0][recycled] == 0)


def test_speculative_state_layer_view_keeps_every_buffer_aliased():
    full = memory_pool.MambaPool.SpeculativeState(
        conv=[torch.zeros(3, 4, 2)],
        temporal=torch.zeros(3, 4, 2, 2),
        intermediate_ssm=torch.zeros(3, 4, 2, 2, 2),
        intermediate_conv_window=[torch.zeros(3, 4, 2, 2)],
    )
    layer = full.at_layer_idx(2)
    for view, parent in (
        (layer.conv[0], full.conv[0]),
        (layer.temporal, full.temporal),
        (layer.intermediate_ssm, full.intermediate_ssm),
        (layer.intermediate_conv_window[0], full.intermediate_conv_window[0]),
    ):
        assert view._base is None
        assert view.data_ptr() == parent[2].data_ptr()
        view.fill_(7)
        assert torch.all(parent[2] == 7)
        assert torch.all(parent[:2] == 0)
