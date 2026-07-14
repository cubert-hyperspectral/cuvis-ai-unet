"""Unit tests for the boundary-padding and sliding-window utilities (pure torch)."""

import pytest
import torch

from cuvis_ai_unet.net import DynUNetBackbone
from cuvis_ai_unet.tiling import (
    compute_tile_offsets,
    gaussian_weight_map,
    pad_hw_to_multiple,
    sliding_window_inference,
)

pytestmark = pytest.mark.unit


def test_offsets_first_zero_last_flush_even() -> None:
    o = compute_tile_offsets(1000, 128, overlap=0.5)
    assert o[0] == 0 and o[-1] == 1000 - 128
    assert o == sorted(o)
    diffs = [b - a for a, b in zip(o, o[1:])]
    assert max(diffs) - min(diffs) <= 1  # evenly spread (integer rounding)


def test_offsets_single_when_size_equals_tile() -> None:
    assert compute_tile_offsets(128, 128, overlap=0.5) == [0]


def test_offsets_raises_when_smaller_than_tile() -> None:
    with pytest.raises(ValueError, match="pad the input"):
        compute_tile_offsets(100, 128, overlap=0.5)


def test_gaussian_map_properties() -> None:
    w = gaussian_weight_map((32, 24), torch.device("cpu"))
    assert w.shape == (1, 1, 32, 24)
    assert w.dtype == torch.float32
    assert torch.isclose(w.max(), torch.tensor(10.0))  # value scaling
    assert (w > 0).all()  # clamped away from zero
    # peak at the center, monotonically informative toward corners
    assert w[0, 0, 16, 12] > w[0, 0, 0, 0]


def test_pad_to_multiple_and_revert() -> None:
    x = torch.randn(2, 3, 49, 41)
    xp, revert = pad_hw_to_multiple(x, (4, 4))
    assert xp.shape[-2:] == (52, 44)
    assert torch.equal(xp[revert], x)  # revert slices recover the original


def _net() -> DynUNetBackbone:
    return DynUNetBackbone("2d", in_channels=3, out_channels=4, features=(8, 16)).eval()


def test_equivalence_input_equals_tile_uniform_bitwise() -> None:
    net = _net()
    x = torch.randn(2, 3, 16, 12)
    with torch.no_grad():
        direct = net(x)
    tiled = sliding_window_inference(net, x, (16, 12), overlap=0.5, gaussian=False)
    assert torch.equal(direct, tiled)  # (1*x)/1 — bitwise


def test_equivalence_input_equals_tile_gaussian_allclose() -> None:
    net = _net()
    x = torch.randn(2, 3, 16, 12)
    with torch.no_grad():
        direct = net(x)
    tiled = sliding_window_inference(net, x, (16, 12), overlap=0.5, gaussian=True)
    assert torch.allclose(direct, tiled, atol=1e-6)  # (w*x)/w per element


def test_sliding_window_arbitrary_frame_restores_size() -> None:
    net = _net()
    x = torch.randn(2, 3, 50, 41)  # non-divisible, larger than tile
    out = sliding_window_inference(net, x, (16, 16), overlap=0.5, gaussian=True)
    assert out.shape == (2, 4, 50, 41)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("overlap", [0.0, 0.5])
@pytest.mark.parametrize("tile_batch", [2, 8])
def test_tile_batch_matches_serial(overlap: float, tile_batch: int) -> None:
    """Batching tiles into one forward changes only throughput, not the result.

    ``b=2`` exercises the per-tile row slicing in the scatter; ``tile_batch`` values
    that both divide and exceed the tile count are covered.
    """
    net = _net()
    x = torch.randn(2, 3, 50, 41)  # larger than tile, non-divisible
    with torch.no_grad():
        serial = sliding_window_inference(net, x, (16, 16), overlap=overlap, tile_batch=1)
        batched = sliding_window_inference(net, x, (16, 16), overlap=overlap, tile_batch=tile_batch)
    assert batched.shape == serial.shape == (2, 4, 50, 41)
    assert torch.allclose(serial, batched, atol=1e-6)
