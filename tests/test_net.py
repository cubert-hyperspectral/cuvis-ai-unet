"""Unit tests for the DynUNet backbone (pure torch).

The backbone is strict (MONAI-DynUNet contract): spatial sizes must be
divisible by the stride grid. Arbitrary-size handling lives in the node and is
tested in ``test_nodes.py``; the utilities in ``test_tiling.py``.
"""

import pytest
import torch

from cuvis_ai_unet.net import DynUNetBackbone

pytestmark = pytest.mark.unit


def _count(net: torch.nn.Module) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize(
    "mode,spectral_downsample",
    [("2d", False), ("2p5d", False), ("2p5d", True), ("3d", False), ("3d", True)],
)
def test_forward_shape_and_grad_divisible(mode: str, spectral_downsample: bool) -> None:
    # Grid-divisible 48x40; bands=15 makes the spectral_downsample=True cases
    # exercise the internal depth pad (15 -> 16 for depth grid 4) for free.
    b, bands, h, w, k = 2, 15, 48, 40, 4
    net = DynUNetBackbone(
        mode,
        in_channels=bands,
        out_channels=k,
        features=(16, 32, 64),
        spectral_downsample=spectral_downsample,
    )
    x = torch.randn(b, bands, h, w)
    y = net(x)
    assert y.shape == (b, k, h, w)
    torch.nn.functional.cross_entropy(y, torch.randint(0, k, (b, h, w))).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_strict_raises_on_indivisible_hw(mode: str) -> None:
    net = DynUNetBackbone(mode, in_channels=8, out_channels=3, features=(16, 32, 64))
    assert net.spatial_grid == (4, 4)
    with pytest.raises(ValueError, match="multiples"):
        net(torch.randn(1, 8, 49, 41))


def test_depth_pad_internal() -> None:
    # Depth (bands) is padded internally: 15 bands pass through depth grid 4 ...
    net = DynUNetBackbone(
        "3d", in_channels=15, out_channels=3, features=(16, 32, 64), spectral_downsample=True
    )
    assert net.depth_grid == 4
    assert net(torch.randn(1, 15, 48, 40)).shape == (1, 3, 48, 40)
    # ... while H stays strict — proving only the depth axis is auto-padded.
    with pytest.raises(ValueError, match="multiples"):
        net(torch.randn(1, 15, 49, 40))


def test_2p5d_matches_3d_param_count() -> None:
    kw = {"in_channels": 16, "out_channels": 4, "features": (16, 32, 64)}
    p_2p5d = _count(DynUNetBackbone("2p5d", **kw))
    p_3d = _count(DynUNetBackbone("3d", **kw))
    # R(2+1)D is designed to param-match the full 3-D net.
    assert abs(p_2p5d / p_3d - 1.0) < 0.05


def test_explicit_per_axis_strides_roundtrip() -> None:
    # Lists (as a YAML round-trip would produce) must be accepted like tuples.
    net = DynUNetBackbone(
        "3d",
        in_channels=15,
        out_channels=4,
        features=(16, 32, 64),
        strides=[1, [1, 2, 2], [2, 2, 2]],
    )
    assert net.spatial_grid == (4, 4)
    assert net.depth_grid == 2
    y = net(torch.randn(2, 15, 48, 40))
    assert y.shape == (2, 4, 48, 40)


def test_rejects_wrong_input_channels() -> None:
    net = DynUNetBackbone("2d", in_channels=8, out_channels=2, features=(8, 16))
    with pytest.raises(ValueError, match="channels"):
        net(torch.randn(2, 5, 16, 16))  # 5 != 8 bands


def test_binary_single_logit_head() -> None:
    net = DynUNetBackbone("2d", in_channels=8, out_channels=1, features=(8, 16))
    assert net(torch.randn(2, 8, 16, 16)).shape == (2, 1, 16, 16)
