"""Unit tests for the mode-aware conv blocks (pure torch)."""

import pytest
import torch

from cuvis_ai_unet.blocks import (
    ConvUnit,
    DoubleConvBlock,
    is_volumetric,
    make_upconv,
    r2plus1d_hidden,
)

pytestmark = pytest.mark.unit


def _inp(mode: str) -> torch.Tensor:
    """A small input tensor of the right rank for ``mode``."""
    return torch.randn(2, 4, 6, 8) if mode == "2d" else torch.randn(2, 4, 5, 6, 8)


def test_is_volumetric() -> None:
    assert not is_volumetric("2d")
    assert is_volumetric("2p5d") and is_volumetric("3d")


@pytest.mark.parametrize("mode", ["2d", "2p5d", "3d"])
def test_convunit_shape(mode: str) -> None:
    out = ConvUnit(mode, 4, 7)(_inp(mode))
    assert out.shape[1] == 7
    assert out.shape[0] == 2


@pytest.mark.parametrize("mode", ["2d", "2p5d", "3d"])
@pytest.mark.parametrize("skip", ["none", "identity", "project"])
def test_double_block_channels_and_stride(mode: str, skip: str) -> None:
    x = _inp(mode)
    block = DoubleConvBlock(mode, 4, 8, stride=2, skip=skip)
    out = block(x)
    assert out.shape[1] == 8  # out channels
    # spatial dims halved (last two axes), input 6x8 -> 3x4
    assert out.shape[-2:] == (3, 4)


def test_identity_skip_preserves_when_unchanged() -> None:
    # in==out and stride==1 -> "identity" keeps a true identity (no projection module)
    block = DoubleConvBlock("2d", 5, 5, stride=1, skip="identity")
    assert block.proj is None
    # a shape change forces a projection even under "identity"
    block2 = DoubleConvBlock("2d", 5, 9, stride=1, skip="identity")
    assert block2.proj is not None


@pytest.mark.parametrize(
    "in_ch,out_ch",
    [(1, 32), (32, 64), (64, 128), (320, 320)],
)
def test_r2plus1d_parity(in_ch: int, out_ch: int) -> None:
    kd = kh = kw = 3
    m = r2plus1d_hidden(in_ch, out_ch, kd, kh, kw)
    full = kd * kh * kw * in_ch * out_ch
    factored = m * (kh * kw * in_ch + kd * out_ch)
    # factorised param count matches the full 3-D conv within 5% (floor on M aside)
    assert abs(factored / full - 1.0) < 0.05


def test_upconv_doubles_spatial() -> None:
    up2d = make_upconv("2d", 8, 4, stride=2)
    assert up2d(torch.randn(2, 8, 5, 7)).shape[-2:] == (10, 14)
    up3d = make_upconv("3d", 8, 4, stride=2)
    assert up3d(torch.randn(2, 8, 3, 5, 7)).shape[-3:] == (6, 10, 14)
