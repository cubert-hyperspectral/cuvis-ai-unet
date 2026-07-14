"""Mode-aware convolutional building blocks for the configurable U-Net.

Three convolution modes share one block interface so the U-Net assembly code is
mode-agnostic:

- ``"2d"``   : spectral bands are treated as input channels; ``Conv2d`` over the
               spatial ``(H, W)`` plane. Internal tensors are ``[B, C, H, W]``.
- ``"3d"``   : the cube is a volume with the spectral axis as depth; ``Conv3d``
               over ``(D, H, W)``. Internal tensors are ``[B, C, D, H, W]``.
- ``"2p5d"`` : factorised (2+1)D (Tran et al., 2018) — each convolution is split
               into a spatial ``Conv3d`` ``(1, k, k)`` and a spectral ``Conv3d``
               ``(k, 1, 1)`` with an intermediate normalisation + activation. The
               hidden width between the two factors is chosen to match the
               parameter count of the full 3-D convolution it replaces, so
               ``2p5d`` and ``3d`` are directly comparable at equal capacity.

Block topology follows MONAI's DynUNet: a stage is a double convolution (two
conv units), the first carrying the optional downsampling stride. The ``skip``
option selects the residual behaviour (see :class:`DoubleConvBlock`).
Normalisation is InstanceNorm and the activation is LeakyReLU(0.01), matching
DynUNet's defaults.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch import Tensor

Mode = Literal["2d", "2p5d", "3d"]
SkipKind = Literal["none", "identity", "project"]
_VOLUMETRIC: frozenset[str] = frozenset({"2p5d", "3d"})


def is_volumetric(mode: str) -> bool:
    """Return True if ``mode`` operates on 5-D ``[B, C, D, H, W]`` volumes."""
    return mode in _VOLUMETRIC


def _expand(value: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Broadcast an int to a length-``ndim`` tuple, or validate a given tuple."""
    if isinstance(value, int):
        return (value,) * ndim
    out = tuple(value)
    if len(out) != ndim:
        raise ValueError(f"expected a length-{ndim} tuple, got {value!r}")
    return out


def _norm(mode: str, num_features: int) -> nn.Module:
    """InstanceNorm matching the tensor rank of ``mode`` (DynUNet default)."""
    if mode == "2d":
        return nn.InstanceNorm2d(num_features, affine=True)
    return nn.InstanceNorm3d(num_features, affine=True)


def _act() -> nn.Module:
    """LeakyReLU(0.01) — DynUNet's default activation."""
    return nn.LeakyReLU(negative_slope=0.01, inplace=True)


def r2plus1d_hidden(in_ch: int, out_ch: int, kd: int, kh: int, kw: int) -> int:
    """Hidden channel width that param-matches a full 3-D conv (Tran et al., 2018).

    A (2+1)D block factorises a full ``(kd, kh, kw)`` conv ``in_ch -> out_ch``
    (which has ``kd*kh*kw*in*out`` weights) into a spatial ``(1, kh, kw)`` conv
    ``in_ch -> M`` followed by a spectral ``(kd, 1, 1)`` conv ``M -> out_ch``
    (together ``M*(kh*kw*in + kd*out)`` weights). Equating the two counts gives

        M = kd*kh*kw*in*out / (kh*kw*in + kd*out)

    floored and clamped to at least 1.
    """
    numer = kd * kh * kw * in_ch * out_ch
    denom = kh * kw * in_ch + kd * out_ch
    return max(1, numer // denom)


def make_proj(
    mode: Mode, in_ch: int, out_ch: int, stride: int | tuple[int, ...] = 1, *, bias: bool = False
) -> nn.Module:
    """Rank-appropriate 1x1(x1) convolution for a residual-skip projection.

    Never factorised and carries no activation — a purely linear channel/stride
    match, so a residual add stays a linear shortcut.
    """
    if mode == "2d":
        s = _expand(stride, 2)
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s, bias=bias)
    s = _expand(stride, 3)
    return nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=s, bias=bias)


def make_upconv(
    mode: Mode, in_ch: int, out_ch: int, stride: int | tuple[int, ...] = 2, *, bias: bool = False
) -> nn.Module:
    """Transpose convolution for decoder upsampling (kernel == stride).

    For ``"2p5d"`` a full 3-D transpose conv is used: factorisation applies only
    to the feature-extracting convolutions, not the resampling operators.
    """
    if mode == "2d":
        s = _expand(stride, 2)
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=s, stride=s, bias=bias)
    s = _expand(stride, 3)
    return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=s, stride=s, bias=bias)


class ConvUnit(nn.Module):
    """One unit of a DynUNet double convolution: feature conv -> norm -> activation.

    For ``"2d"``/``"3d"`` the feature conv is a single ``Conv2d``/``Conv3d``. For
    ``"2p5d"`` it is a param-matched R(2+1)D pair with its own intermediate
    InstanceNorm + LeakyReLU, so the full unit is
    ``spatial -> norm -> act -> spectral -> norm [-> act]``.

    ``final_act=False`` drops the trailing activation; used for the second unit
    of a residual block, whose activation is applied after the skip-add.
    """

    def __init__(
        self,
        mode: Mode,
        in_ch: int,
        out_ch: int,
        kernel: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        *,
        final_act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if mode == "2p5d":
            kd, kh, kw = _expand(kernel, 3)
            sd, sh, sw = _expand(stride, 3)
            mid = r2plus1d_hidden(in_ch, out_ch, kd, kh, kw)
            layers += [
                nn.Conv3d(in_ch, mid, (1, kh, kw), (1, sh, sw), (0, kh // 2, kw // 2), bias=False),
                nn.InstanceNorm3d(mid, affine=True),
                _act(),
                nn.Conv3d(mid, out_ch, (kd, 1, 1), (sd, 1, 1), (kd // 2, 0, 0), bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            ]
        elif mode in ("2d", "3d"):
            ndim = 2 if mode == "2d" else 3
            k = _expand(kernel, ndim)
            s = _expand(stride, ndim)
            pad = tuple(kk // 2 for kk in k)
            conv_cls = nn.Conv2d if mode == "2d" else nn.Conv3d
            layers += [conv_cls(in_ch, out_ch, k, s, pad, bias=False), _norm(mode, out_ch)]
        else:
            raise ValueError(f"unknown mode {mode!r}; expected one of '2d', '2p5d', '3d'")
        if final_act:
            layers.append(_act())
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feature conv, normalisation, and (optionally) activation."""
        return self.block(x)


class DoubleConvBlock(nn.Module):
    """A DynUNet stage: two conv units, the first carrying the stride.

    ``skip`` selects the residual behaviour:

    - ``"project"`` (default) — learned 1x1 projected skip (DynUNet ``UnetResBlock``).
    - ``"identity"`` — identity skip; falls back to a 1x1 projection only when it
      must (channel change or ``stride > 1``), since an identity add is
      impossible once shapes differ.
    - ``"none"`` — no residual (DynUNet ``UnetBasicBlock``).

    For residual variants the trailing activation is applied *after* the
    skip-add, matching DynUNet.
    """

    def __init__(
        self,
        mode: Mode,
        in_ch: int,
        out_ch: int,
        kernel: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        skip: SkipKind = "project",
    ) -> None:
        super().__init__()
        if skip not in ("none", "identity", "project"):
            raise ValueError(f"unknown skip {skip!r}; expected 'none', 'identity', 'project'")
        self.skip = skip
        residual = skip != "none"

        self.unit1 = ConvUnit(mode, in_ch, out_ch, kernel, stride)
        self.unit2 = ConvUnit(mode, out_ch, out_ch, kernel, stride=1, final_act=not residual)

        ndim = 2 if mode == "2d" else 3
        shape_changes = (in_ch != out_ch) or any(s != 1 for s in _expand(stride, ndim))

        self.proj: nn.Module | None = None
        self.post_norm: nn.Module | None = None
        self.post_act: nn.Module | None = None
        if residual:
            # "identity" keeps a true identity shortcut only when shapes are
            # preserved; otherwise (and always for "project") project linearly.
            if skip == "project" or shape_changes:
                self.proj = make_proj(mode, in_ch, out_ch, stride)
                self.post_norm = _norm(mode, out_ch)
            self.post_act = _act()

    def forward(self, x: Tensor) -> Tensor:
        """Run the double conv, adding the (identity or projected) skip if residual."""
        out = self.unit2(self.unit1(x))
        if self.skip == "none":
            return out
        residual = x if self.proj is None else self.post_norm(self.proj(x))
        return self.post_act(out + residual)
