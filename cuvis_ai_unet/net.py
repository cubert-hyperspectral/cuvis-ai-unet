"""The configurable U-Net (DynUNet) as a plain PyTorch module.

Network depth and downsampling are data, not code: ``features`` gives the
channel width per stage and ``strides`` the (per-stage) downsampling factor, in
the spirit of MONAI's DynUNet. The module is layout-native PyTorch
(``[B, C, H, W]`` for ``2d``; ``[B, C, D, H, W]`` for ``2p5d``/``3d``) — the
BHWC<->NCHW transpose and the cuvis-ai port contract live in the node wrapper.

Size contract (matches MONAI DynUNet / nnU-Net practice)
---------------------------------------------------------
The backbone is *strict*: spatial input sizes must be divisible by the per-axis
product of all stage strides (:attr:`DynUNetBackbone.spatial_grid`), otherwise
``forward`` raises. Boundary padding/cropping for arbitrary sizes is the
responsibility of the caller — the ``DynUNet`` node pads with constant zeros
and crops the logits back automatically (nnU-Net's pad → predict → revert).

The spectral/depth axis is the one exception: the backbone *creates* that axis
itself (volumetric modes unsqueeze the band axis to depth), so it also owns its
divisibility — depth is trailing-padded to the depth grid internally and the
padded slices are cropped off again *before* the depth-collapse ``mean`` so
they can never bias the head input.

Volumetric modes (``2p5d``/``3d``) carry the spectral axis as depth ``D`` and
collapse it to a 2-D segmentation map at the head (mean over ``D`` then a 1x1
conv to ``out_channels``). Whether the spectral axis is downsampled alongside
the spatial axes is controlled by ``spectral_downsample`` (or by passing fully
explicit per-axis tuple strides).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cuvis_ai_unet.blocks import DoubleConvBlock, Mode, SkipKind, is_volumetric, make_upconv


def _stage_stride(
    mode: Mode, spec: int | tuple[int, ...], spectral_downsample: bool
) -> tuple[int, ...]:
    """Expand a per-stage stride entry to a mode-appropriate tuple.

    A tuple/list is used verbatim. An int ``s`` becomes ``(s, s)`` for ``2d``;
    for volumetric modes it becomes ``(s, s, s)`` when ``spectral_downsample``
    else ``(1, s, s)`` (spatial-only, keeping the spectral axis intact).
    """
    if isinstance(spec, (list, tuple)):
        return tuple(spec)
    s = int(spec)
    if mode == "2d":
        return (s, s)
    return (s, s, s) if spectral_downsample else (1, s, s)


def _grid(stage_strides: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    """Elementwise product of per-stage strides -> divisibility grid per axis."""
    return tuple(math.prod(s[d] for s in stage_strides) for d in range(len(stage_strides[0])))


class DynUNetBackbone(nn.Module):
    """Configurable U-Net backbone with 2D, factorised 2.5D, and 3D convolution modes.

    Layout-native PyTorch module; the cuvis-ai node wrapper (``node/dynunet.py``)
    owns the BHWC transpose, boundary padding, and port contract.
    """

    def __init__(
        self,
        mode: Mode,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (32, 64, 128, 256),
        strides: Sequence[int | tuple[int, ...]] | None = None,
        kernel_size: int = 3,
        skip: SkipKind = "project",
        spectral_downsample: bool = False,
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError(f"features needs >=2 stages, got {tuple(features)}")
        n = len(features)
        if strides is None:
            strides = [1] + [2] * (n - 1)  # first stage keeps resolution, rest halve
        if len(strides) != n:
            raise ValueError(f"strides ({len(strides)}) must match features ({n})")

        self.mode = mode
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.volumetric = is_volumetric(mode)
        self.stage_strides = [_stage_stride(mode, s, spectral_downsample) for s in strides]

        enc_in = 1 if self.volumetric else self.in_channels  # volumetric: 1 channel, depth=bands

        self.encoder = nn.ModuleList()
        prev = enc_in
        for i, f in enumerate(features):
            self.encoder.append(
                DoubleConvBlock(mode, prev, f, kernel_size, self.stage_strides[i], skip)
            )
            prev = f

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(n - 1, 0, -1):
            self.upconvs.append(
                make_upconv(mode, features[i], features[i - 1], self.stage_strides[i])
            )
            self.decoder.append(
                DoubleConvBlock(mode, features[i - 1] * 2, features[i - 1], kernel_size, 1, skip)
            )

        # 2-D head: volumetric features are collapsed over depth before this.
        self.head = nn.Conv2d(features[0], self.out_channels, kernel_size=1)

    # ------------------------------------------------------------------ grids

    @property
    def spatial_grid(self) -> tuple[int, int]:
        """Required divisibility of the input H, W (product of stage strides)."""
        g = _grid(self.stage_strides)
        return (g[-2], g[-1])

    @property
    def depth_grid(self) -> int:
        """Required divisibility of the depth/spectral axis (1 when never strided)."""
        if not self.volumetric:
            return 1
        return _grid(self.stage_strides)[0]

    # ---------------------------------------------------------------- forward

    def forward(self, x: Tensor) -> Tensor:
        """Map ``[B, in_channels, H, W]`` to segmentation logits ``[B, out_channels, H, W]``.

        H and W must be multiples of :attr:`spatial_grid`; the depth axis is
        padded/cropped internally (see module docstring).
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")

        gh, gw = self.spatial_grid
        H, W = x.shape[-2], x.shape[-1]
        if H % gh or W % gw:
            raise ValueError(
                f"spatial dims ({H}, {W}) must be multiples of ({gh}, {gw}) — the per-axis "
                f"products of the stage strides. Pad at the network boundary; the DynUNet "
                f"node does this automatically (constant-0 pad, logits cropped back)."
            )

        if self.volumetric:
            x = x.unsqueeze(1)  # [B, 1, D=bands, H, W]
            gd = self.depth_grid
            pad_d = (-x.shape[2]) % gd
            if pad_d:
                # Trailing pad so [:, :, :in_channels] is the exact revert later.
                x = F.pad(x, (0, 0, 0, 0, 0, pad_d))

        skips: list[Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        for j in range(len(self.decoder)):
            x = self.upconvs[j](x)
            skip = skips[-(j + 2)]
            if x.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    f"decoder/skip shape mismatch at stage {j}: upsampled {tuple(x.shape[2:])} "
                    f"vs skip {tuple(skip.shape[2:])}. With grid-divisible inputs these always "
                    f"match; this indicates inconsistent encoder/decoder stride configuration."
                )
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[j](x)

        if self.volumetric:
            # Crop any padded depth slices BEFORE the collapse so they cannot bias the mean.
            x = x[:, :, : self.in_channels].mean(dim=2)
        return self.head(x)
