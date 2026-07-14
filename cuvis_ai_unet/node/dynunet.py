"""The ``DynUNet`` segmentation node — a cuvis-ai wrapper over the U-Net backbone.

Input is a BHWC hyperspectral cube; output is per-pixel class logits in BHWC
layout (last-axis size == number of classes). The node owns the layout
transpose, the boundary size handling, and the port/serialization contract;
the network itself lives in :mod:`cuvis_ai_unet.net`.

Size handling (nnU-Net practice)
--------------------------------
The backbone is strict about divisibility; this node makes arbitrary input
sizes work at the boundary:

- **Direct path** (training, or small inputs): the input is center-padded with
  constant zeros to the next multiple of the backbone's stride grid, and the
  logits are cropped back to the original H, W.
- **Tiled path** (sliding-window inference): when ``tile_size`` is configured
  and the input exceeds it in INFERENCE/VAL/TEST (or, without a stage context,
  when the module is in eval mode), the input is processed in overlapping
  tiles of exactly ``tile_size``, blended with a Gaussian importance map and
  accumulated in float32 — mirroring nnU-Net v2. TRAIN always runs the direct
  path (the datamodule/augmentation owns training patch sizes).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_unet.blocks import Mode, SkipKind
from cuvis_ai_unet.net import DynUNetBackbone
from cuvis_ai_unet.tiling import (
    gaussian_weight_map,
    pad_hw_to_multiple,
    sliding_window_inference,
)


class DynUNet(Node):
    """Configurable U-Net for hyperspectral segmentation (2D / factorised 2.5D / 3D).

    Parameters
    ----------
    mode
        ``"2d"`` (bands as channels), ``"2p5d"`` (factorised (2+1)D spectral-spatial),
        or ``"3d"`` (full volumetric).
    in_channels
        Number of spectral bands in the input cube.
    num_classes
        Number of segmentation classes (output logit channels).
    features
        Channel width per U-Net stage; its length sets the network depth.
    strides
        Per-stage downsampling factor (int or per-axis tuple). Defaults to no
        downsampling on the first stage and a factor of 2 on the rest.
    kernel_size
        Convolution kernel size (single value, applied to every stage).
    skip
        Residual behaviour of each stage: ``"project"`` (default), ``"identity"``,
        or ``"none"``.
    spectral_downsample
        For volumetric modes, whether an int stride also downsamples the spectral
        axis (``True``) or only the spatial axes (``False``, default).
    tile_size
        Sliding-window tile (int or ``(h, w)``); use the training patch size.
        ``None`` (default) disables tiling — every input runs the direct path.
    tile_overlap
        Fraction of a tile shared by neighbouring tiles (default 0.5).
    tile_gaussian
        Blend overlapping tiles with a Gaussian importance map (default True).
    tile_batch
        Number of sliding-window tiles stacked into one backbone forward during
        tiled inference (default 1 = one tile per forward). Larger values fill the
        GPU better for small tiles; the result is unchanged (same tiles/weights),
        only throughput and memory differ.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.HYPERSPECTRAL,
            NodeTag.SEGMENTATION,
            NodeTag.LEARNABLE,
            NodeTag.DIFFERENTIABLE,
            NodeTag.INFERENCE,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C_bands]",
        )
    }

    OUTPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel class logits [B, H, W, num_classes]",
        )
    }

    def __init__(
        self,
        *,
        mode: Mode,
        in_channels: int,
        num_classes: int,
        features: Sequence[int] = (32, 64, 128, 256),
        strides: Sequence[int | tuple[int, ...]] | None = None,
        kernel_size: int = 3,
        skip: SkipKind = "project",
        spectral_downsample: bool = False,
        tile_size: int | Sequence[int] | None = None,
        tile_overlap: float = 0.5,
        tile_gaussian: bool = True,
        tile_batch: int = 1,
        **kwargs: Any,
    ) -> None:
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0.0 <= float(tile_overlap) < 1.0:
            raise ValueError(f"tile_overlap must be in [0, 1), got {tile_overlap}")
        if int(tile_batch) < 1:
            raise ValueError(f"tile_batch must be >= 1, got {tile_batch}")
        self.mode = mode
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.features = tuple(features)
        self.strides = None if strides is None else list(strides)
        self.kernel_size = int(kernel_size)
        self.skip = skip
        self.spectral_downsample = bool(spectral_downsample)
        self.tile_size = self._normalize_tile(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.tile_gaussian = bool(tile_gaussian)
        self.tile_batch = int(tile_batch)

        super().__init__(
            mode=mode,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            features=self.features,
            strides=self.strides,
            kernel_size=self.kernel_size,
            skip=skip,
            spectral_downsample=self.spectral_downsample,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            tile_gaussian=self.tile_gaussian,
            tile_batch=self.tile_batch,
            **kwargs,
        )

        self.net = DynUNetBackbone(
            mode=mode,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            features=self.features,
            strides=self.strides,
            kernel_size=self.kernel_size,
            skip=skip,
            spectral_downsample=self.spectral_downsample,
        )
        self._validate_tiling()
        # Plain-attribute cache (NOT a buffer: must stay out of state_dict).
        self._gauss_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------- validation

    @staticmethod
    def _normalize_tile(tile_size: int | Sequence[int] | None) -> tuple[int, int] | None:
        """Normalize ``tile_size`` to a ``(h, w)`` tuple (yaml lists accepted)."""
        if tile_size is None:
            return None
        if isinstance(tile_size, int):
            tile = (tile_size, tile_size)
        else:
            tile = tuple(int(t) for t in tile_size)
        if len(tile) != 2 or any(t <= 0 for t in tile):
            raise ValueError(f"tile_size must be a positive int or (h, w) pair, got {tile_size!r}")
        return tile  # type: ignore[return-value]

    def _validate_tiling(self) -> None:
        """Validate tile config against the backbone grid; warn on degenerate sizes."""
        gh, gw = self.net.spatial_grid
        if self.tile_size is not None:
            th, tw = self.tile_size
            if th % gh or tw % gw:
                raise ValueError(
                    f"tile_size {self.tile_size} must be divisible by the stride grid "
                    f"({gh}, {gw}) so tiles run without internal padding."
                )
            if min(th // gh, tw // gw) < 4:
                warnings.warn(
                    f"tile_size {self.tile_size} leaves a bottleneck below 4x4 for stride "
                    f"grid ({gh}, {gw}); InstanceNorm degrades (and errors at 1x1 in train "
                    f"mode). Use a larger tile or a shallower network.",
                    stacklevel=3,
                )
        if self.net.volumetric and self.in_channels < self.net.depth_grid:
            warnings.warn(
                f"in_channels={self.in_channels} is below the spectral stride grid "
                f"({self.net.depth_grid}); most of the depth axis will be padding. "
                f"Reduce spectral downsampling stages.",
                stacklevel=3,
            )

    # ---------------------------------------------------------------- forward

    def _should_tile(self, stage: ExecutionStage | None, h: int, w: int) -> bool:
        """Tiling gate: never in TRAIN, otherwise when the input exceeds the tile."""
        if self.tile_size is None:
            return False
        oversize = h > self.tile_size[0] or w > self.tile_size[1]
        if stage is None:
            return (not self.training) and oversize
        if stage is ExecutionStage.TRAIN:
            return False
        return oversize

    def _gauss(self, device: torch.device) -> torch.Tensor | None:
        """Cached Gaussian importance map for the configured tile on ``device``."""
        if not self.tile_gaussian or self.tile_size is None:
            return None
        key = f"{self.tile_size}|{device}"
        if key not in self._gauss_cache:
            self._gauss_cache[key] = gaussian_weight_map(self.tile_size, device)
        return self._gauss_cache[key]

    def forward(
        self, data: torch.Tensor, context: Context | None = None, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Segment a BHWC cube; return BHWC per-pixel class logits.

        Arbitrary H, W are supported: the direct path grid-pads and crops back;
        oversize inputs in eval stages run the sliding-window path.
        """
        if data.dim() != 4:
            raise ValueError(f"expected a 4-D [B, H, W, C] cube, got shape {tuple(data.shape)}")
        x = data.permute(0, 3, 1, 2).contiguous()  # BHWC -> NCHW (C = bands)
        stage = context.stage if context is not None else None

        if self._should_tile(stage, x.shape[-2], x.shape[-1]):
            assert self.tile_size is not None
            logits = sliding_window_inference(
                self.net,
                x,
                self.tile_size,
                overlap=self.tile_overlap,
                gaussian=self.tile_gaussian,
                weight_map=self._gauss(x.device),
                tile_batch=self.tile_batch,
            )
        else:
            x_p, revert = pad_hw_to_multiple(x, self.net.spatial_grid)
            logits = self.net(x_p)[revert]

        logits = logits.permute(0, 2, 3, 1).contiguous()  # -> BHWC [B, H, W, num_classes]
        return {"logits": logits}
