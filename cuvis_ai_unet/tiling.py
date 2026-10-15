"""Deprecated: the padding and sliding-window helpers live in ``cuvis_ai_core.utils.tiling``.

cuvis-ai-core 0.18.5 carries this module's former code unchanged (centred zero padding, nnU-Net
tile offsets, the Gaussian blend map and ``sliding_window_inference``), shared with the cuvis-ai
nodes. This module re-exports it for one release; import from ``cuvis_ai_core.utils.tiling``.
"""

from __future__ import annotations

import warnings

from cuvis_ai_core.utils.tiling import (
    _pad_hw_centered,
    compute_tile_offsets,
    gaussian_weight_map,
    pad_hw_to_min,
    pad_hw_to_multiple,
    sliding_window_inference,
)

warnings.warn(
    "cuvis_ai_unet.tiling is deprecated; import the helpers from cuvis_ai_core.utils.tiling.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "_pad_hw_centered",
    "compute_tile_offsets",
    "gaussian_weight_map",
    "pad_hw_to_min",
    "pad_hw_to_multiple",
    "sliding_window_inference",
]
