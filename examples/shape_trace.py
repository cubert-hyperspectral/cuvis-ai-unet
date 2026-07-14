"""Trace how the DynUNet stack handles arbitrary input sizes (pure torch).

The backbone follows the MONAI-DynUNet contract: spatial sizes must be
divisible by the per-axis product of the stage strides, otherwise it raises.
Arbitrary sizes are handled at the boundary, nnU-Net style — constant-0 pad to
the grid, forward, crop the logits back — which is what the DynUNet node does
automatically. This script demonstrates all three behaviours plus the
sliding-window tile layout.
"""

from __future__ import annotations

import torch

from cuvis_ai_unet.net import DynUNetBackbone
from cuvis_ai_unet.tiling import compute_tile_offsets, pad_hw_to_multiple


def trace(mode: str, bands: int, h: int, w: int, **kw) -> None:
    """Show strict behaviour and the boundary pad->forward->crop for one input."""
    net = DynUNetBackbone(mode, in_channels=bands, out_channels=3, features=(8, 16, 32), **kw).eval()
    gh, gw = net.spatial_grid
    print(f"\n=== mode={mode} input [1,{bands},{h},{w}]  grid=({gh},{gw}) depth_grid={net.depth_grid} ===")
    x = torch.zeros(1, bands, h, w)
    with torch.no_grad():
        if h % gh == 0 and w % gw == 0:
            print(f"  divisible -> direct forward        OUTPUT {tuple(net(x).shape)}")
            return
        try:
            net(x)
        except ValueError as e:
            print(f"  strict backbone raises: {str(e)[:84]}...")
        xp, revert = pad_hw_to_multiple(x, net.spatial_grid)
        print(f"  boundary pad {h}x{w} -> {xp.shape[-2]}x{xp.shape[-1]} (constant 0, centered)")
        out = net(xp)[revert]
        print(f"  forward + crop back                 OUTPUT {tuple(out.shape)}  (== input H,W)")


if __name__ == "__main__":
    trace("2d", 8, 64, 64)                                # even, divisible
    trace("2d", 8, 49, 41)                                # odd -> strict raise + pad/crop
    trace("3d", 8, 50, 49)                                # mixed parity, volumetric
    trace("3d", 61, 48, 40, spectral_downsample=True)     # 61 bands: depth padded internally

    print("\n=== sliding-window offsets (nnU-Net: first 0, last flush, even) ===")
    for size, tile in [(1000, 128), (1080, 128), (256, 128), (128, 128)]:
        o = compute_tile_offsets(size, tile, overlap=0.5)
        print(f"  size {size:>5} tile {tile}: {len(o):>3} tiles  first={o[0]} last={o[-1]} (flush={size - tile})")
