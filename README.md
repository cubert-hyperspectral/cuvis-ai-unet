# cuvis-ai-unet

Configurable U-Net (`DynUNet`) nodes for the [cuvis-ai](https://docs.cuvis.ai/) hyperspectral
pipeline framework, plus segmentation loss nodes.

The `DynUNet` node reimplements the dynamic-topology U-Net design of MONAI's DynUNet — network
depth and downsampling are driven by per-stage kernel/stride lists — in standalone PyTorch, with
**no MONAI dependency**. It adds a factorised 2.5D mode MONAI does not provide. Three convolution
modes select how the spectral axis is treated:

| mode     | tensor (internal) | convolution | use |
|----------|-------------------|-------------|-----|
| `2d`     | `[B, C, H, W]`    | `Conv2d`, bands as channels | fast spatial baseline |
| `2p5d`   | `[B, C, D, H, W]` | factorised (2+1)D: spatial `(1,k,k)` then spectral `(k,1,1)` | spectral-spatial, cheaper than 3D |
| `3d`     | `[B, C, D, H, W]` | full `Conv3d` over `(D,H,W)` | full volumetric spectral-spatial |

Segmentation losses (`DiceLoss`, `CrossEntropyLoss`) support binary and multiclass targets.

Status: alpha, under active development (ALL-5702).

## Inference and tiling

At inference the node runs a sliding window over each frame (training runs the direct padded
forward on the datamodule's patch crops). Full frames far exceed the training patch, so the frame
is split into `tile_size` tiles, each is passed through the backbone, and overlapping tile logits
are blended with a Gaussian importance map (nnU-Net v2 recipe) and accumulated in float32. Three
`DynUNet` knobs control the speed / quality / memory trade-off:

- **`tile_size`** — the sliding-window tile (use the training patch size, e.g. `128`). `None`
  disables tiling: every input runs the direct padded forward.
- **`tile_overlap`** (default `0.5`) — fraction of a tile shared by neighbours. Cost is linear in
  tile count (`0.0 / 0.25 / 0.5` → `1.0× / 1.7× / 3.3×` tiles). `tile_overlap: 0` with
  `tile_gaussian: false` is a ~3.3× faster fast-eval mode; on the lentils 2D model it costs
  ~1.5–2% relative foreground IoU (tile-seam effect).
- **`tile_batch`** (default `1`) — how many tiles are stacked into one backbone forward. A 128 px
  tile barely occupies a modern GPU, so the default one-tile-per-forward path is
  launch/underfill bound. `tile_batch: 16`–`32` packs tiles onto the batch axis and gives a large
  speedup for the 2D mode (~16× on an RTX 4090: 734 → 46 ms/frame at overlap 0.5) with **identical
  output** and a small memory increase. It does not help the 3D mode, which is already
  compute-bound. `tile_batch: 1` reproduces the serial path exactly, so it is a safe default.

## Layout

- `cuvis_ai_unet/blocks.py` — mode-aware convolution building blocks.
- `cuvis_ai_unet/net.py` — the pure-PyTorch U-Net module.
- `cuvis_ai_unet/tiling.py` — sliding-window inference + boundary padding (tile batching lives here).
- `cuvis_ai_unet/node/dynunet.py` — the `DynUNet` cuvis-ai node (BHWC port wrapper).
- `cuvis_ai_unet/node/losses.py` — `DiceLoss` and `CrossEntropyLoss` loss nodes.
- `plugins.yaml` — plugin manifest.
