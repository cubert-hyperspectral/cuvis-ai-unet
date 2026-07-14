# cuvis-ai-unet

[![CI](https://github.com/cubert-hyperspectral/cuvis-ai-unet/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/actions/workflows/ci.yml)

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
They are planned to migrate into cuvis-ai's builtin loss library; the copies here will be removed
in a later minor release once that lands.

## Installation

The package depends only on `cuvis-ai-core`, `cuvis-ai-schemas`, and `torch`.

```bash
# from a released tag
pip install "cuvis-ai-unet @ git+https://github.com/cubert-hyperspectral/cuvis-ai-unet.git@v0.1.0"

# or editable, for development
git clone https://github.com/cubert-hyperspectral/cuvis-ai-unet.git
uv pip install -e ./cuvis-ai-unet
```

## Plugin manifest

One yaml = one plugin. Local-path form (edit-and-reload development; `path` resolves relative to
the manifest file):

```yaml
name: unet
package_name: cuvis-ai-unet
path: "."
capabilities:
  - class_name: cuvis_ai_unet.node.dynunet.DynUNet
  - class_name: cuvis_ai_unet.node.losses.DiceLoss
  - class_name: cuvis_ai_unet.node.losses.CrossEntropyLoss
```

Git-tag form (frozen, reproducible install):

```yaml
name: unet
package_name: cuvis-ai-unet
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-unet.git"
tag: "v0.1.0"
capabilities:
  - class_name: cuvis_ai_unet.node.dynunet.DynUNet
  - class_name: cuvis_ai_unet.node.losses.DiceLoss
  - class_name: cuvis_ai_unet.node.losses.CrossEntropyLoss
```

The repo root ships the local-path manifest as [plugins.yaml](plugins.yaml).

## Results — lentils foreign-object segmentation

Trained and evaluated on the public
[`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils)
dataset (1000×1080×61 cubes; 808 train / 148 val / 180 test frames; ~0.06 % foreground). Two-phase
training: normalizer statistics, then 20 epochs of Dice+CE on foreground-biased 128 px crops.

| config | fg-IoU | fg-Dice | image-AUROC |
|--------|--------|---------|-------------|
| **2D @ 128 (bands as channels)** | **0.79** | **0.88** | **0.998** |
| 3D @ 128 (volumetric) | 0.76 | 0.86 | 0.991 |

2D beats 3D on every metric while being ~5× faster per tile — the spectral 3D convolution does
not help this task. See `examples/lentils/` for the train / evaluate / profile CLIs that
reproduce this table.

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
  ~1.5–2 % relative foreground IoU (tile-seam effect).
- **`tile_batch`** (default `1`) — how many tiles are stacked into one backbone forward. A 128 px
  tile barely occupies a modern GPU, so the default one-tile-per-forward path is
  launch/underfill bound. `tile_batch: 16` packs tiles onto the batch axis and gives a large
  speedup for the 2D mode (~16× on an RTX 4090: 734 → 46 ms/frame at overlap 0.5) with **identical
  output** and a small memory increase; the shipped examples default to 16. It does not help the
  3D mode, which is already compute-bound. `tile_batch: 1` reproduces the serial path exactly.

## Prerequisites for the examples and notebooks

The **package** needs only released dependencies. The **lentils examples/notebooks** additionally
rely on:

| requirement | why | status |
|---|---|---|
| `cuvis-ai-augment` (not on PyPI) | `AugmentationCompose` + the crop transform in training | install from git; the examples ship a git-tag manifest |
| running-stats `ZScoreNormalizer` | champion config normalizes with dataset statistics | pending in [cuvis-ai PR #53](https://github.com/cubert-hyperspectral/cuvis-ai/pull/53); until it merges, train with `--normalizer persample` or run from a checkout of that branch |
| Cuvis C++ SDK + `cuvis-ai-dataloader[cu3s]` | only for converting `.cu3s` sessions to npz | `uv sync --extra notebooks` + a machine-matched `cuvis` pin (documented in the notebooks) |

## Known limitations

- **Data loading dominates end-to-end inference.** Loading one 263 MB deflate-compressed npz frame
  takes ~1 s vs ~46 ms of 2D compute (tile_batch 16). Fix directions (uncompressed/fp16 storage,
  prefetch, crop-in-dataset) are tracked in
  [cuvis-ai-dataloader#14](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/issues/14).
- Deep supervision is not implemented (multiple resolution heads complicate the output-port
  contract; add if metrics require it).

## Layout

- `cuvis_ai_unet/blocks.py` — mode-aware convolution building blocks.
- `cuvis_ai_unet/net.py` — the pure-PyTorch U-Net module.
- `cuvis_ai_unet/tiling.py` — sliding-window inference + boundary padding (tile batching lives here).
- `cuvis_ai_unet/node/dynunet.py` — the `DynUNet` cuvis-ai node (BHWC port wrapper).
- `cuvis_ai_unet/node/losses.py` — `DiceLoss` and `CrossEntropyLoss` loss nodes.
- `cuvis_ai_unet/transforms.py` — vendored `RandomForegroundBiasedCrop` (leaves once upstream ships).
- `examples/lentils/` — shared engine + `train.py` / `evaluate.py` / `profile.py` / `gen_splits.py`.
- `examples/diagnostics/` — development-era diagnostic scripts (kept runnable, see its README).
- `notebooks/lentils_segmentation/` — train + inference tutorials on the public lentils dataset.
- `plugins.yaml` — plugin manifest.

## License

Apache-2.0. See [LICENSE](LICENSE).
