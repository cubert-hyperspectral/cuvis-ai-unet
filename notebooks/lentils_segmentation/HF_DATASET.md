# The lentils dataset

The tutorials run on the public Cubert dataset
[`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils):
lentils on a conveyor belt, recorded with a Cubert ULTRIS XMR hyperspectral camera, with small
foreign objects placed among them.

- **Cubes**: 1000 × 1080 pixels × **61 spectral bands** (VNIR, 430–910 nm), stored as merged
  per-day `.cu3s` sessions.
- **Labels**: per-session COCO JSON with polygon annotations for 7 foreign-object categories
  (`stem_k`, `stone`, `alu_shard`, `blue_paper`, `white_paper`, `fly`, `rubber`); pixels outside
  any polygon are background (normal lentils + belt).
- **Split** (`splits.csv`): 808 train / 148 val / 180 test frames. Supervised segmentation uses
  it **as published, with every frame** — object frames supervise the foreground, normal frames
  supervise clean background. (The anomaly-detection tutorials elsewhere in the plugin family
  instead filter train to normal frames only.) Foreground is rare: ~0.06 % of pixels.

## How the notebooks consume it

Both notebooks provision the data **inline** (section 1 of `01_train.ipynb`), in two steps that each
skip when their output already exists:

1. **Fetch** the raw dataset with `PublicDatasets.download_dataset("industrial_fod_lentils", ...)`
   (`HF_TOKEN` recommended — anonymous downloads are rate-limited).
2. **Convert** the `splits.csv` frames to per-frame NPZ — `cube [H,W,61] float32`,
   `wavelengths [61]`, `mask [H,W] int32` (binary foreground, rasterized from the polygons),
   `class_mask [H,W] uint8` (category ids) — with `cuvis_ai_dataloader`'s `convert_split_manifest`
   (requires the **Cuvis C++ SDK** + `cuvis-ai-dataloader[cu3s,coco]`). It emits two artifacts:
   a **`universe.csv`** (`source, index, materialized_path`: one row per frame) and a **`splits.json`** (a core
   `DataSplitConfig` of file-index selectors), which `MultiNpzDataModule` (`npz_multi`) reads.

The train-frame multiplicity that used to be baked into the CSV is now `samples_per_frame` on the
data module (N independent foreground-biased crops per frame per epoch), so the universe holds each
frame once.

Already have converted NPZ frames and want to skip the SDK entirely?
`examples/lentils/gen_splits.py` builds the same `universe.csv` + `splits.json` from a directory of
per-frame NPZ (plus a legacy `(split, npz_path, image_id)` CSV), preserving the exact split — no
cu3s or Cuvis SDK needed.

## Storage expectations

Full-resolution frames are large: ~145 MB on disk per compressed NPZ (~263 MB decompressed),
so the full 1136-frame conversion needs roughly 165 GB plus the ~41 GB cu3s download cache.
The notebooks default to a small `SMOKE_LIMIT` (frames per split) so a first run fits in a few GB;
set `SMOKE_LIMIT = 0` to convert the full split for a champion reproduction.
