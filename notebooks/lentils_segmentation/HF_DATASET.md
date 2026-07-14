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

`utils.ensure_lentils_npz` (hf mode, the default):

1. downloads `splits.csv` and the referenced `.cu3s` + COCO files from HuggingFace
   (`HF_TOKEN` recommended — anonymous downloads are rate-limited),
2. converts each needed frame to a per-frame NPZ — `cube [H,W,61] float32`,
   `wavelengths [61]`, `mask [H,W] int32` (binary foreground, rasterized from the polygons),
   `class_mask [H,W] uint8` (category ids) — via `cuvis_ai_dataloader`'s converter
   (requires the **Cuvis C++ SDK** and a matching `cuvis` Python pin), and
3. writes the `(split, npz_path, image_id)` CSV that `MultiNpzDataModule` reads, with the train
   rows repeated `repeat` times (the patches-per-frame multiplicity).

Already have converted NPZ frames? Skip the SDK path entirely with:

```bash
export LENTILS_DATA_SOURCE=local
export LENTILS_SPLITS_CSV=/path/to/lentils_seg_splits.csv
```

`examples/lentils/gen_splits.py` builds such a CSV from any directory of per-frame NPZs.

## Storage expectations

Full-resolution frames are large: ~145 MB on disk per compressed NPZ (~263 MB decompressed),
so the full 1136-frame conversion needs roughly 165 GB plus the ~41 GB cu3s download cache.
The tutorials default to a small `limit` so a first run fits in a few GB; scale up for champion
reproductions.
