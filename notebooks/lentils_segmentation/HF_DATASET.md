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
- **Split** (`splits/adaclip.json`): 808 train / 148 val / 180 test frames. Supervised segmentation uses
  it **as published, with every frame** — object frames supervise the foreground, normal frames
  supervise clean background. (The anomaly-detection tutorials elsewhere in the plugin family
  instead filter train to normal frames only.) Foreground is rare: ~0.06 % of pixels.

## How the notebooks consume it

Both notebooks fetch the dataset **inline** (section 1 of `01_train.ipynb`) with
`PublicDatasets.download_dataset("industrial_fod_lentils", ...)`, which skips the download when the
folder is already on disk (`HF_TOKEN` recommended: anonymous downloads are rate-limited).

Nothing is converted. cuvis-ai-dataloader's `MultiCu3sDataModule` (`cu3s_multi`, requires the
**Cuvis C++ SDK** + `cuvis-ai-dataloader[cu3s,coco]`) reads every frame straight from its cu3s
recording, given the two files the dataset ships: **`universe.csv`** (one row per frame: recording,
index, COCO file) and **`splits/adaclip.json`** (a core `DataSplitConfig` of file-index selectors).
Each frame carries `cube [H,W,61]`, `wavelengths [61]` and `mask [H,W] int32` with the COCO
category ids rasterized from the polygons (0 = background); with two classes the losses fold every
object category into the foreground class.

Several foreground-biased crops per frame come from the crop node
(`RandomForegroundBiasedMultiCrop`, `samples_per_input` = N), which cuts N fresh crops from every
frame after one read, so the universe holds each frame once. The data module's
`samples_per_frame` gives the same crops per epoch by putting every frame into the loader N
times, which reads the whole frame once per crop.

## Storage expectations

The download is the only copy: the 15 recordings take about 57 GB. The cu3s data module keeps a
bounded number of recordings open (`max_open_sessions`); each open one holds about 0.3 to 0.6 GB of
GPU memory for the SDK.
