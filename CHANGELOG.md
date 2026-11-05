# Changelog

## Unreleased

### Changed

- The lentils examples (`examples/lentils/`, `examples/diagnostics/`) and the segmentation notebooks read the cu3s recordings directly with cuvis-ai-dataloader's `MultiCu3sDataModule` (`cu3s_multi`) instead of per-frame NPZ. They take the `universe.csv` and `splits/adaclip.json` the HF dataset ships; that split is the legacy segmentation split frame for frame (808 train / 148 val / 180 test). The conversion step and its NPZ copy (about 165 GB for the 1136 frames) are gone.
- `CrossEntropyLoss` and `OHEMCrossEntropyLoss` on two-class logits fold target ids above 1 into 1, keeping `ignore_index` pixels (new `functional.fold_targets`); with more classes the targets pass through unchanged, so an out-of-range id still raises. The cu3s data modules deliver the COCO category ids (0 to 7 on lentils) as the mask, so a two-class net trains on them as background vs any object, the same target the binary NPZ masks gave; binary masks are unaffected.
- `examples/lentils/train.py`: `--num-workers` defaults to 0 (the cuvis SDK reads in the training process), with new `--max-open-sessions` (default 3) and `--read-ahead` (default 1), which fit an 8 GB GPU (a laptop), and `--cuda-cubes`; every loader keeps its own open recordings, and 15 open per loader ran a 24 GB GPU out of memory within two epochs. `--dataset-crop` and `_engine.build_graph(dataset_crop=...)` are gone (an `npz_multi` crop that no dataloader release shipped). `evaluate.py` and `profile_pipeline.py` read the frames through the split's cu3s dataset.
- `examples/lentils/`: `train.py`, `evaluate.py` and `profile_pipeline.py` take `--sdk-cuda/--no-sdk-cuda` (the Cuvis SDK processes the cubes on the GPU or the CPU), and `_engine.make_datamodule` / `split_dataset` / `evaluate` / `profile` pass it through. The segmentation notebooks have an `SDK_ON_GPU` variable, and `01_train` and `train.py` name a setting for an 8 GB laptop GPU (batch 2, features 16 to 256, 128 px crops): every frame of a batch reaches the GPU whole, 264 MB, before it is cropped.
- Added `cuvis_ai_unet.transforms.RandomForegroundBiasedMultiCrop` (`samples_per_input` foreground-biased crops from every input frame) and `train.py --crops-per-frame N` / `build_graph(crops_per_frame=N)`: the loader ships each frame once and the crops multiply inside the graph, after the normalizer, where `--samples-per-frame N` reads the whole frame once per crop. A step sees `--batch` x N crops. Vendored here until cuvis-ai-augment ships it. The lentils examples use it: `train.py`'s champion recipe is `--batch 2 --crops-per-frame 4` (was `--batch 8 --samples-per-frame 4`) and `01_train` takes `CROPS_PER_FRAME = 8` with `BATCH = 2` frames per step (was `SAMPLES_PER_FRAME = 8`, `BATCH = 4`). Over 20 epochs on an RTX 4090 (`--normalizer persample`, test split of `splits/adaclip.json`) the read-once recipe reached foreground IoU 0.810 in 11716 s from cu3s, the old one 0.814 in 18250 s from NPZ; from cu3s the old recipe re-read every frame once per crop and was about twice as slow as NPZ.
- `examples/diagnostics/`: the scripts read frames through the lentils engine's cu3s data module (`--universe-csv` / `--splits-json` replace `--csv`, `--splits-csv` and `--npz-dir`; 0 workers), `smoke_npz_gradienttrainer.py` is now `smoke_cu3s_gradienttrainer.py`, and the five scripts that still built `GradientTrainer(trainer_config=..., optimizer_config=...)` with the removed `TrainerConfig` (cuvis-ai-schemas 0.8.0, cuvis-ai-core 0.11.0) take a `TrainingConfig`, so they import again. The smokes draw their frames from the train split of `splits/adaclip.json`.
- Removed `examples/lentils/gen_splits.py` and `examples/lentils/lentils_adaclip.splits.json`: they turned the legacy npz split CSV into a universe and splits pair; the HF `splits/adaclip.json` holds the same split.
- The `notebooks` extra floors `cuvis-ai-dataloader[cu3s,coco]>=0.8.2` and `examples/lentils/cuvis_ai_dataloader.yaml` moves from v0.5.0 to v0.8.2, the first release whose `cu3s_multi` keeps a bounded set of recordings open.
- The lentils examples and notebooks normalize with cuvis-ai's `Standardize`; the `notebooks` extra floors `cuvis-ai>=0.21.0` (lock 0.20.1 to 0.21.0), which removed `ZScoreNormalizer` and ships its running-statistics mode as `Standardize(stats="fit")`. `train.py --normalizer zscore`, the default, fits one mean and std on the first `--max-init-frames` train frames (`axis="global"`, the champion's z-score), `zscore-perband` fits them per band (`axis="channel"`) and `persample` standardizes every frame by itself (`axis="spatial"` with `ddof=0`, `eps=1e-6`, `eps_mode="add"`, the former per-sample normalizer bit for bit); `zscore` no longer stops with an error. `01_train` fits the global z-score as the July champion did; on cuvis-ai 0.20 its `use_running_stats` keyword was ignored and it standardized every frame by itself. `02_inference` calls the normalizer through its `cube` port. The diagnostics keep `ZScoreNormalizer` for the July checkpoints and need cuvis-ai 0.20.x; their README says so.
- The `notebooks` extra floors `cuvis-ai-core>=0.19.3` (lock 0.19.1 to 0.19.3): its post-fit steps reuse the readers the fit already holds instead of reopening every val recording next to the open ones, which with the Cuvis SDK on the GPU ran an 8 GB card out of memory.
- The lentils examples and notebooks pin cuvis-ai-augment v0.4.3 (was v0.3.2; `examples/lentils/augment.yaml`, the notebooks' Colab install, the `notebooks` extra note). v0.3.2 passes `execution_stages=` to the node constructor, which the cuvis-ai-core these examples now need (0.19.3 or later) rejects since 0.15.0. v0.4.x ships `RandomForegroundBiasedCrop` itself, so the vendored copy in `cuvis_ai_unet.transforms` stands down (its registration is guarded); with `probabilistic=True`, which the examples and notebooks use, the two crops are the same code.

## 0.5.0 - 2026-11-05

### Added

- `DynUNet.export_module()`: an ONNX export adapter for configurations without tiling, running the direct inference path (BHWC to NCHW, grid padding, the backbone, cropping, back to BHWC) with tensor ops only. A tiled configuration (`tile_size` set) raises `NotImplementedError` naming the hparam to change, since sliding-window inference branches on the input size and cannot be exported as a static graph. Tagged `NodeTag.ONNX`. Floors `cuvis-ai-core>=0.19.2`; the dev extra adds `cuvis-ai-core[onnx]` and `cuvis-ai[onnx]` for the parity tests and the deploy-folder round trip (`torch`/`torchvision` unchanged).

## 0.4.0 - 2026-11-04

### Changed

- `DynUNet` pads, tiles and blends with cuvis-ai-core's `utils.tiling` (core 0.18.5), which carries this plugin's former tiling helpers with the same behavior and shares them with the cuvis-ai nodes. Outputs are unchanged. The examples import from core. Floors `cuvis-ai-core>=0.18.5`; the lock moves cuvis-ai-core 0.14.1 to 0.18.5 and cuvis-ai-schemas 0.10.0 to 0.14.1, plus the packages they pull along.

### Removed

- `cuvis_ai_unet.tiling`. Import the helpers from `cuvis_ai_core.utils.tiling`.

## 0.3.0 - 2026-09-04

### Changed

- The segmentation losses, `SegmentationAnomalyScore` and `SegMetrics` declare their execution stages on the class (`EXECUTION_STAGES`, cuvis-ai-core 0.14.1) instead of passing `execution_stages=` to the constructor. The `execution_stages` parameter of `SegMetrics` is removed (`**kwargs` stays, so a yaml `execution_stages: null` still loads); `consume_base_kwargs` is no longer used. The profiling examples no longer assign `node.execution_stages` (read-only since core 0.14.1). Floors `cuvis-ai-core>=0.14.1`.

## 0.2.2 - 2026-08-31

### Changed
- Scoped the torch / torchvision cu128 index pin to a `cuda` dependency group (installed by
  default in this checkout), matching cuvis-ai 0.13.4 and the dataloader v0.6.2 / augment v0.4.2 /
  inspecscrap v0.2.4 / wafer-thickness v0.3.1 sweep. This corrects the 0.2.1 documentation: uv DOES
  read a git dependency's `[tool.uv.sources]`, so the previous unscoped pin reached every composed
  child environment that pulls this repo from git and collided with the host-mirrored torch index
  on non-cu128 hosts (Jetson Thor cu130, CPU-only). Consumers never install a dependency's groups,
  so the scoped pin binds nothing outside this checkout. The committed lock is now resolved with
  the group source (torch 2.11.0+cu128, the cu128 index ceiling); CI `--locked` runs drop
  `--no-sources` accordingly. On an aarch64 checkout, sync without the pin:
  `uv sync --no-default-groups`. Base dependencies audited for aarch64 wheel coverage
  (`uv pip compile --python-platform aarch64-manylinux_2_39 --only-binary :all:`): clean.

## 0.2.1 - 2026-08-20

### Changed
- Documented the torch cu128 index tables as local-development-only: installs of this package as
  a git or registry dependency never read them, the committed lock is resolved without these
  sources, and composed child environments mirror the host's torch build
  (cuvis-ai-core >= 0.12.1). Corrected the torchvision dependency comment that claimed the cu128
  override binds for dependency installs.

## [0.2.0] - 2026-08-14

### Added
- Added `DynUNet` node — a pure-PyTorch reimplementation of MONAI DynUNet's dynamic topology
  (per-stage kernel/stride lists, basic or residual double-conv blocks, InstanceNorm + LeakyReLU)
  with three spectral-axis modes: `2d` (bands as channels), `2p5d` (factorised (2+1)D, not in
  MONAI), and `3d` (full volumetric). No MONAI dependency.
- Added sliding-window tiled inference with Gaussian-blended overlaps (nnU-Net v2 recipe),
  fp16 autocast, and `tile_batch` — stacking multiple tiles into one backbone forward for up to
  ~16x faster 2D inference at identical output.
- Added `DiceLoss` and `CrossEntropyLoss` segmentation loss nodes (binary and multiclass, BHWC
  logits). These are planned to migrate into cuvis-ai's builtin loss library; the plugin copies
  will be removed in a later minor release once that lands.
- Added `OHEMCrossEntropyLoss` segmentation loss node: cross-entropy with online hard-example
  mining on the background — keeps all foreground pixels plus the hardest `clamp(ratio * n_fg,
  min_kept, n_bg)` background pixels, so the strong easy-background majority cannot average away
  the signal on hard normal pixels. Same migration path as `DiceLoss`/`CrossEntropyLoss`: it will
  move into cuvis-ai's builtin loss library alongside them once that lands (see cuvis-ai#58).
- Added `SegmentationAnomalyScore` node (`cuvis_ai_unet.node.scoring`): adapts per-pixel
  segmentation logits into anomaly-detection inputs — a foreground-probability map, a per-image
  score (mean of the top-`top_frac` score pixels, default 0.1 %), and, when an integer `targets`
  mask is connected, a boolean foreground mask for a paired AUROC node. The per-image reduction
  uses the integer-floor quantile convention `k = int((1 - top_frac) * N)`, mean of `flat[k:]`,
  identical to `cuvis_ai_rfdetr.functional.top_frac_mean`, so a segmentation and a detection head
  scored side by side use the same image-score rule. `targets` is optional so the node also serves
  inference-time thresholding; default stages are VAL/TEST/INFERENCE.
- Added `RandomForegroundBiasedCrop` transform (vendored from a pending cuvis-ai-augment change;
  removed here once it ships upstream).
- Added the lentils segmentation example suite: shared engine plus `train.py` / `evaluate.py` /
  `profile.py` CLIs (code-built pipeline, two-phase training, artifact save/restore, fg-IoU /
  fg-Dice / image-AUROC evaluation, per-node profiling) and `gen_splits.py` split generation.
- Added `SegMetrics` metric node (foreground IoU / Dice / pixel accuracy over per-pixel class
  logits, VAL/TEST stages, `compute()` aggregation over an eval pass), registered in the plugin
  manifest alongside the DynUNet and loss nodes.
- Added training conveniences to the lentils example engine: `--save-best-val` (best-`val_loss`
  checkpointing, the saved artifact is the best epoch, robust to late divergence), `--tensorboard`
  (wires SegMetrics + the cuvis-ai TensorBoard monitor into the graph), `--grad-clip`, vertical
  flips, and a guarded `--dataset-crop` for dataloaders with npz_multi crop support.
- Added executed tutorial notebooks (`notebooks/lentils_segmentation/01_train.ipynb` /
  `02_inference.ipynb`): inline data provisioning from HuggingFace, two-phase training, full
  test-split evaluation with prediction overlays.
- Added a pipeline save -> load -> forward round-trip integration test
  (`tests/test_pipeline_roundtrip.py`) that rebuilds the pipeline from its config through a fresh
  `NodeRegistry` and asserts the restored output matches.
- Added unit + integration tests (network blocks, functional losses, tiling equivalence, node
  contracts, manifest loading) and CI (lint, typecheck, security scans, coverage, build).

### Changed
- Migrated the lentils examples and notebooks to the cuvis-ai-dataloader 0.5.0 data-spec:
  `universe.csv` (`source, index, materialized_path`) + committable `splits.json` selectors
  replace the legacy `(split, npz_path, image_id)` CSV; `gen_splits.py` now converts a legacy CSV
  into both artifacts, and patches-per-frame multiplicity moved to the datamodule's
  `samples_per_frame`.
- Raised dependency floors to the released compatible pair `cuvis-ai-core>=0.11.0` /
  `cuvis-ai-schemas>=0.8.0`; the notebooks extra now floors
  `cuvis-ai-dataloader[cu3s,coco]>=0.5.0` and includes `cuvis-ai` (builtin nodes the tutorials
  wire).

<!--
Conventions (Keep-a-Changelog flavour used across the cuvis-ai plugin family):

  - The Unreleased heading is always bracketed: `## [Unreleased]`.
  - Released versions use plain `## X.Y.Z - YYYY-MM-DD` (no brackets, ISO date).
    The release workflow greps for this heading to extract release notes;
    `## X.Y.Z` (no date) and `## [X.Y.Z]` (bracketed) will both fail the extraction.
  - Terse past-tense bullets: "Added X.", "Fixed Y." Not "Add X.", not "Adding X.".
  - No "All notable changes to this project..." preamble. The heading shape and
    Keep-a-Changelog convention are self-explanatory.
  - No bold severity prefixes (`**Fix (critical):**`). Severity is communicated
    through the wording of the bullet.
  - Section order within a version: Added, Changed, Fixed, Removed.
    Omit sections that have no bullets before tagging.
-->
