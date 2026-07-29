# Changelog

## [Unreleased]

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
