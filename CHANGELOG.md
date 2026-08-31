# Changelog

## [Unreleased]

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
