# Diagnostics

Development-era scripts kept runnable for reproducing and extending the findings
behind the published lentils numbers. The **front door** for normal use is
`../lentils/` (`train.py` / `evaluate.py` / `profile_pipeline.py`); everything here takes
explicit arguments (`--splits-csv`, `--ckpt`, `--npz-dir`, ...) and defaults its
plugin manifests to the repo's own `plugins.yaml` + `../lentils/augment.yaml`.

| script | purpose |
|---|---|
| `eval_ckpt.py` | metric oracle for raw index-keyed checkpoints (fg-IoU / fg-Dice / image-AUROC) |
| `convert_ckpt.py` | one-time: raw `torch_layers.state_dict()` → canonical `load_pipeline` artifact |
| `diag_2d128.py` | short retrain + tiled-vs-direct per-frame foreground statistics dump |
| `prof_pipeline.py` | per-node timing via the built-in pipeline profiler (artifact input) |
| `prof_infer.py` | hand-rolled Norm/DynUNet timing (validated the built-in profiler's numbers) |
| `benchmark_infer.py` | CPU vs GPU(fp32) vs GPU(autocast) per-tile timing, untrained weights |
| `smoke_*.py` | end-to-end wiring smokes (trainer paths, save/reload round-trip, tiling equivalence) |

Configs: `lentils_unet_npz_aug_adaclip2d128.yaml` / `..._adaclip3d.yaml` are the
two configurations behind the published 2D-vs-3D comparison, and
`lentils_unet_binary.yaml` / `lentils_unet_npz.yaml` are the minimal smoke
configs. **The `nodes:` list order in the adaclip configs is load-bearing**: raw
diagnostic checkpoints are index-keyed against it (`2.zscore_mean` = node 2), so
reordering the list silently misassigns weights. Their `connections:` were
rewired to run Norm upstream of Augment (numerically equivalent for
running-stats z-score; the loaders hard-fail on any key mismatch).
