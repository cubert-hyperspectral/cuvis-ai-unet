"""Thin helpers for the lentils × DynUNet segmentation tutorial notebooks.

Both notebooks build the pipeline, provision the data, and read metrics **inline** (the cuvis-ai
tutorial convention); this module only holds the plugin-manifest paths, the champion reference, and
small IO / matplotlib helpers:

- ``01_train.ipynb``     — two-phase training (z-score statistics, then Dice+CE gradient training on
                           foreground-biased crops) -> a saved pipeline artifact
- ``02_inference.ipynb`` — load a trained artifact, evaluate the 180-frame test split via a
                           ``Predictor`` + ``SegMetrics.compute()``, render prediction overlays

Unlike the anomaly tutorials (train on normal frames only), supervised segmentation trains on
**all** frames — object frames supervise the foreground classes, normal frames supervise clean
background — using the dataset's published full split (808 train / 148 val / 180 test).

Data workflow
-------------
The notebooks fetch the 61-band VNIR (430–910 nm) foreign-object dataset from HuggingFace with
:class:`cuvis_ai_core.data.public_datasets.PublicDatasets`. It ships a **universe.csv** (one row per
frame: recording, index, COCO file) and **splits/adaclip.json** (a core ``DataSplitConfig`` of
file-index selectors), which cuvis-ai-dataloader's ``MultiCu3sDataModule`` (``cu3s_multi``) reads
as they are: every frame comes straight from its cu3s recording, its ``mask`` holds the COCO
category ids. Reading cu3s needs ``cuvis-ai-dataloader[cu3s,coco]`` + the Cuvis SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- config
REPO_ROOT = Path(__file__).resolve().parents[2]
UNET_MANIFEST = REPO_ROOT / "plugins.yaml"
AUGMENT_MANIFEST = REPO_ROOT / "examples" / "lentils" / "augment.yaml"

#: Trained-artifact dir the inference notebook reads by default (the train notebook's output).
DEFAULT_PIPELINE_DIR = (
    REPO_ROOT
    / "notebooks"
    / "lentils_segmentation"
    / "outputs"
    / "lentils_unet_run"
    / "trained_models"
)

#: Champion reference (2D @ 128 px, full split, 20 epochs) — what a full reproduction aims for.
CHAMPION = {"2d_128": {"fg_iou": 0.7905, "fg_dice": 0.8787, "image_auroc": 0.998}}


def resolve_pipeline(pipeline_dir: str | Path = DEFAULT_PIPELINE_DIR) -> tuple[Path, Path]:
    """Return ``(yaml_path, pt_path)`` for a trained artifact dir (a ``train`` notebook / CLI output).

    Picks the single ``*.yaml`` in ``pipeline_dir`` + its sibling ``.pt``. The trainrun path writes
    ``lentils_unet_restored.*``; the manual path ``lentils_unet_manual.*`` — pass the dir to choose.
    """
    d = Path(pipeline_dir)
    yamls = sorted(d.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No trained pipeline *.yaml in {d}. Run 01_train.ipynb first, or point PIPELINE_DIR "
            f"(the inference notebook's configuration cell) at a train.py / trained_models dir."
        )
    yaml_path = yamls[0]
    pt_path = yaml_path.with_suffix(".pt")
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing weights next to {yaml_path.name}: {pt_path}")
    return yaml_path, pt_path


# --------------------------------------------------------------------------- visualisation
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else np.clip((x - lo) / (hi - lo), 0, 1)


def false_color(
    cube_hwc: np.ndarray, wavelengths: np.ndarray, targets_nm=(650.0, 550.0, 450.0)
) -> np.ndarray:
    """Nearest-wavelength 3-channel false-color from a 61-ch cube (for display only)."""
    wl = np.asarray(wavelengths).ravel().astype(float)
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets_nm]
    return _norm(cube_hwc[..., idx])


def render_segmentation_panel(
    cube_hwc,
    fg_prob,
    pred_mask,
    *,
    wavelengths,
    gt_mask=None,
    targets_nm=(650.0, 550.0, 450.0),
    title=None,
    figsize=(16.0, 4.0),
) -> Any:
    """Per-frame story: false-color RGB, foreground probability, prediction (cyan) vs GT (red)."""
    import matplotlib.pyplot as plt

    rgb = false_color(cube_hwc, wavelengths, targets_nm)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(rgb)
    ax[0].set_title("false-color RGB")
    ax[0].axis("off")
    ax[1].imshow(_norm(fg_prob), cmap="inferno")
    ax[1].set_title("foreground probability")
    ax[1].axis("off")
    ax[2].imshow(rgb)
    ax[2].contour(np.asarray(pred_mask) > 0, levels=[0.5], colors="cyan", linewidths=1.2)
    if gt_mask is not None:
        if np.asarray(gt_mask).ndim > 2:
            gt_mask = np.squeeze(gt_mask)
        ax[2].contour(np.asarray(gt_mask) > 0, levels=[0.5], colors="red", linewidths=1.0)
    ax[2].set_title("prediction (cyan) vs GT (red)")
    ax[2].axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig
