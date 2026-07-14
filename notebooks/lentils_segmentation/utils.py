"""Shared helpers for the lentils x DynUNet segmentation tutorial notebooks.

Two notebooks ride on these helpers, both on the ``cuvis-ai-unet`` plugin
(``cuvis_ai_unet`` / ``DynUNet`` + ``DiceLoss`` + ``CrossEntropyLoss``) and the
shared example engine in ``examples/lentils/_engine.py``:

- ``01_train.ipynb``     — two-phase training (normalizer statistics, then
                           gradient training on fg-biased crops) -> saved artifact
- ``02_inference.ipynb`` — load a trained artifact, evaluate the 180-frame test
                           split (fg-IoU / fg-Dice / image-AUROC), prediction overlays

Unlike the anomaly tutorials (train on normal frames only), supervised
segmentation trains on **all** frames — object frames supervise the foreground
classes, normal frames supervise clean background — using the dataset's
published full split (808 train / 148 val / 180 test).

Lentils dataset
---------------
61-channel VNIR (430-910 nm) foreign-object detection, published on HuggingFace
at ``cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`` (merged
``.cu3s`` sessions per day + per-session COCO polygons). ``ensure_lentils_npz``
downloads the cu3s, converts to per-frame NPZ (mask rasterized from COCO), and
writes the ``(split, npz_path, image_id)`` CSV that ``MultiNpzDataModule``
reads; ``repeat`` bakes the patches-per-frame multiplicity into the train rows.

``LENTILS_DATA_SOURCE`` selects the data path: ``hf`` (default) downloads +
converts on demand; ``local`` reads pre-existing NPZ from the CSV in
``LENTILS_SPLITS_CSV``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- config
REPO_ROOT = Path(__file__).resolve().parents[2]
UNET_MANIFEST = REPO_ROOT / "plugins.yaml"
AUGMENT_MANIFEST = REPO_ROOT / "examples" / "lentils" / "augment.yaml"
ENGINE_DIR = REPO_ROOT / "examples" / "lentils"

LENTILS_HF_REPO_ID = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"
LENTILS_HF_CACHE = Path(os.environ.get("LENTILS_HF_CACHE", str(Path.home() / ".cache" / "cuvis_lentils")))

#: "hf" (default) downloads cu3s from HF + converts to NPZ on demand; "local" reads pre-existing
#: NPZ from the CSV in ``LENTILS_SPLITS_CSV``. Env-overridable.
LENTILS_DATA_SOURCE = os.environ.get("LENTILS_DATA_SOURCE", "hf").lower()

#: A local (split, npz_path, image_id) CSV — only used in ``local`` mode; set it via the
#: ``LENTILS_SPLITS_CSV`` env var (no default path).
LOCAL_SPLITS_CSV = Path(os.environ["LENTILS_SPLITS_CSV"]) if os.environ.get("LENTILS_SPLITS_CSV") else None
#: Where HF cu3s are downloaded + converted to NPZ (hf mode).
LENTILS_NPZ_OUT = LENTILS_HF_CACHE / "npz"

#: Trained-artifact dir the inference notebook reads. Defaults to the train notebook's output;
#: set ``LENTILS_PIPELINE_DIR`` to evaluate another run (e.g. the champion reproduction).
LOCAL_PIPELINE_DIR = Path(
    os.environ.get(
        "LENTILS_PIPELINE_DIR",
        str(REPO_ROOT / "notebooks" / "lentils_segmentation" / "outputs" / "unet2d"),
    )
)

#: COCO category id -> name (per-session COCO; 0 = background/unlabeled normal lentils + belt).
LENTILS_CATEGORIES: dict[int, str] = {
    0: "Unlabeled", 1: "stem_k", 2: "stone", 3: "alu_shard",
    4: "blue_paper", 5: "white_paper", 6: "fly", 7: "rubber",
}

#: Champion reference (2D @ 128 px, full split, 20 epochs) — what a full reproduction aims for.
CHAMPION = {"2d_128": {"fg_iou": 0.7905, "fg_dice": 0.8787, "image_auroc": 0.998}}


def import_engine():
    """Import the shared example engine (registers no plugins by itself)."""
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    import _engine as eng  # noqa: PLC0415

    return eng


def resolve_config() -> dict[str, Any]:
    """Notebook-time config (no downloads). Asserts the plugin manifests (ship in the repo)."""
    cfg = {
        "data_source": LENTILS_DATA_SOURCE,
        "hf_repo_id": LENTILS_HF_REPO_ID,
        "unet_manifest": UNET_MANIFEST,
        "augment_manifest": AUGMENT_MANIFEST,
        "splits_csv": LOCAL_SPLITS_CSV,
        "npz_out": LENTILS_NPZ_OUT,
        "local_pipeline_dir": LOCAL_PIPELINE_DIR,
        "categories": LENTILS_CATEGORIES,
    }
    assert UNET_MANIFEST.exists(), (
        f"Plugin manifest not found at {UNET_MANIFEST}. Run from inside the cuvis-ai-unet repo."
    )
    if LENTILS_DATA_SOURCE == "local" and (LOCAL_SPLITS_CSV is None or not LOCAL_SPLITS_CSV.is_file()):
        raise FileNotFoundError(
            "LENTILS_DATA_SOURCE='local' needs LENTILS_SPLITS_CSV pointing at a "
            f"(split, npz_path, image_id) CSV; got {LOCAL_SPLITS_CSV!r}. Use the default "
            "'hf' source to download + convert from HuggingFace instead."
        )
    return cfg


def resolve_pipeline(pipeline_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Return ``(yaml_path, pt_path)`` for a trained artifact dir (train.py / notebook output)."""
    d = Path(pipeline_dir) if pipeline_dir is not None else LOCAL_PIPELINE_DIR
    yamls = sorted(d.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No trained pipeline *.yaml in {d}. Run the train notebook first, or set "
            f"LENTILS_PIPELINE_DIR to a train.py output dir."
        )
    yaml_path = yamls[0]
    pt_path = yaml_path.with_suffix(".pt")
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing weights next to {yaml_path.name}: {pt_path}")
    return yaml_path, pt_path


# --------------------------------------------------------------------------- data
def repeat_train_rows(csv_in: str | Path, csv_out: str | Path, repeat: int) -> Path:
    """Duplicate each TRAIN row ``repeat`` times (N independent fg-biased crops per frame
    per epoch); val/test rows pass through once. Returns ``csv_out``."""
    import csv as _csv

    with open(csv_in, newline="") as f:
        rows = list(_csv.DictReader(f))
    out_rows = []
    for r in rows:
        reps = repeat if r.get("split") == "train" else 1
        out_rows.extend([r] * reps)
    with open(csv_out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["split", "npz_path", "image_id"])
        w.writeheader()
        w.writerows(out_rows)
    return Path(csv_out)


def subsample_splits_csv(csv_path: str | Path, n_per_split: int, out_path: str | Path) -> Path:
    """Write a splits CSV keeping at most ``n_per_split`` unique frames per split (fast dry-runs)."""
    import csv as _csv
    from collections import defaultdict

    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty splits CSV: {csv_path}")
    kept: list[dict[str, str]] = []
    seen_frames: dict[str, set] = defaultdict(set)
    for r in rows:
        s = r.get("split", "")
        if s not in ("train", "val", "test"):
            continue
        key = r.get("npz_path") or r.get("local_image_id") or str(len(kept))
        if key in seen_frames[s] or len(seen_frames[s]) < n_per_split:
            kept.append(r)
            seen_frames[s].add(key)
    with open(out_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    return Path(out_path)


def _download_source_splits(cache: str) -> tuple[list[dict], str]:
    """Return (rows, which) for the published FULL split.

    Supervised segmentation uses every frame — object frames supervise the
    foreground, normal frames supervise clean background — so the published
    ``splits.csv`` is used as-is (this is the 808/148/180 split behind the
    champion numbers). Rows carry at least ``split``, ``cu3s_path``,
    ``json_path``, ``local_image_id``.
    """
    import csv as _csv

    from huggingface_hub import hf_hub_download

    p = hf_hub_download(LENTILS_HF_REPO_ID, repo_type="dataset", filename="splits.csv", cache_dir=cache)
    with open(p, newline="") as f:
        rows = [r for r in _csv.DictReader(f) if r.get("split") in ("train", "val", "test")]
    return rows, "splits.csv (full split, all frames)"


def ensure_lentils_npz(
    out_dir: str | Path, *, limit: int = 0, repeat: int = 1, splits_csv: str | Path | None = None
) -> Path:
    """Download the lentils cu3s + per-session COCO from HF, convert to per-frame NPZ.

    Returns a splits CSV ``(split, npz_path, image_id)`` that ``MultiNpzDataModule`` reads.

    Each per-session ``.cu3s`` is indexed by its measurement index = ``local_image_id`` (0..N-1),
    which is also the COCO ``image_id`` in that session's sibling JSON, so frames are read + labeled
    by ``local_image_id``. Each cu3s is downloaded once and its needed frames converted together.

    Parameters
    ----------
    limit
        If > 0, keep at most this many frames per split (fast dry-run / HF-path smoke test).
    repeat
        Duplicate each TRAIN row this many times in the output CSV — the patches-per-frame
        multiplicity used by the champion runs (val/test rows stay single).
    splits_csv
        Use this pre-filtered HF split CSV instead of auto-resolving (must carry ``cu3s_path``,
        ``json_path``, ``local_image_id``, ``split``).

    Notes
    -----
    Requires ``cuvis-ai-dataloader[cu3s]`` and the matching **cuvis SDK** (cu3s reading is native).
    The converter is imported lazily so the pure-``local`` path has no cu3s/SDK dependency.
    """
    import csv as _csv
    from collections import OrderedDict, defaultdict

    from cuvis_ai_dataloader.data.npz_converter import convert_cu3s_file
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = str(LENTILS_HF_CACHE)

    if splits_csv is not None:
        with open(splits_csv, newline="") as f:
            rows = [r for r in _csv.DictReader(f) if r.get("split") in ("train", "val", "test")]
        which = str(splits_csv)
    else:
        rows, which = _download_source_splits(cache)
    print(f"[data] using HF split: {which} ({len(rows)} frames)", flush=True)

    if limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in rows:
            s = r["split"]
            if seen[s] < limit:
                kept.append(r)
                seen[s] += 1
        rows = kept
        print(f"[data] limit={limit}/split -> {len(rows)} frames", flush=True)

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        groups.setdefault(r["cu3s_path"], []).append(r)

    out_rows: list[dict[str, Any]] = []
    for gi, (cu3s_rel, grp) in enumerate(groups.items(), 1):
        json_rel = (grp[0].get("json_path") or "").strip()
        print(f"[data] cu3s {gi}/{len(groups)}: {cu3s_rel} ({len(grp)} frames)", flush=True)
        cu3s = hf_hub_download(LENTILS_HF_REPO_ID, repo_type="dataset", filename=cu3s_rel, cache_dir=cache)
        coco = (
            hf_hub_download(LENTILS_HF_REPO_ID, repo_type="dataset", filename=json_rel, cache_dir=cache)
            if json_rel
            else None  # no sibling COCO -> convert without masks (frames read as normal)
        )
        recs = convert_cu3s_file(
            cu3s, out_dir, annotation_json=coco,
            frame_indices=[int(r["local_image_id"]) for r in grp],
        )
        for r, rec in zip(grp, recs, strict=False):
            out_rows.append({"split": r["split"], "npz_path": rec["npz_path"], "image_id": rec["image_id"]})

    csv_plain = out_dir / "lentils_seg_splits_fromhf.csv"
    with open(csv_plain, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["split", "npz_path", "image_id"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"[data] wrote {csv_plain} ({len(out_rows)} frames)", flush=True)
    if repeat > 1:
        csv_rep = out_dir / f"lentils_seg_splits_fromhf_x{repeat}.csv"
        repeat_train_rows(csv_plain, csv_rep, repeat)
        print(f"[data] wrote {csv_rep} (train rows x{repeat})", flush=True)
        return csv_rep
    return csv_plain


def load_lentils_frame(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a per-frame NPZ -> ``{cube [H,W,C] f32, wavelengths [C] i32, mask [H,W] i32,
    class_mask [H,W] u8}`` (mask/class_mask zeros when the frame is normal / unlabeled)."""
    with np.load(npz_path) as z:
        cube = np.asarray(z["cube"], dtype=np.float32)
        wl = np.asarray(z["wavelengths"]).ravel().astype(np.int32, copy=False)
        h, w = cube.shape[0], cube.shape[1]
        mask = np.asarray(z["mask"], np.int32) if "mask" in z.files else np.zeros((h, w), np.int32)
        class_mask = (
            np.asarray(z["class_mask"], np.uint8) if "class_mask" in z.files
            else np.zeros((h, w), np.uint8)
        )
    return {"cube": cube, "wavelengths": wl, "mask": mask, "class_mask": class_mask}


# --------------------------------------------------------------------------- visualisation
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else np.clip((x - lo) / (hi - lo), 0, 1)


def false_color(cube_hwc: np.ndarray, wavelengths: np.ndarray, targets_nm=(650.0, 550.0, 450.0)) -> np.ndarray:
    """Nearest-wavelength 3-channel false-color from a 61-ch cube (for display only)."""
    wl = np.asarray(wavelengths).ravel().astype(float)
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets_nm]
    return _norm(cube_hwc[..., idx])


def render_segmentation_panel(cube_hwc, fg_prob, pred_mask, *, wavelengths, gt_mask=None,
                              targets_nm=(650.0, 550.0, 450.0), title=None, figsize=(16.0, 4.0)) -> Any:
    """Per-frame story: false-color RGB, foreground probability, prediction vs GT contours."""
    import matplotlib.pyplot as plt

    rgb = false_color(cube_hwc, wavelengths, targets_nm)
    n = 3
    fig, ax = plt.subplots(1, n, figsize=figsize)
    ax[0].imshow(rgb); ax[0].set_title("false-color RGB"); ax[0].axis("off")
    ax[1].imshow(_norm(fg_prob), cmap="inferno"); ax[1].set_title("foreground probability"); ax[1].axis("off")
    ax[2].imshow(rgb)
    ax[2].contour(np.asarray(pred_mask) > 0, levels=[0.5], colors="cyan", linewidths=1.2)
    if gt_mask is not None:
        if np.asarray(gt_mask).ndim > 2:
            gt_mask = np.squeeze(gt_mask)
        ax[2].contour(np.asarray(gt_mask) > 0, levels=[0.5], colors="red", linewidths=1.0)
    ax[2].set_title("prediction (cyan) vs GT (red)"); ax[2].axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig
