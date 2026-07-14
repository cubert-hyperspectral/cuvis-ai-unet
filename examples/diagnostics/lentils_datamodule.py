"""Lentils npz data loading for the cuvis-ai-unet segmentation smoke.

Experiment glue — NOT part of the ``cuvis_ai_unet`` package. Reads the labeled
lentils npz frames, crops foreground-biased patches, z-score normalises per
band, and yields samples keyed by pipeline input-port names: ``data`` (BHWC
cube) and ``targets`` (BHW1 integer mask). cuvis-ai's ``pipeline.forward``
distributes batch keys to any node whose INPUT_SPECS declares that port, so
``data`` feeds the DynUNet node and ``targets`` feeds the loss nodes with no
explicit source node.

The lentils split CSV is built for normal-only anomaly detection (train = normal
frames); for supervised segmentation we ignore it and split the labeled frames
ourselves.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Directory of per-frame .npz files; callers pass it explicitly (or set
# LENTILS_NPZ_DIR when using the datamodule wrapper's default).
NPZ_DIR = os.environ.get("LENTILS_NPZ_DIR", "./lentils_npz")


def labeled_frames(npz_dir: str, target_key: str) -> list[str]:
    """Return npz paths whose ``target_key`` mask contains any foreground."""
    out = []
    for f in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        z = np.load(f)
        if target_key in z.files and bool((z[target_key] != 0).any()):
            out.append(f)
    return out


class LentilsPatchDataset(Dataset):
    """Foreground-biased patches precomputed from labeled lentils frames.

    Each frame is loaded once at construction; ``per_frame`` patches centred on
    random foreground pixels are cropped and z-score normalised, then cached as
    small arrays (so __getitem__ never reloads a 260 MB cube).
    """

    def __init__(
        self, files: list[str], target_key: str, patch: int, per_frame: int, seed: int = 0
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for fi, f in enumerate(files):
            z = np.load(f)
            cube = z["cube"].astype(np.float32)  # [H, W, C]
            mask = np.asarray(z[target_key])  # [H, W]
            h, w, c = cube.shape
            ys, xs = np.nonzero(mask)
            rng = np.random.default_rng(seed * 100003 + fi)
            for _ in range(per_frame):
                if len(ys):
                    j = int(rng.integers(len(ys)))
                    cy, cx = int(ys[j]), int(xs[j])
                else:
                    cy, cx = int(rng.integers(h)), int(rng.integers(w))
                y0 = int(np.clip(cy - patch // 2, 0, max(0, h - patch)))
                x0 = int(np.clip(cx - patch // 2, 0, max(0, w - patch)))
                cp = cube[y0 : y0 + patch, x0 : x0 + patch, :]
                mp = mask[y0 : y0 + patch, x0 : x0 + patch]
                mu = cp.reshape(-1, c).mean(0)
                sd = cp.reshape(-1, c).std(0) + 1e-6
                cp = (cp - mu) / sd
                self.samples.append(
                    (np.ascontiguousarray(cp, dtype=np.float32), np.ascontiguousarray(mp).astype(np.int64))
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cube, mask = self.samples[idx]
        return {"data": torch.from_numpy(cube), "targets": torch.from_numpy(mask).unsqueeze(-1)}


def split_labeled(npz_dir: str, target_key: str, max_frames: int | None, val_frac: float):
    """Deterministic train/val split over labeled frames (ignores the CSV split)."""
    files = labeled_frames(npz_dir, target_key)
    if max_frames:
        files = files[:max_frames]
    n_val = max(1, int(len(files) * val_frac))
    return files[:-n_val], files[-n_val:]


try:
    from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule

    class LentilsDataModule(BaseCuvisAIDataModule):
        """cuvis-ai datamodule wrapper (used by the GradientTrainer path)."""

        DATA_MODULE_NAME = "lentils_seg"

        def __init__(
            self,
            *,
            npz_dir: str = NPZ_DIR,
            target: str = "binary",
            patch: int = 128,
            per_frame: int = 4,
            max_frames: int | None = None,
            val_frac: float = 0.25,
            batch_size: int = 2,
            num_workers: int = 0,
            **kwargs: Any,
        ) -> None:
            super().__init__(batch_size=batch_size, num_workers=num_workers, **kwargs)
            self.npz_dir = npz_dir
            self.target_key = "mask" if target == "binary" else "class_mask"
            self.patch = patch
            self.per_frame = per_frame
            self.max_frames = max_frames
            self.val_frac = val_frac

        def build_stage_dataset(self, stage: str) -> Dataset:
            """Build the patch dataset for a Lightning stage."""
            train, val = split_labeled(self.npz_dir, self.target_key, self.max_frames, self.val_frac)
            is_train = stage in ("train", "fit")
            return LentilsPatchDataset(
                train if is_train else val, self.target_key, self.patch, self.per_frame,
                seed=0 if is_train else 1,
            )

except Exception:  # noqa: BLE001 — cuvis_ai_core absent on a bare authoring box
    LentilsDataModule = None  # type: ignore[assignment,misc]
