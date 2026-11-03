"""Lentils cu3s data loading for the cuvis-ai-unet segmentation smoke.

Experiment glue — NOT part of the ``cuvis_ai_unet`` package. Reads the labeled
lentils frames from their cu3s recordings, crops foreground-biased patches,
z-score normalises per band, and yields samples keyed by pipeline input-port
names: ``data`` (BHWC cube) and ``targets`` (BHW1 integer mask). cuvis-ai's
``pipeline.forward`` distributes batch keys to any node whose INPUT_SPECS
declares that port, so ``data`` feeds the DynUNet node and ``targets`` feeds
the loss nodes with no explicit source node.

The frames come from one split of the HF lentils download (its ``universe.csv``
and ``splits/adaclip.json``). The smokes take the first labeled frames of that
split and hold the last of them out as validation themselves. Checking a frame
for foreground reads it, cube included, so the frame-count arguments cap how
many frames are read.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# The lentils examples' engine (examples/lentils/_engine.py) resolves the splits.
LENTILS_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lentils"
)


def split_dataset(universe_csv: str, splits_json: str, split: str) -> Dataset:
    """Return one split of the lentils cu3s recordings as a dataset.

    ``dataset[i]`` reads frame i from its recording: ``cube`` ``[H, W, C]`` and the
    ``mask`` ``[H, W]`` of COCO category ids (0 is background).
    """
    if LENTILS_EXAMPLES not in sys.path:
        sys.path.insert(0, LENTILS_EXAMPLES)
    import _engine

    return _engine.split_dataset(universe_csv, splits_json, split)


def labeled_frames(dataset: Dataset, max_frames: int | None = None) -> list[int]:
    """Return the indices of ``dataset`` frames whose mask contains any foreground.

    Checking a frame reads it, so the scan stops once ``max_frames`` are found.
    """
    out: list[int] = []
    for i in range(len(dataset)):
        if max_frames and len(out) >= max_frames:
            break
        if bool((np.asarray(dataset[i]["mask"]) != 0).any()):
            out.append(i)
    return out


class LentilsPatchDataset(Dataset):
    """Foreground-biased patches precomputed from labeled lentils frames.

    Each frame is read once at construction; ``per_frame`` patches centred on
    random foreground pixels are cropped and z-score normalised, then cached as
    small arrays (so __getitem__ never rereads a 260 MB cube). ``target`` is
    ``binary`` (background vs any object, ``mask > 0``) or ``multiclass`` (the
    category ids of the cu3s ``mask``).
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        target: str,
        patch: int,
        per_frame: int,
        seed: int = 0,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for fi, i in enumerate(indices):
            frame = dataset[i]
            cube = np.asarray(frame["cube"], dtype=np.float32)  # [H, W, C]
            mask = np.asarray(frame["mask"])  # [H, W] category ids
            if target == "binary":
                mask = mask > 0
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
                    (
                        np.ascontiguousarray(cp, dtype=np.float32),
                        np.ascontiguousarray(mp).astype(np.int64),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cube, mask = self.samples[idx]
        return {"data": torch.from_numpy(cube), "targets": torch.from_numpy(mask).unsqueeze(-1)}


def split_labeled(frames: list[int], val_frac: float) -> tuple[list[int], list[int]]:
    """Deterministic train/val split of labeled frames (the last ``val_frac`` is val)."""
    n_val = max(1, int(len(frames) * val_frac))
    return frames[:-n_val], frames[-n_val:]


try:
    from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule

    class LentilsDataModule(BaseCuvisAIDataModule):
        """cuvis-ai datamodule wrapper (used by the GradientTrainer path)."""

        DATA_MODULE_NAME = "lentils_seg"

        def __init__(
            self,
            *,
            universe_csv: str,
            splits_json: str,
            split: str = "train",
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
            self.universe_csv = universe_csv
            self.splits_json = splits_json
            self.split = split
            self.target = target
            self.patch = patch
            self.per_frame = per_frame
            self.max_frames = max_frames
            self.val_frac = val_frac
            self._frames: tuple[Dataset, list[int], list[int]] | None = None

        def build_stage_dataset(self, stage: str) -> Dataset:
            """Build the patch dataset for a Lightning stage."""
            if self._frames is None:
                # Scan once: every stage crops from the same labeled frames.
                dataset = split_dataset(self.universe_csv, self.splits_json, self.split)
                train, val = split_labeled(labeled_frames(dataset, self.max_frames), self.val_frac)
                self._frames = (dataset, train, val)
            dataset, train, val = self._frames
            is_train = stage in ("train", "fit")
            return LentilsPatchDataset(
                dataset,
                train if is_train else val,
                self.target,
                self.patch,
                self.per_frame,
                seed=0 if is_train else 1,
            )

except Exception:  # noqa: BLE001 — cuvis_ai_core absent on a bare authoring box
    LentilsDataModule = None  # type: ignore[assignment,misc]
