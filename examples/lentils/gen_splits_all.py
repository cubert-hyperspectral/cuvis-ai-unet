"""Stratified train/val/test split over ALL local lentils frames (foreground +
normal), so the model learns both object and clean-frame appearance.

Foreground frames (mask has any nonzero) and normal frames (all-zero mask) are each
split 60/20/20 and combined, so every split contains both. Train rows are repeated
``--repeat`` times (N independent fg-biased crops per frame per epoch). Normal frames
carry an all-zero mask (valid background target); the fg-biased crop falls back to
uniform crops on them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz-dir", default="/mnt/data/dev/lentils_npz")
ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
ap.add_argument("--repeat", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "lentils_seg_splits_600.csv"))
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
if args.max_frames:
    files = files[: args.max_frames]

fg, bg = [], []
for f in files:
    z = np.load(f)
    has_fg = "mask" in z.files and bool((z["mask"] != 0).any())
    (fg if has_fg else bg).append(f)

rng = random.Random(args.seed)
rng.shuffle(fg)
rng.shuffle(bg)


def split3(lst):
    n = len(lst)
    ntr, nval = int(n * 0.6), int(n * 0.2)
    return lst[:ntr], lst[ntr:ntr + nval], lst[ntr + nval:]


rows = []
fid = 0
counts = {"train": 0, "val": 0, "test": 0}
for stratum in (fg, bg):
    tr, va, te = split3(stratum)
    for grp, name in ((tr, "train"), (va, "val"), (te, "test")):
        for f in grp:
            reps = args.repeat if name == "train" else 1
            for _ in range(reps):
                rows.append((name, f, fid))
            counts[name] += 1
            fid += 1

with open(args.out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "npz_path", "image_id"])
    w.writerows(rows)

print(f"wrote {args.out}")
print(f"  frames: {len(files)} (fg={len(fg)} normal={len(bg)})")
print(f"  split frames: train={counts['train']} val={counts['val']} test={counts['test']}")
print(f"  train rows (x{args.repeat}): {sum(1 for r in rows if r[0] == 'train')}")
