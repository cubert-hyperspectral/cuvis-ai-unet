"""Generate a supervised train/val/test splits CSV over labeled lentils frames.

The dataset's own CSV split is normal-only (for anomaly detection); for
supervised segmentation we split the labeled frames ourselves. Small by default
so the GradientTrainer smoke over full frames stays quick.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lentils_datamodule import labeled_frames

ap = argparse.ArgumentParser()
ap.add_argument("--npz-dir", default="/mnt/data/dev/lentils_npz")
ap.add_argument("--max-frames", type=int, default=20)
ap.add_argument("--repeat", type=int, default=1,
                help="duplicate each TRAIN row N times: each duplicate is one more "
                     "independent random crop per frame per epoch (patches per frame)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lentils_seg_splits.csv"))
args = ap.parse_args()

files = labeled_frames(args.npz_dir, "mask")[: args.max_frames]
n = len(files)
n_tr, n_val = int(n * 0.6), int(n * 0.2)
rows = []
for i, f in enumerate(files):
    split = "train" if i < n_tr else ("val" if i < n_tr + n_val else "test")
    reps = args.repeat if split == "train" else 1  # val/test stay unrepeated
    for _ in range(reps):
        rows.append((split, f, i))

with open(args.out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "npz_path", "image_id"])
    w.writerows(rows)
print(f"wrote {args.out}: n={n} train={n_tr} val={n_val} test={n - n_tr - n_val}")
