"""Generate a train/val/test splits CSV over local lentils npz frames.

Two schemes:

- ``stratified-all`` (default): every frame participates; foreground frames
  (mask has any nonzero pixel) and normal frames are each split by ``--fracs``
  and combined, so the model sees both object and clean-frame appearance.
  Normal frames carry an all-zero mask (a valid background target); the
  fg-biased crop falls back to uniform crops on them.
- ``labeled-only``: only frames with foreground, for quick supervised smokes.

Train rows are repeated ``--repeat`` times: each duplicate is one more
independent fg-biased crop of that frame per epoch (the patches-per-frame
multiplicity used by the champion runs). Val/test rows are never repeated.

    python gen_splits.py --npz-dir /data/lentils_npz --repeat 4 --out splits.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True, help="directory of per-frame .npz files (cube + mask)")
    ap.add_argument("--out", required=True, help="output CSV path (split,npz_path,image_id)")
    ap.add_argument("--scheme", default="stratified-all", choices=["stratified-all", "labeled-only"])
    ap.add_argument("--fracs", type=float, nargs=2, default=[0.6, 0.2], metavar=("TRAIN", "VAL"),
                    help="train/val fractions; the remainder is test")
    ap.add_argument("--repeat", type=int, default=4,
                    help="duplicate each TRAIN row N times (N independent crops per frame per epoch)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if args.max_frames:
        files = files[: args.max_frames]
    if not files:
        raise SystemExit(f"no .npz files under {args.npz_dir}")

    fg, bg = [], []
    for f in files:
        z = np.load(f)
        has_fg = "mask" in z.files and bool((z["mask"] != 0).any())
        (fg if has_fg else bg).append(f)

    rng = random.Random(args.seed)
    rng.shuffle(fg)
    rng.shuffle(bg)
    strata = [fg] if args.scheme == "labeled-only" else [fg, bg]

    def split3(lst: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(lst)
        ntr, nval = int(n * args.fracs[0]), int(n * args.fracs[1])
        return lst[:ntr], lst[ntr : ntr + nval], lst[ntr + nval :]

    rows = []
    fid = 0
    counts = {"train": 0, "val": 0, "test": 0}
    for stratum in strata:
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
    print(f"  frames: {len(files)} (fg={len(fg)} normal={len(bg)}, scheme={args.scheme})")
    print(f"  split frames: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"  train rows (x{args.repeat}): {sum(1 for r in rows if r[0] == 'train')}")


if __name__ == "__main__":
    main()
