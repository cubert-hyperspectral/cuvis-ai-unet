"""Convert a legacy lentils split CSV into the dataloader data-spec.

The multi-npz datamodule (``npz_multi``) is driven by two artifacts:

* a ``universe.csv`` (``source, index, path``): one row per frame, mapping the
  logical identity ``(source, index)`` to the physical ``.npz``.  ``source`` is
  the cu3s origin baked into each npz (``source_cu3s``), normalized to its
  dataset-relative form (the path tail after ``/data/``) so selectors are clean
  and machine-independent; ``index`` is the read position (== COCO image_id).
* a committable ``splits.json`` (a ``DataSplitConfig``): per-source
  ``FILE_INDICES`` selectors that pick each stage's frames by ``(source, index)``.

This tool ingests the legacy ``(split, npz_path, image_id)`` CSV and emits both,
preserving the *exact* split assignment. Legacy train rows may be repeated (the
old patches-per-frame trick); that multiplicity is now ``samples_per_frame`` at
the datamodule, so the universe holds unique frames only. Selector construction
reuses the dataloader's own ``selectors_from_refs`` so the result is byte-for-byte
what ``resolve-splits`` would produce from the equivalent cu3s CSV.

    python gen_splits.py --from-legacy-csv lentils_seg_splits_adaclip.csv \
        --out-universe lentils_universe.csv \
        --out-splits lentils_adaclip.splits.json

Regenerate the machine-specific ``universe.csv`` on any host with the same npz;
commit only the portable ``splits.json``.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np

_STAGES = ("train", "val", "test")


def _normalize_source(source_cu3s: str) -> str:
    """cu3s origin -> clean dataset-relative identity (the tail after the last ``/data/``).

    Splits on the *last* ``/data/`` so an absolute HF-cache path
    ``/mnt/data/.../snapshots/<hash>/data/day2/<ts>.cu3s`` collapses to the stable,
    machine- and snapshot-independent ``day2/<ts>.cu3s`` (not the ``/mnt/data/`` tail).
    """
    s = str(source_cu3s).replace("\\", "/")
    marker = "/data/"
    if marker not in s:
        raise ValueError(
            f"source_cu3s {s!r} has no '/data/' segment; cannot derive a stable identity"
        )
    return s.rsplit(marker, 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--from-legacy-csv",
        required=True,
        help="legacy split CSV with columns (split, npz_path, image_id)",
    )
    ap.add_argument("--out-universe", required=True, help="output universe.csv path")
    ap.add_argument("--out-splits", required=True, help="output splits.json path")
    ap.add_argument(
        "--relative-paths",
        action="store_true",
        help="write universe paths relative to the universe.csv dir (default: absolute)",
    )
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.from_legacy_csv, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{args.from_legacy_csv}: no rows")

    # Dedup to unique frames (collapse the legacy train-row repetition), asserting
    # each npz maps to exactly one split and one image_id.
    frames: OrderedDict[str, dict] = OrderedDict()
    for r in rows:
        npz = r["npz_path"]
        prev = frames.get(npz)
        if prev is None:
            frames[npz] = {"split": r["split"], "image_id": int(r["image_id"])}
        elif prev["split"] != r["split"] or prev["image_id"] != int(r["image_id"]):
            raise SystemExit(
                f"{npz}: inconsistent legacy rows "
                f"({prev} vs split={r['split']} image_id={r['image_id']})"
            )

    out_universe = Path(args.out_universe).resolve()
    csv_dir = out_universe.parent

    # Build universe records: read the cu3s identity from each npz.
    records: list[dict] = []
    seen_identity: set[tuple[str, int]] = set()
    stage_pairs: dict[str, list[tuple[str, int]]] = {s: [] for s in _STAGES}
    for npz, meta in frames.items():
        npz_path = Path(npz).resolve()
        with np.load(npz_path) as z:
            if "source_cu3s" not in z.files:
                raise SystemExit(f"{npz_path}: npz has no 'source_cu3s' key")
            source = _normalize_source(np.asarray(z["source_cu3s"]).item())
        index = meta["image_id"]
        identity = (source, index)
        if identity in seen_identity:
            raise SystemExit(f"duplicate identity {identity} across frames")
        seen_identity.add(identity)
        stored = str(npz_path)
        if args.relative_paths:
            stored = str(Path(npz_path).relative_to(csv_dir))
        records.append({"source": source, "index": index, "path": stored})
        if meta["split"] in stage_pairs:
            stage_pairs[meta["split"]].append(identity)

    records.sort(key=lambda rec: (rec["source"], rec["index"]))

    out_universe.parent.mkdir(parents=True, exist_ok=True)
    with out_universe.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "index", "path"])
        w.writeheader()
        w.writerows(records)

    # Build the split config from per-stage identities, reusing the dataloader's
    # canonical selector construction (per-source FILE_INDICES, sorted + deduped).
    from cuvis_ai_dataloader.data.resolvers import selectors_from_refs
    from cuvis_ai_schemas.training.data import DataSplitConfig, SampleRef

    def selectors(pairs: list[tuple[str, int]]):
        refs = [SampleRef(source=s, index=i, label_id=i) for s, i in pairs]
        return selectors_from_refs(refs)

    config = DataSplitConfig(
        train=selectors(stage_pairs["train"]),
        val=selectors(stage_pairs["val"]),
        test=selectors(stage_pairs["test"]),
        # predict mirrors test so a Predictor pass (predict split) evaluates the test frames --
        # the same convention as convert_split_manifest's predict_from="test".
        predict=selectors(stage_pairs["test"]),
        leakage_check="error",
    )

    from cuvis_ai_core.data.splits_io import save_splits

    save_splits(config, args.out_splits)

    print(f"wrote {out_universe}: {len(records)} frames, {len(seen_identity)} identities")
    print(
        f"wrote {args.out_splits}: "
        f"train={len(config.train)} val={len(config.val)} test={len(config.test)} selectors "
        f"(frames: train={len(stage_pairs['train'])} val={len(stage_pairs['val'])} "
        f"test={len(stage_pairs['test'])})"
    )

    _validate(out_universe, args.out_splits, stage_pairs)


def _validate(universe_csv: Path, splits_json: str, stage_pairs: dict) -> None:
    """Round-trip: instantiate the datamodule, resolve, assert per-stage counts."""
    from cuvis_ai_core.data.splits_io import load_splits
    from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule

    dm = MultiNpzDataModule(
        universe_csv=str(universe_csv), splits=load_splits(splits_json), num_workers=0
    )
    dm.setup(stage=None)
    got = {
        "train": len(dm.train_ds) if dm.train_ds else 0,
        "val": len(dm.val_ds) if dm.val_ds else 0,
        "test": len(dm.test_ds) if dm.test_ds else 0,
    }
    want = {s: len(stage_pairs[s]) for s in _STAGES}
    if got != want:
        raise SystemExit(f"round-trip mismatch: resolved {got} != expected {want}")
    print(f"round-trip OK: resolved {got} frames per stage (leakage check passed)")


if __name__ == "__main__":
    main()
