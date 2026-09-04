"""Profile per-node inference timing of a saved pipeline (built-in profiler).

Sweeps tile_overlap x tile_batch over test frames pre-loaded to the GPU and
prints DynUNet/Norm ms per frame, fps, and peak memory (plus the full per-node
table when exactly one combination is requested)::

    python profile_pipeline.py --pipeline runs/2d128/pipeline.yaml \
        --universe-csv lentils_universe.csv --splits-json lentils_adaclip.splits.json \
        --overlaps 0,0.5 --tile-batches 1,16
"""

from __future__ import annotations

import argparse

import _engine as eng


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pipeline", required=True, help="artifact YAML from train.py")
    ap.add_argument(
        "--weights", default=None, help="artifact weights .pt (default: alongside --pipeline)"
    )
    ap.add_argument("--universe-csv", required=True, help="universe.csv from gen_splits.py")
    ap.add_argument("--splits-json", required=True, help="splits.json from gen_splits.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--skip", type=int, default=2, help="warmup forwards discarded per combination")
    ap.add_argument("--overlaps", default="0,0.25,0.5", help="comma-separated tile overlaps")
    ap.add_argument("--tile-batches", default="1,16", help="comma-separated tile_batch values")
    ap.add_argument("--device", default=None)
    ap.add_argument("--unet-manifest", default=str(eng.UNET_MANIFEST))
    ap.add_argument("--augment-manifest", default=str(eng.AUGMENT_MANIFEST))
    args = ap.parse_args()

    registry = eng.register_plugins(args.unet_manifest, args.augment_manifest)
    pipe = eng.load_artifact(
        args.pipeline, args.weights, device=args.device, registry=registry, drop_node="DataSource"
    )
    eng.profile(
        pipe,
        args.universe_csv,
        args.splits_json,
        split=args.split,
        frames=args.frames,
        skip=args.skip,
        overlaps=tuple(float(x) for x in args.overlaps.split(",")),
        tile_batches=tuple(int(x) for x in args.tile_batches.split(",")),
    )


if __name__ == "__main__":
    main()
