"""Evaluate a saved lentils segmentation pipeline: fg-IoU / fg-Dice / image-AUROC.

Artifact mode (the ``train.py`` output — YAML + co-located name-keyed .pt)::

    python evaluate.py --pipeline runs/2d128/pipeline.yaml \
        --splits-csv lentils_seg_splits_adaclip.csv

Legacy mode (a raw index-keyed ``torch_layers.state_dict()`` checkpoint plus the
pipeline config it was saved under — any missing/unexpected key is a hard error
because index-keyed weights silently misassign under a reordered config)::

    python evaluate.py --config diagnostics/lentils_unet_npz_aug_adaclip2d128.yaml \
        --raw-ckpt diag_2d128_pipeline.pt --splits-csv lentils_seg_splits_adaclip.csv

Prints the metrics and, for the champion 2D configuration, the delta against
the published reference numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _engine as eng


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pipeline", help="artifact YAML from train.py (weights = co-located .pt)")
    src.add_argument("--config", help="legacy: pipeline config YAML for a raw checkpoint")
    ap.add_argument(
        "--weights", default=None, help="artifact weights .pt (default: alongside --pipeline)"
    )
    ap.add_argument("--raw-ckpt", default=None, help="legacy: raw torch_layers state_dict .pt")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="legacy: tolerate missing/unexpected checkpoint keys (DANGEROUS)",
    )
    ap.add_argument("--splits-csv", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--tile-overlap", type=float, default=None, help="override the artifact's overlap"
    )
    ap.add_argument(
        "--tile-batch", type=int, default=None, help="override the artifact's tile batch"
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames in the split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="also write metrics JSON here")
    ap.add_argument("--unet-manifest", default=str(eng.UNET_MANIFEST))
    ap.add_argument("--augment-manifest", default=str(eng.AUGMENT_MANIFEST))
    args = ap.parse_args()

    registry = eng.register_plugins(args.unet_manifest, args.augment_manifest)
    if args.pipeline:
        pipe = eng.load_artifact(args.pipeline, args.weights, device=args.device, registry=registry)
    else:
        if not args.raw_ckpt:
            ap.error("--config requires --raw-ckpt")
        pipe = eng.load_raw_ckpt(
            args.config,
            args.raw_ckpt,
            device=args.device,
            registry=registry,
            allow_partial=args.allow_partial,
        )

    m = eng.evaluate(
        pipe,
        args.splits_csv,
        split=args.split,
        tile_overlap=args.tile_overlap,
        tile_batch=args.tile_batch,
        max_frames=args.max_frames,
    )
    print(
        f"[eval] frames: {m['frames']} total | {m['object_frames']} object | "
        f"{m['normal_frames']} normal",
        flush=True,
    )
    print(
        f"[eval] SEGMENTATION (object frames): fg-IoU={m['fg_iou']:.4f}  fg-Dice={m['fg_dice']:.4f}",
        flush=True,
    )
    print(
        f"[eval] IMAGE-LEVEL AUROC: max-prob={m['image_auroc_maxprob']:.4f}  "
        f"pred-fg-area={m['image_auroc_area']:.4f}",
        flush=True,
    )
    champ = eng.CHAMPION["2d_128"]
    print(
        f"[eval] champion delta (2d_128 reference {champ['fg_iou']:.4f}): "
        f"{m['fg_iou'] - champ['fg_iou']:+.4f} fg-IoU",
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=2))
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
