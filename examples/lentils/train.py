"""Train the lentils DynUNet segmentation pipeline (two-phase) and save the artifact.

Champion reproduction (2D @ 128, AdaCLIP split, expect fg-IoU ≈ 0.79 on eval)::

    python train.py --universe-csv lentils_universe.csv \
        --splits-json lentils_adaclip.splits.json \
        --epochs 20 --batch 8 --num-workers 4 --out runs/2d128

The universe.csv + splits.json come from ``gen_splits.py``. Patches-per-frame
multiplicity is ``--samples-per-frame N`` (the base datamodule draws N fresh
fg-biased crops per train frame per epoch; val/test are unaffected). Evaluate the
saved artifact with ``evaluate.py``; profile it with ``profile_pipeline.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _engine as eng


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--universe-csv", required=True, help="universe.csv from gen_splits.py")
    ap.add_argument("--splits-json", required=True, help="splits.json from gen_splits.py")
    ap.add_argument(
        "--out", required=True, help="output dir for the artifact (pipeline.yaml + .pt + run.json)"
    )
    ap.add_argument("--mode", default="2d", choices=["2d", "2p5d", "3d"])
    ap.add_argument("--patch", type=int, default=128, help="training crop and inference tile size")
    ap.add_argument("--features", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--in-channels", type=int, default=61)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument(
        "--normalizer", default="zscore", choices=["zscore", "zscore-perband", "persample"]
    )
    ap.add_argument("--max-init-frames", type=int, default=100)
    ap.add_argument(
        "--fg-percent", type=float, default=0.5, help="foreground-biased crop probability"
    )
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--samples-per-frame",
        type=int,
        default=None,
        help="N crops per frame per epoch via the dataloader (needs a dataloader "
        "release with base-module support; otherwise use gen_splits.py --repeat)",
    )
    ap.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="validate every N epochs (0 = no in-training validation)",
    )
    ap.add_argument("--tile-overlap", type=float, default=0.5)
    ap.add_argument("--tile-batch", type=int, default=16)
    ap.add_argument(
        "--dataset-crop",
        action="store_true",
        help="crop foreground-biased patches in the dataloader (npz_multi crop_size=[patch,patch]) "
        "instead of the augment graph node — ships ~patch-sized samples for a large I/O win "
        "(needs a dataloader with crop support, ALL-5905). fg-percent applies to the dataset crop.",
    )
    ap.add_argument("--hflip-prob", type=float, default=0.5, help="horizontal-flip probability")
    ap.add_argument(
        "--vflip-prob", type=float, default=0.5, help="vertical-flip probability (0 disables it)"
    )
    ap.add_argument(
        "--save-best-val",
        action="store_true",
        help="checkpoint + save the pipeline from the best val_loss epoch (needs --val-every N); "
        "a divergence to NaN val_loss never overwrites the best",
    )
    ap.add_argument("--grad-clip", type=float, default=None, help="gradient clip value (None = off)")
    ap.add_argument(
        "--tensorboard",
        action="store_true",
        help="wire SegMetrics + TensorBoardMonitorNode into the graph "
        "(view: tensorboard --logdir <out>/tensorboard); pair with --val-every N for val curves",
    )
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--unet-manifest", default=str(eng.UNET_MANIFEST))
    ap.add_argument("--augment-manifest", default=str(eng.AUGMENT_MANIFEST))
    args = ap.parse_args()

    eng.register_plugins(args.unet_manifest, args.augment_manifest)
    pipe = eng.build_graph(
        mode=args.mode,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        features=tuple(args.features),
        patch=args.patch,
        tile_overlap=args.tile_overlap,
        tile_batch=args.tile_batch,
        normalizer=args.normalizer,
        max_init_frames=args.max_init_frames,
        fg_percent=args.fg_percent,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        dataset_crop=args.dataset_crop,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        with_metrics=args.tensorboard,
        tb_dir=str(Path(args.out) / "tensorboard"),
        tb_run_name=f"{args.mode}{args.patch}",
    )
    print("pipeline nodes:", [n.name for n in pipe.nodes], flush=True)
    dm = eng.make_datamodule(
        args.universe_csv,
        args.splits_json,
        batch_size=args.batch,
        num_workers=args.num_workers,
        samples_per_frame=args.samples_per_frame,
        crop_size=(args.patch, args.patch) if args.dataset_crop else None,
        crop_fg_percent=args.fg_percent,
    )
    eng.train(
        pipe,
        dm,
        epochs=args.epochs,
        lr=args.lr,
        accelerator=args.accelerator,
        val_every=args.val_every,
        save_best_val=args.save_best_val,
        gradient_clip_val=args.grad_clip,
        out_dir=Path(args.out),
        run_meta={"args": vars(args)},
    )


if __name__ == "__main__":
    main()
