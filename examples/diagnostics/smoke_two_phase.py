"""Two-phase training smoke: StatisticalTrainer -> GradientTrainer -> tiled Predictor.

Runs the full production path on lentils via the official cu3s_multi loader:
Phase 1 statistically initializes the normalizer (its stats serialize with the
pipeline), Phase 2 gradient-trains DynUNet on fg-biased augment patches, then
Predictor runs tiled full-frame inference and the script reports val loss and
foreground IoU (used for the zscore-vs-minmax comparison).

Run inside the cuvis-ai env with the plugin on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.training.predictor import Predictor
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from pytorch_lightning.callbacks import Callback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
UNET_MANIFEST = os.path.join(REPO_ROOT, "plugins.yaml")
AUGMENT_MANIFEST = os.path.join(REPO_ROOT, "examples", "lentils", "augment.yaml")
# The cu3s datamodule is built by the lentils examples' engine.
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "lentils"))
import _engine as eng  # noqa: E402


class EpochLog(Callback):
    """Print a timestamped, flushed line per training epoch (loss + seconds/epoch)."""

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: ANN001, D102
        self._t0 = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: ANN001, D102
        dt = time.monotonic() - getattr(self, "_t0", time.monotonic())
        metrics = {
            k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else v)
            for k, v in trainer.callback_metrics.items()
        }
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] epoch {trainer.current_epoch:>3} | {dt:6.1f}s | {metrics}", flush=True)


def fg_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """IoU over foreground (>=1) pixels."""
    p, t = pred >= 1, target >= 1
    union = (p | t).sum().item()
    return ((p & t).sum().item() / union) if union else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline", default=os.path.join(HERE, "lentils_unet_npz_aug_adaclip2d128.yaml")
    )
    ap.add_argument("--universe-csv", required=True, help="the HF dataset's universe.csv")
    ap.add_argument("--splits-json", required=True, help="the HF dataset's splits/adaclip.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 (the cuvis SDK reads in this process and workers "
        "cannot use it on the GPU)",
    )
    args = ap.parse_args()

    reg = NodeRegistry()
    reg.register_plugin(UNET_MANIFEST)
    reg.register_plugin(AUGMENT_MANIFEST)

    pipeline = PipelineBuilder().build_from_config(args.pipeline)
    print("pipeline nodes:", [n.name for n in pipeline.nodes])
    nodes = {n.name: n for n in pipeline.nodes}

    dm = eng.make_datamodule(
        args.universe_csv, args.splits_json, batch_size=args.batch, num_workers=args.num_workers
    )

    # ---- Phase 1: statistical initialization (normalizer stats on full frames)
    StatisticalTrainer(pipeline=pipeline, datamodule=dm).fit()
    stat_nodes = [n for n in pipeline.nodes() if getattr(n, "requires_initial_fit", False)]
    for n in stat_nodes:
        assert getattr(n, "_statistically_initialized", False), f"{n.name} not initialized"
    norm = nodes["Norm"]
    if hasattr(norm, "zscore_mean"):
        print(
            f"Phase 1 OK: zscore mean|std norms = "
            f"{norm.zscore_mean.norm():.3f} | {norm.zscore_std.norm():.3f}"
        )
    elif hasattr(norm, "running_min"):
        print(
            f"Phase 1 OK: minmax running_min/max = "
            f"{float(norm.running_min):.4f} / {float(norm.running_max):.4f}"
        )

    # ---- Phase 2: gradient training of DynUNet only
    pipeline.unfreeze_nodes_by_name(["DynUNet"])
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=dm,
        loss_nodes=[nodes["DiceLoss"], nodes["CrossEntropyLoss"]],
        training_config=TrainingConfig(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            log_every_n_steps=1,
            enable_checkpointing=False,
            check_val_every_n_epoch=args.val_every,  # tiled full-frame val is expensive
            optimizer=OptimizerConfig(name="adam", lr=1e-3),
        ),
        callbacks=[EpochLog()],
    )
    trainer.fit()
    val_metrics = trainer.validate()
    print("val metrics:", val_metrics)

    # ---- Tiled full-frame inference + fg-IoU on the predict split
    # Lightning moves the module back to CPU after fit/validate; the Predictor runs
    # the pipeline on whatever device its parameters are on (it only moves the batch).
    # Put the whole graph on the device so tiled inference runs on GPU (fp16 via
    # autocast), not CPU — CuvisPipeline.to() cascades to every node module, the
    # house convention (dinomaly's export/predict scripts + build_from_config).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    dm.setup(stage="predict")
    outs = Predictor(pipeline=pipeline, datamodule=dm).predict(max_batches=4, collect_outputs=True)
    ious = []
    for batch_out in outs or []:
        logits = batch_out.get(("DynUNet", "logits"))
        mask = batch_out.get(("DataSource", "mask"))
        if logits is None or mask is None:
            continue
        pred = logits.argmax(-1).cpu()
        ious.append(fg_iou(pred, mask.squeeze(-1).cpu() if mask.dim() == 4 else mask.cpu()))
    mean_iou = float(torch.tensor([x for x in ious if x == x]).mean()) if ious else float("nan")
    print(f"predict batches: {len(outs or [])}  full-frame fg-IoU (tiled): {mean_iou:.4f}")
    print("RESULT: TWO-PHASE", os.path.basename(args.pipeline), "PASS" if outs else "FAIL")


if __name__ == "__main__":
    main()
