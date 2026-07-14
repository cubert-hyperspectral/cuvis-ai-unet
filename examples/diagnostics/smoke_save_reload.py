"""Save -> reload round-trip of a trained, tiling-configured pipeline (matrix #9).

Builds the two-phase zscore pipeline, statistically initializes + briefly
trains it, saves it with weights, reloads it through
``CuvisPipeline.load_pipeline`` (yaml hparams round-trip: tile_size list ->
tuple), and hard-asserts that the reloaded pipeline produces identical tiled
logits on a full lentils frame.

Run inside the cuvis-ai env with the plugin on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os

import torch
from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule

from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
UNET_MANIFEST = os.path.join(REPO_ROOT, "plugins.yaml")
AUGMENT_MANIFEST = os.path.join(REPO_ROOT, "examples", "lentils", "augment.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(HERE, "lentils_unet_npz_aug_adaclip2d128.yaml"))
    ap.add_argument("--csv", default=os.path.join(HERE, "lentils_seg_splits.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "saved_pipeline"))
    args = ap.parse_args()

    reg = NodeRegistry()
    reg.register_plugin(UNET_MANIFEST)
    reg.register_plugin(AUGMENT_MANIFEST)

    pipeline = PipelineBuilder().build_from_config(args.pipeline)
    dm = MultiNpzDataModule(splits_csv=args.csv, split="test", batch_size=2, num_workers=0)
    StatisticalTrainer(pipeline=pipeline, datamodule=dm).fit()
    pipeline.unfreeze_nodes_by_name(["DynUNet"])
    GradientTrainer(
        pipeline=pipeline, datamodule=dm,
        loss_nodes=[n for n in pipeline.nodes if n.name in ("DiceLoss", "CrossEntropyLoss")],
        trainer_config=TrainerConfig(
            max_epochs=1, accelerator="auto", devices=1, enable_progress_bar=False,
            log_every_n_steps=1, enable_checkpointing=False, check_val_every_n_epoch=10,
        ),
        optimizer_config=OptimizerConfig(name="adam", lr=1e-3),
    ).fit()

    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "pipeline.yaml")
    pipeline.save_to_file(cfg_path, save_weights=True)
    print("saved:", sorted(os.listdir(args.out)))

    # save_to_file writes weights alongside the config as <stem>.pt; load_pipeline
    # must be pointed at it to restore buffers AND the statistical-init flag
    # (node.py marks statistical nodes initialized inside the weight-restore path).
    weights_path = os.path.join(args.out, "pipeline.pt")
    reloaded = CuvisPipeline.load_pipeline(cfg_path, weights_path=weights_path, node_registry=reg)

    # Compare tiled logits on one full frame at INFERENCE stage.
    dm.setup(stage="predict")
    batch = next(iter(dm.predict_dataloader()))
    ctx = Context(stage=ExecutionStage.INFERENCE)
    with torch.no_grad():
        out_a = pipeline.forward(batch=dict(batch), context=ctx)
        out_b = reloaded.forward(batch=dict(batch), context=ctx)
    la = out_a[("DynUNet", "logits")].cpu()
    lb = out_b[("DynUNet", "logits")].cpu()
    identical = torch.allclose(la, lb, atol=1e-6)
    max_diff = (la - lb).abs().max().item()
    print(f"reloaded tiled logits identical: {identical}  (max diff {max_diff:.2e})")
    # Also confirm the tiling hparams survived the yaml round-trip.
    node = {n.name: n for n in reloaded.nodes}["DynUNet"]
    print(f"reloaded tile config: tile_size={node.tile_size} overlap={node.tile_overlap} "
          f"gaussian={node.tile_gaussian}")
    assert identical, "save->reload round-trip produced different logits"
    assert tuple(node.tile_size) == (128, 128)
    print("RESULT: SAVE->RELOAD ROUND-TRIP PASS")


if __name__ == "__main__":
    main()
