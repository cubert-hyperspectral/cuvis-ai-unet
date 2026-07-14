"""GradientTrainer + pipeline integration smoke.

Trains DynUNet on lentils through cuvis-ai's real training stack — build the
pipeline from yaml, wire the loss nodes, GradientTrainer.fit(), then Predictor
inference — rather than the direct-node loop. Proves the plugin works on the
production training path.

Run inside the cuvis-ai env with the plugin on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lentils_datamodule import LentilsDataModule  # noqa: E402

from cuvis_ai_core.pipeline.factory import PipelineBuilder  # noqa: E402
from cuvis_ai_core.training import GradientTrainer  # noqa: E402
from cuvis_ai_core.training.predictor import Predictor  # noqa: E402
from cuvis_ai_core.utils.node_registry import NodeRegistry  # noqa: E402
from cuvis_ai_schemas.training.optimizer import OptimizerConfig  # noqa: E402
from cuvis_ai_schemas.training.trainer import TrainerConfig  # noqa: E402

PLUGIN_MANIFEST = "/mnt/data/anish/cuvis-ai-unet/plugins.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(os.path.dirname(__file__), "lentils_unet_binary.yaml"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    NodeRegistry().register_plugin(PLUGIN_MANIFEST)
    pipeline = PipelineBuilder().build_from_config(args.pipeline)
    print("pipeline nodes:", [n.name for n in pipeline.nodes])
    pipeline.unfreeze_nodes_by_name(["DynUNet"])
    nodes = {n.name: n for n in pipeline.nodes}
    loss_nodes = [nodes["DiceLoss"], nodes["CrossEntropyLoss"]]

    dm = LentilsDataModule(target="binary", patch=128, per_frame=4, max_frames=args.frames, batch_size=4)

    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=dm,
        loss_nodes=loss_nodes,
        trainer_config=TrainerConfig(
            max_epochs=args.epochs, accelerator="auto", devices=1,
            enable_progress_bar=False, log_every_n_steps=1,
            enable_checkpointing=False, check_val_every_n_epoch=1,
        ),
        optimizer_config=OptimizerConfig(name="adam", lr=1e-3),
    )
    trainer.fit()
    print("TRAINING via GradientTrainer.fit() completed")

    dm.setup(stage="predict")
    outs = Predictor(pipeline=pipeline, datamodule=dm).predict(max_batches=2, collect_outputs=True)
    n_out = len(outs) if outs else 0
    print("inference batches:", n_out)
    if outs:
        print("output port keys:", list(outs[0].keys())[:8])
    print("RESULT:", "GradientTrainer + Predictor integration PASS" if n_out else "inference produced no output")


if __name__ == "__main__":
    main()
