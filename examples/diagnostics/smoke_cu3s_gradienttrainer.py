"""GradientTrainer integration via the official cu3s_multi loader (cuvis-ai-dataloader).

Path: MultiCu3sDataModule (cube, category-id mask) into the builtin CU3SDataNode,
then DynUNet and Dice + CE, trained through GradientTrainer, then Predictor
inference. Proves the plugin works with the official cu3s data module.

Run inside the cuvis-ai env with the plugin on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The cu3s datamodule is built by the lentils examples' engine.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lentils"))

import _engine as eng  # noqa: E402
from cuvis_ai_core.pipeline.factory import PipelineBuilder  # noqa: E402
from cuvis_ai_core.training import GradientTrainer  # noqa: E402
from cuvis_ai_core.training.predictor import Predictor  # noqa: E402
from cuvis_ai_core.utils.node_registry import NodeRegistry  # noqa: E402
from cuvis_ai_schemas.training.config import TrainingConfig  # noqa: E402
from cuvis_ai_schemas.training.optimizer import OptimizerConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(HERE, "lentils_unet_npz.yaml"))
    ap.add_argument("--universe-csv", required=True, help="the HF dataset's universe.csv")
    ap.add_argument("--splits-json", required=True, help="the HF dataset's splits/adaclip.json")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    # Only our plugin needs registering; CU3SDataNode is a built-in cuvis-ai node.
    NodeRegistry().register_plugin(
        os.path.join(os.path.dirname(os.path.dirname(HERE)), "plugins.yaml")
    )

    pipeline = PipelineBuilder().build_from_config(args.pipeline)
    print("pipeline nodes:", [n.name for n in pipeline.nodes])
    pipeline.unfreeze_nodes_by_name(["DynUNet"])
    nodes = {n.name: n for n in pipeline.nodes}
    loss_nodes = [nodes["DiceLoss"], nodes["CrossEntropyLoss"]]

    dm = eng.make_datamodule(args.universe_csv, args.splits_json, batch_size=1)

    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=dm,
        loss_nodes=loss_nodes,
        training_config=TrainingConfig(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            log_every_n_steps=5,
            enable_checkpointing=False,
            optimizer=OptimizerConfig(name="adam", lr=1e-3),
        ),
    )
    trainer.fit()
    print("TRAINING via cu3s_multi + GradientTrainer completed")

    dm.setup(stage="predict")
    outs = Predictor(pipeline=pipeline, datamodule=dm).predict(max_batches=2, collect_outputs=True)
    n_out = len(outs) if outs else 0
    print("inference batches:", n_out)
    if outs:
        print("output port keys:", list(outs[0].keys())[:8])
    print(
        "RESULT:",
        "cu3s_multi -> DynUNet via GradientTrainer PASS" if n_out else "no inference output",
    )


if __name__ == "__main__":
    main()
