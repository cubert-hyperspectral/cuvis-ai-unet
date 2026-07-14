"""GradientTrainer integration via the official npz_multi loader (cuvis-ai-dataloader PR #11).

Path: MultiNpzDataModule (cube/mask/class_mask) -> LentilsAdapter (data/targets)
-> DynUNet -> Dice + CE, trained through GradientTrainer, then Predictor
inference. Proves the plugin works with the official npz data module.

Run inside the cuvis-ai env with the plugin + examples/lentils on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuvis_ai_core.pipeline.factory import PipelineBuilder  # noqa: E402
from cuvis_ai_core.training import GradientTrainer  # noqa: E402
from cuvis_ai_core.training.predictor import Predictor  # noqa: E402
from cuvis_ai_core.utils.node_registry import NodeRegistry  # noqa: E402
from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule  # noqa: E402
from cuvis_ai_schemas.training.optimizer import OptimizerConfig  # noqa: E402
from cuvis_ai_schemas.training.trainer import TrainerConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(HERE, "lentils_unet_npz.yaml"))
    ap.add_argument("--csv", default=os.path.join(HERE, "lentils_seg_splits.csv"))
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

    dm = MultiNpzDataModule(splits_csv=args.csv, split="test", batch_size=1, num_workers=0)

    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=dm,
        loss_nodes=loss_nodes,
        trainer_config=TrainerConfig(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            log_every_n_steps=5,
            enable_checkpointing=False,
        ),
        optimizer_config=OptimizerConfig(name="adam", lr=1e-3),
    )
    trainer.fit()
    print("TRAINING via npz_multi + GradientTrainer completed")

    dm.setup(stage="predict")
    outs = Predictor(pipeline=pipeline, datamodule=dm).predict(max_batches=2, collect_outputs=True)
    n_out = len(outs) if outs else 0
    print("inference batches:", n_out)
    if outs:
        print("output port keys:", list(outs[0].keys())[:8])
    print(
        "RESULT:",
        "npz_multi -> DynUNet via GradientTrainer PASS" if n_out else "no inference output",
    )


if __name__ == "__main__":
    main()
