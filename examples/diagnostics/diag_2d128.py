"""Diagnose adaclip 2D@128 fg-IoU=0.0. Train briefly, SAVE the pipeline weights,
then on foreground test frames dump: target fg-pixel count, argmax fg-pixel count,
max/mean foreground softmax prob, and pixel counts over thresholds — for BOTH the
tiled path and a direct full-frame forward. Distinguishes soft-prob-never-crosses-0.5
(imbalance/threshold) from per-tile InstanceNorm washing foreground out (tiled fails,
direct succeeds)."""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule

from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--splits-csv", required=True)
ap.add_argument("--ckpt-out", required=True, help="where to save the raw torch_layers state_dict")
ap.add_argument("--config", default=os.path.join(HERE, "lentils_unet_npz_aug_adaclip2d128.yaml"))
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--num-workers", type=int, default=4)
ap.add_argument("--batch", type=int, default=8, help="3D@128 needs 4 (batch 8 OOMs the GPU)")
ap.add_argument("--no-val", action="store_true", help="disable the trainer's full-frame validation")
ap.add_argument("--unet-manifest", default=os.path.join(REPO_ROOT, "plugins.yaml"))
ap.add_argument("--augment-manifest",
                default=os.path.join(REPO_ROOT, "examples", "lentils", "augment.yaml"))
args = ap.parse_args()
UNET, AUG = args.unet_manifest, args.augment_manifest
CSV, YAML, CKPT = args.splits_csv, args.config, args.ckpt_out
EPOCHS, NW, BATCH = args.epochs, args.num_workers, args.batch


def fg_stats(logits_bhwc):
    """(argmax-fg count, max fg-prob, fg-prob map [H,W]) for one frame's BHWC logits."""
    p = F.softmax(logits_bhwc.float(), dim=-1)[0, ..., 1]
    a = logits_bhwc.argmax(-1)[0]
    return int((a >= 1).sum()), float(p.max()), p


def main() -> None:
    reg = NodeRegistry()
    reg.register_plugin(UNET)
    reg.register_plugin(AUG)
    pipe = PipelineBuilder(node_registry=reg).build_from_config(YAML)
    nodes = {n.name: n for n in pipe.nodes}
    dm = MultiNpzDataModule(
        splits_csv=CSV, split="test", batch_size=BATCH, num_workers=NW, persistent_workers=NW > 0
    )
    if args.no_val:
        # The trainer's built-in val runs DynUNet on FULL frames; in 3D that OOMs
        # (InstanceNorm3d over the whole 1000x1080 volume). We don't need it — real
        # metrics come from eval_ckpt.py (tiled, batch 1). Feed an EMPTY val loader
        # (0 batches) so Lightning's sanity-check/val loop runs nothing.
        from torch.utils.data import DataLoader, Dataset

        class _EmptyDS(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, i):
                raise IndexError

        dm.val_dataloader = lambda *a, **k: DataLoader(_EmptyDS())  # type: ignore[method-assign]
        print("[diag] trainer validation disabled via empty loader (metrics via eval_ckpt.py)", flush=True)

    print(f"[diag] {os.path.basename(YAML)} | Phase 1 stat-init + Phase 2 train ({EPOCHS} epochs, nw={NW})", flush=True)
    StatisticalTrainer(pipeline=pipe, datamodule=dm).fit()
    pipe.unfreeze_nodes_by_name(["DynUNet"])
    GradientTrainer(
        pipeline=pipe, datamodule=dm,
        loss_nodes=[nodes["DiceLoss"], nodes["CrossEntropyLoss"]],
        trainer_config=TrainerConfig(
            max_epochs=EPOCHS, accelerator="auto", devices=1, enable_progress_bar=False,
            log_every_n_steps=1, enable_checkpointing=False, check_val_every_n_epoch=EPOCHS,
        ),
        optimizer_config=OptimizerConfig(name="adam", lr=1e-3),
    ).fit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    torch.save(pipe.torch_layers.state_dict(), CKPT)
    print(f"[diag] saved weights -> {CKPT}", flush=True)

    net = nodes["DynUNet"]
    norm = nodes["Norm"]
    net.eval()
    norm.eval()
    infer = Context(stage=ExecutionStage.INFERENCE, batch_idx=0, global_step=0)

    dm.setup(stage="predict")
    seen = 0
    with torch.no_grad():
        for batch in dm.predict_dataloader():
            cube = batch["cube"].to(device)
            mask = batch["mask"].to(device)
            normed = norm.forward(data=cube)["normalized"]
            for b in range(cube.shape[0]):
                m = mask[b].squeeze(-1) if mask[b].dim() == 3 else mask[b]
                tgt = int((m >= 1).sum())
                if tgt == 0:
                    continue
                xb = normed[b : b + 1]
                lt = net.forward(xb, context=infer)["logits"]  # tiled
                saved = net.tile_size
                net.tile_size = None
                ld = net.forward(xb, context=infer)["logits"]  # direct full-frame
                net.tile_size = saved
                at, mt, pt = fg_stats(lt)
                ad, md, pd = fg_stats(ld)
                print(
                    f"[frame] tgt_fg={tgt:6d} | TILED argmax_fg={at:6d} maxP={mt:.3f} "
                    f"p>0.5={int((pt > 0.5).sum()):6d} p>0.3={int((pt > 0.3).sum()):7d} | "
                    f"DIRECT argmax_fg={ad:6d} maxP={md:.3f} p>0.5={int((pd > 0.5).sum()):6d}",
                    flush=True,
                )
                seen += 1
                if seen >= 10:
                    break
            if seen >= 10:
                break
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
