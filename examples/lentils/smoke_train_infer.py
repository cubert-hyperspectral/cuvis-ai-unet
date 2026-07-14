"""Direct-node train + inference smoke for the DynUNet segmentation node on lentils.

Trains the plugin's DynUNet node with Dice + cross-entropy on foreground-biased
lentils patches and runs inference — using the plugin's real node/loss forward
paths plus a plain optimizer (no GradientTrainer). Proves the U-Net learns to
segment lentils foreign objects end to end. Reports loss trajectory + validation
foreground IoU and saves an overlay PNG.

Run inside the cuvis-ai env with the plugin on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lentils_datamodule import LentilsPatchDataset, split_labeled  # noqa: E402

from cuvis_ai_unet.node.dynunet import DynUNet  # noqa: E402
from cuvis_ai_unet.node.losses import CrossEntropyLoss, DiceLoss  # noqa: E402


def foreground_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """IoU over the non-background classes (>=1)."""
    p, t = pred >= 1, target >= 1
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--mode", choices=["2d", "2p5d", "3d"], default="2d")
    ap.add_argument("--spectral-downsample", action="store_true",
                    help="volumetric modes: also stride the spectral axis (exercises depth pad)")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--per-frame", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="/mnt/data/anish/cuvis-ai-unet/examples/lentils/out")
    args = ap.parse_args()

    target_key = "mask" if args.target == "binary" else "class_mask"
    num_classes = 2 if args.target == "binary" else 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print(f"target={args.target} mode={args.mode} classes={num_classes} device={device}")

    train_files, val_files = split_labeled("/mnt/data/dev/lentils_npz", target_key, args.frames, 0.25)
    print(f"labeled frames: train={len(train_files)} val={len(val_files)}")
    train_ds = LentilsPatchDataset(train_files, target_key, args.patch, args.per_frame, seed=0)
    val_ds = LentilsPatchDataset(val_files, target_key, args.patch, args.per_frame, seed=1)
    print(f"patches: train={len(train_ds)} val={len(val_ds)}, bands={train_ds.samples[0][0].shape[-1]}")

    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    bands = train_ds.samples[0][0].shape[-1]

    node = DynUNet(mode=args.mode, in_channels=bands, num_classes=num_classes,
                   features=(16, 32, 64),
                   spectral_downsample=args.spectral_downsample).to(device).train()
    dice, ce = DiceLoss(), CrossEntropyLoss()
    opt = torch.optim.Adam(node.parameters(), lr=1e-3)

    losses = []
    for step, batch in zip(range(args.steps), itertools.cycle(loader)):
        data = batch["data"].to(device)
        targets = batch["targets"].to(device)
        logits = node.forward(data)["logits"]
        loss = dice.forward(logits, targets)["loss"] + ce.forward(logits, targets)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss {losses[-1]:.4f}")

    # inference / validation
    node.eval()
    ious, accs = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            s = val_ds[i]
            data = s["data"].unsqueeze(0).to(device)
            tgt = s["targets"].squeeze(-1)  # [H,W]
            pred = node.forward(data)["logits"].argmax(-1).squeeze(0).cpu()  # [H,W]
            ious.append(foreground_iou(pred, tgt))
            accs.append((pred == tgt).float().mean().item())
    mean_iou = float(torch.tensor([x for x in ious if x == x]).mean())
    print(f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f}   val pixel-acc={sum(accs)/len(accs):.4f}   "
          f"val fg-IoU={mean_iou:.4f}")

    # overlay PNG on one val patch
    _save_overlay(val_ds, node, device, os.path.join(args.out, f"overlay_{args.target}_{args.mode}.png"))

    ok = losses[-1] < losses[0]
    print("RESULT:", "TRAIN LOSS DECREASED — smoke PASS" if ok else "loss did not decrease — INSPECT")
    sys.exit(0 if ok else 1)


def _save_overlay(val_ds, node, device, path) -> None:
    """Save false-RGB | ground-truth | prediction for one foreground val patch."""
    import numpy as np

    # pick the val patch with the most foreground
    idx = max(range(len(val_ds)), key=lambda i: int((val_ds[i]["targets"] != 0).sum()))
    s = val_ds[idx]
    cube = s["data"].numpy()  # [H,W,C] normalized
    gt = s["targets"].squeeze(-1).numpy()
    with torch.no_grad():
        pred = node.forward(s["data"].unsqueeze(0).to(device))["logits"].argmax(-1)[0].cpu().numpy()
    c = cube.shape[-1]
    rgb = cube[:, :, [int(c * 0.7), int(c * 0.4), int(c * 0.1)]]
    rgb = (rgb - rgb.min()) / (float(np.ptp(rgb)) + 1e-6)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(rgb); ax[0].set_title("false-RGB")
        ax[1].imshow(gt, cmap="tab10", vmin=0, vmax=7); ax[1].set_title("ground truth")
        ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=7); ax[2].set_title("prediction")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        print("saved overlay:", path)
    except Exception as exc:  # noqa: BLE001
        np.savez(path.replace(".png", ".npz"), rgb=rgb, gt=gt, pred=pred)
        print("matplotlib unavailable, saved arrays instead:", exc)


if __name__ == "__main__":
    main()
