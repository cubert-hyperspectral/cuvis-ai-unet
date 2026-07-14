"""Tiled vs single-pass full-frame inference comparison (InstanceNorm shift demo).

Trains DynUNet briefly on fg-biased 128px patches (direct-node loop), then runs
one labeled full lentils frame through (a) a single direct grid-padded forward
and (b) sliding-window inference at the training tile size. Reports logit MAE,
argmax agreement, and fg-IoU of both against the mask — quantifying the
InstanceNorm patch<->frame statistics shift. Also hard-asserts the
input==tile equivalence of the tiled path.

Run inside the cuvis-ai env with the plugin on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lentils_datamodule import LentilsPatchDataset, labeled_frames, split_labeled  # noqa: E402

from cuvis_ai_unet.node.dynunet import DynUNet  # noqa: E402
from cuvis_ai_unet.node.losses import CrossEntropyLoss, DiceLoss  # noqa: E402
from cuvis_ai_unet.tiling import pad_hw_to_multiple, sliding_window_inference  # noqa: E402


def fg_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """IoU over foreground (>=1) pixels."""
    p, t = pred >= 1, target >= 1
    union = (p | t).sum().item()
    return ((p & t).sum().item() / union) if union else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, help="directory of lentils .npz frames")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- brief patch training (binary, 2d)
    train_files, val_files = split_labeled(args.npz_dir, "mask", args.frames, 0.25)
    ds = LentilsPatchDataset(train_files, "mask", args.tile, 4, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    bands = ds.samples[0][0].shape[-1]
    node = (
        DynUNet(
            mode="2d", in_channels=bands, num_classes=2, features=(16, 32, 64), tile_size=args.tile
        )
        .to(device)
        .train()
    )
    dice, ce = DiceLoss(), CrossEntropyLoss()
    opt = torch.optim.Adam(node.parameters(), lr=1e-3)
    for _step, batch in zip(range(args.steps), itertools.cycle(loader), strict=False):
        logits = node.forward(batch["data"].to(device))["logits"]
        loss = (
            dice.forward(logits, batch["targets"].to(device))["loss"]
            + ce.forward(logits, batch["targets"].to(device))["loss"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"trained {args.steps} steps on {len(ds)} patches ({bands} bands)")
    node.eval()

    # ---- hard gate: input == tile => tiled path equivalent to direct forward
    sample = ds[0]["data"].unsqueeze(0).to(device)  # [1, tile, tile, C]
    x_nchw = sample.permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        direct_t = node.net(x_nchw)
    tiled_u = sliding_window_inference(node.net, x_nchw, (args.tile, args.tile), gaussian=False)
    tiled_g = sliding_window_inference(node.net, x_nchw, (args.tile, args.tile), gaussian=True)
    eq_u = torch.equal(direct_t, tiled_u)
    eq_g = torch.allclose(direct_t, tiled_g, atol=1e-5)
    print(f"equivalence @tile: uniform torch.equal={eq_u}  gaussian allclose={eq_g}")
    assert eq_u and eq_g, "input==tile equivalence FAILED"

    # ---- full labeled frame: single-pass vs tiled
    frame = labeled_frames(args.npz_dir, "mask")[args.frames]  # unseen frame
    z = np.load(frame)
    cube = z["cube"].astype(np.float32)
    mu = cube.reshape(-1, cube.shape[-1]).mean(0)
    sd = cube.reshape(-1, cube.shape[-1]).std(0) + 1e-6
    cube = (cube - mu) / sd
    mask = torch.from_numpy(np.asarray(z["mask"]).astype(np.int64))
    x = torch.from_numpy(cube).unsqueeze(0).to(device)  # [1, H, W, C] BHWC

    x_nchw = x.permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        xp, revert = pad_hw_to_multiple(x_nchw, node.net.spatial_grid)
        single = node.net(xp)[revert]  # (a) one full-frame pass
    tiled_bhwc = node.forward(x)["logits"]  # (b) eval+oversize -> tiled
    tiled = tiled_bhwc.permute(0, 3, 1, 2)

    mae = (single - tiled).abs().mean().item()
    agree = (single.argmax(1) == tiled.argmax(1)).float().mean().item()
    iou_single = fg_iou(single.argmax(1)[0].cpu(), mask)
    iou_tiled = fg_iou(tiled.argmax(1)[0].cpu(), mask)
    print(f"full frame {tuple(x.shape[1:3])}: logit MAE={mae:.4f}  argmax agreement={agree:.4f}")
    print(f"fg-IoU single-pass={iou_single:.4f}  fg-IoU tiled={iou_tiled:.4f}")
    print("RESULT: TILED-INFERENCE SMOKE PASS (equivalence gates held)")


if __name__ == "__main__":
    main()
