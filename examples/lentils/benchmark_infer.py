"""Tiled sliding-window inference timing: CPU vs GPU(fp32) vs GPU(autocast).

Times the exact tiled-predict path (cuvis_ai_unet.tiling.sliding_window_inference)
on one real lentils frame, for 3D@128 and 2D@128 with the deep [32..512] net.
CPU is timed on a small region (tractable) and extrapolated per-tile; GPU is timed
on the full frame. Weights are untrained — timing is weight-independent."""
import glob
import time

import numpy as np
import torch

from cuvis_ai_unet.node.dynunet import DynUNet
from cuvis_ai_unet.tiling import compute_tile_offsets, sliding_window_inference

CUBE = np.load(sorted(glob.glob("/mnt/data/dev/lentils_npz_adaclip/*.npz"))[0])["cube"].astype(
    "float32"
)
Hf, Wf, C = CUBE.shape
TILE = 128


def backbone(mode):
    return (
        DynUNet(
            mode=mode, in_channels=61, num_classes=2, features=[32, 64, 128, 256, 512],
            tile_size=128, tile_overlap=0.5, tile_gaussian=True,
        )
        .net.eval()
    )


def ntiles(h, w):
    return len(compute_tile_offsets(h, TILE, 0.5)) * len(compute_tile_offsets(w, TILE, 0.5))


def run(mode, dev, amp, h, w):
    net = backbone(mode).to(dev)
    x = torch.from_numpy(CUBE[:h, :w, :]).permute(2, 0, 1).unsqueeze(0).contiguous().to(dev)
    with torch.no_grad():
        if dev == "cuda":  # warm up CUDA kernels/allocator
            sliding_window_inference(net, x, (TILE, TILE), 0.5, True, amp=amp)
            torch.cuda.synchronize()
        t0 = time.monotonic()
        sliding_window_inference(net, x, (TILE, TILE), 0.5, True, amp=amp)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.monotonic() - t0
    return dt, ntiles(h, w)


full = ntiles(Hf, Wf)
print(f"frame {Hf}x{Wf}x{C}  |  full-frame tiles @128/0.5 = {full}", flush=True)
cuda = torch.cuda.is_available()
for mode in ["3d", "2d"]:
    print(f"=== DynUNet {mode} @128, deep [32..512] ===", flush=True)
    dt, nt = run(mode, "cpu", False, 256, 256)
    per = dt / nt
    print(f"  CPU  fp32     : {per*1000:7.0f} ms/tile  -> full frame ~{per*full:7.1f} s", flush=True)
    if cuda:
        dt, nt = run(mode, "cuda", False, Hf, Wf)
        print(f"  GPU  fp32     : {dt/nt*1000:7.1f} ms/tile  -> full frame  {dt:7.2f} s", flush=True)
        dt, nt = run(mode, "cuda", True, Hf, Wf)
        print(f"  GPU  autocast : {dt/nt*1000:7.1f} ms/tile  -> full frame  {dt:7.2f} s", flush=True)
