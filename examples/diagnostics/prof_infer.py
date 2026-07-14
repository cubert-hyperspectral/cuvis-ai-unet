"""Inference timing profiler for the DynUNet segmentation pipeline.

Times the two inference-time nodes (Norm, DynUNet) and their sum — the whole
inference pipeline, since losses/augmentation do not run at inference — on a
trained checkpoint, sweeping ``tile_overlap`` in {0.0, 0.25, 0.5}.

Method notes:
- Frames are pre-loaded to the GPU, so the reported numbers are pure compute.
  Disk read is the known full-frame-reload bottleneck; it is measured once and
  reported separately, NOT folded into the node timings.
- GPU work is async, so every timed region is bracketed by torch.cuda.synchronize().
- A warmup pass at the heaviest overlap absorbs cuDNN autotune + lazy allocation.
- tile_gaussian is left at its configured value. At overlap 0 the Gaussian is a
  mathematical no-op (each pixel lands in exactly one tile), so only tile COUNT
  changes across the sweep — which is exactly the variable we want to isolate.

Same argument contract as eval_ckpt.py: point --config / --ckpt at the 2D or the
3D checkpoint and run once per model. Superseded by prof_pipeline.py (built-in
per-node profiler); kept because it validated those numbers independently.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_unet.tiling import compute_tile_offsets

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--ckpt", required=True, help="raw torch_layers.state_dict() .pt")
ap.add_argument("--splits-csv", required=True)
ap.add_argument("--config", default=os.path.join(HERE, "lentils_unet_npz_aug_adaclip2d128.yaml"))
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--overlaps", default="0,0.25,0.5")
ap.add_argument("--unet-manifest", default=os.path.join(REPO_ROOT, "plugins.yaml"))
ap.add_argument(
    "--augment-manifest", default=os.path.join(REPO_ROOT, "examples", "lentils", "augment.yaml")
)
args = ap.parse_args()
UNET, AUG, CSV = args.unet_manifest, args.augment_manifest, args.splits_csv
YAML, CKPT = args.config, args.ckpt
FRAMES, WARMUP = args.frames, args.warmup
OVERLAPS = [float(x) for x in args.overlaps.split(",")]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tiles_per_frame(hw: tuple[int, int], tile: tuple[int, int], overlap: float) -> int:
    """How many tiles a frame of size ``hw`` splits into at this overlap (post min-pad)."""
    hp, wp = max(hw[0], tile[0]), max(hw[1], tile[1])
    return len(compute_tile_offsets(hp, tile[0], overlap)) * len(
        compute_tile_offsets(wp, tile[1], overlap)
    )


def main() -> None:
    reg = NodeRegistry()
    reg.register_plugin(UNET)
    reg.register_plugin(AUG)
    pipe = PipelineBuilder().build_from_config(YAML)
    nodes = {n.name: n for n in pipe.nodes}
    norm, net = nodes["Norm"], nodes["DynUNet"]

    sd = torch.load(CKPT, map_location="cpu")
    missing, unexpected = pipe.torch_layers.load_state_dict(sd, strict=False)
    norm._statistically_initialized = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    for n in pipe.torch_layers:
        n.eval()
    infer = Context(stage=ExecutionStage.INFERENCE, batch_idx=0, global_step=0)
    print(
        f"[prof] {os.path.basename(YAML)} <- {os.path.basename(CKPT)}: "
        f"missing={len(missing)} unexpected={len(unexpected)} | mode={net.mode} "
        f"tile={net.tile_size} gaussian={net.tile_gaussian} device={device}",
        flush=True,
    )

    # Pre-load frames to the GPU so the node timings are pure compute (disk I/O excluded).
    rows = list(csv.DictReader(open(CSV)))
    seen: set[str] = set()
    test = [
        r["npz_path"]
        for r in rows
        if r["split"] == "test" and not (r["npz_path"] in seen or seen.add(r["npz_path"]))
    ][:FRAMES]
    cubes, load_ms = [], []
    for p in test:
        t0 = time.perf_counter()
        z = np.load(p)
        cube = torch.from_numpy(z["cube"].astype("float32")).unsqueeze(0)
        load_ms.append((time.perf_counter() - t0) * 1e3)
        cubes.append(cube.to(device))
    _sync()
    hw = (cubes[0].shape[1], cubes[0].shape[2])
    print(
        f"[prof] {len(cubes)} frames pre-loaded to {device} | frame {hw[0]}x{hw[1]} "
        f"bands={cubes[0].shape[-1]} | mean disk-read={np.mean(load_ms):.0f} ms/frame "
        f"(EXCLUDED from the compute timings below)",
        flush=True,
    )

    # Warmup at the heaviest overlap (cuDNN autotune + lazy alloc).
    net.tile_overlap = max(OVERLAPS)
    with torch.no_grad():
        for _ in range(WARMUP):
            normed = norm.forward(data=cubes[0])["normalized"]
            net.forward(normed, context=infer)
    _sync()

    print(
        f"\n{'overlap':>7} {'tiles/fr':>8} {'norm ms':>9} {'unet ms':>9} "
        f"{'pipe ms':>9} {'+/- ms':>7} {'fps':>6} {'peak GB':>8}",
        flush=True,
    )
    results = []
    for ov in OVERLAPS:
        net.tile_overlap = ov
        tpf = tiles_per_frame(hw, net.tile_size, ov)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tn, tu = [], []
        with torch.no_grad():
            for cube in cubes:
                _sync()
                a = time.perf_counter()
                normed = norm.forward(data=cube)["normalized"]
                _sync()
                b = time.perf_counter()
                net.forward(normed, context=infer)
                _sync()
                c = time.perf_counter()
                tn.append((b - a) * 1e3)
                tu.append((c - b) * 1e3)
        tn, tu = np.array(tn), np.array(tu)
        tot = tn + tu
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        results.append(
            (ov, tpf, tn.mean(), tu.mean(), tot.mean(), tot.std(), 1e3 / tot.mean(), peak)
        )
        print(
            f"{ov:>7.2f} {tpf:>8d} {tn.mean():>9.2f} {tu.mean():>9.1f} "
            f"{tot.mean():>9.1f} {tot.std():>7.1f} {1e3 / tot.mean():>6.2f} {peak:>8.2f}",
            flush=True,
        )

    # Relative cost vs the non-overlapping baseline (isolates the tile-count effect).
    base = next((r for r in results if r[0] == 0.0), results[0])
    print(f"\n[prof] relative to overlap={base[0]:.2f} (UNet compute):", flush=True)
    for ov, tpf, _, tu, *_ in results:
        print(
            f"  overlap {ov:.2f}: {tu / base[3]:.2f}x time  ({tpf / base[1]:.2f}x tiles)",
            flush=True,
        )
    print("[prof] DONE", flush=True)


if __name__ == "__main__":
    main()
