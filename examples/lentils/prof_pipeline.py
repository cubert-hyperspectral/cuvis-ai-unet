"""Inference timing via cuvis-ai's BUILT-IN per-node profiler.

Loads a saved pipeline artifact with CuvisPipeline.load_pipeline, runs it at
ExecutionStage.INFERENCE over frames pre-loaded to the GPU, and reports per-node
timings straight from the pipeline's own profiler (set_profiling /
format_profiling_summary). Sweeps DynUNet.tile_overlap in {0, 0.25, 0.5}.

Why this is faithful — the full saved graph is loaded, but at inference only Norm and
DynUNet do work:
  - Augment is an identity passthrough when stage != TRAIN (shows as ~0 ms),
  - the loss nodes are TRAIN/VAL/TEST-only (absent from the inference profile),
  - the file-reading DataSource is neutralized (execution_stages=set()) because we
    inject cubes directly via the batch.
Frames are pre-loaded to the device so node timings are pure compute; disk read is
measured once and reported separately. synchronize_cuda=True makes the profiler bracket
each node.forward with a CUDA sync (accurate GPU wall-clock); skip_first_n discards
warmup samples (cuDNN autotune / lazy alloc).

Env: PROF_CONFIG / PROF_WEIGHTS (canonical artifact from convert_ckpt.py),
PROF_FRAMES, PROF_SKIP, PROF_OVERLAPS.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage

HERE = os.path.dirname(os.path.abspath(__file__))
UNET = "/mnt/data/anish/cuvis-ai-unet/plugins.yaml"
AUG = os.path.join(HERE, "augment_local.yaml")
CSV = os.path.join(HERE, "lentils_seg_splits_adaclip.csv")
CONFIG = os.environ.get("PROF_CONFIG", "/mnt/data/dev/infer_2d.yaml")
WEIGHTS = os.environ.get("PROF_WEIGHTS", "/mnt/data/dev/infer_2d.pt")
FRAMES = int(os.environ.get("PROF_FRAMES", "8"))
SKIP = int(os.environ.get("PROF_SKIP", "2"))
OVERLAPS = [float(x) for x in os.environ.get("PROF_OVERLAPS", "0,0.25,0.5").split(",")]
TILE_BATCHES = [int(x) for x in os.environ.get("PROF_TILE_BATCHES", "1").split(",")]


def main() -> None:
    reg = NodeRegistry()
    reg.register_plugin(UNET)
    reg.register_plugin(AUG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = CuvisPipeline.load_pipeline(
        CONFIG, weights_path=WEIGHTS, device=device,
        strict_weight_loading=False, node_registry=reg,
    )
    nodes = {n.name: n for n in pipe.nodes}
    nodes["Norm"]._statistically_initialized = True
    net = nodes["DynUNet"]
    for n in pipe.nodes:  # neutralize the file-reading source; cubes come via batch
        if n.name == "DataSource":
            n.execution_stages = set()
    for layer in pipe.torch_layers:
        layer.eval()
    runs = [n.name for n in pipe.nodes if n.should_execute(ExecutionStage.INFERENCE)]
    print(
        f"[prof] {os.path.basename(CONFIG)} on {device} | mode={net.mode} "
        f"tile={net.tile_size} | nodes executing at inference: {runs}",
        flush=True,
    )

    # Pre-load frames to the device; node timings below are pure compute.
    rows = list(csv.DictReader(open(CSV)))
    seen: set[str] = set()
    test = [
        r["npz_path"]
        for r in rows
        if r["split"] == "test" and not (r["npz_path"] in seen or seen.add(r["npz_path"]))
    ][:FRAMES]
    batches, load_ms = [], []
    for p in test:
        t0 = time.perf_counter()
        z = np.load(p)
        cube = torch.from_numpy(z["cube"].astype("float32")).unsqueeze(0)
        load_ms.append((time.perf_counter() - t0) * 1e3)
        batches.append({"cube": cube.to(device)})
    print(
        f"[prof] {len(batches)} frames on {device} | frame {tuple(batches[0]['cube'].shape)} "
        f"| mean disk-read={np.mean(load_ms):.0f} ms/frame (EXCLUDED from node timings)",
        flush=True,
    )

    single = len(OVERLAPS) == 1 and len(TILE_BATCHES) == 1
    print(f"\n{'overlap':>7} {'tile_batch':>10} {'DynUNet ms':>11} {'Norm ms':>8} {'fps':>6} {'peak GB':>8}", flush=True)
    for ov in OVERLAPS:
        net.tile_overlap = ov
        for tb in TILE_BATCHES:
            net.tile_batch = tb
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            pipe.set_profiling(enabled=True, synchronize_cuda=True, reset=True, skip_first_n=SKIP)
            with torch.no_grad():
                for _ in range(SKIP):  # warmup samples (discarded by skip_first_n)
                    pipe.forward(batch=batches[0], stage=ExecutionStage.INFERENCE)
                for b in batches:
                    pipe.forward(batch=b, stage=ExecutionStage.INFERENCE)
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            stats = {s.node_name: s for s in pipe.get_profiling_summary(ExecutionStage.INFERENCE)}
            dyn, nrm = stats["DynUNet"].mean_ms, stats["Norm"].mean_ms
            print(f"{ov:>7.2f} {tb:>10d} {dyn:>11.1f} {nrm:>8.2f} {1e3 / (dyn + nrm):>6.2f} {peak:>8.2f}", flush=True)
            if single:  # full per-node table when profiling a single config
                print(pipe.format_profiling_summary(total_frames=len(batches)), flush=True)
    print("[prof] DONE", flush=True)


if __name__ == "__main__":
    main()
