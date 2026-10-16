"""
bench_ppo.py - throughput of ppo.py iterations for several batch layouts (timing only)

For each configuration the Trainer is built as for a real run (pool mode, encoder and
--init-from weights) but without eval sets, W&B or checkpoints; a few warmup iterations
absorb torch.compile, then pool refresh, rollout and update are timed with cuda
synchronize. target_kl is disabled so every update runs all epochs and the update times
are comparable across configurations. Nothing is learned that is kept.

    uv run python bench_ppo.py --encoder-init ENC.pt --init-from PPO.pt --out DIR
    uv run python bench_ppo.py ... --configs f1024x128 u1024x128   # a subset
"""

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from typing import Dict, List

import torch

from ppo import PPOConfig, Trainer

# name: (num_envs, rollout_len, encoder_freeze, enc_checkpoint, minibatches)
CONFIGS: Dict[str, tuple] = {
    # Frozen encoder, 131k transitions per iteration
    "f1024x128": (1024, 128, True, True, 8),
    "f2048x64": (2048, 64, True, True, 8),
    "f4096x32": (4096, 32, True, True, 8),
    "f8192x16": (8192, 16, True, True, 8),
    # Frozen encoder, 262k transitions per iteration
    "f2048x128": (2048, 128, True, True, 8),
    "f4096x64": (4096, 64, True, True, 8),
    # Encoder trained
    "u1024x128": (1024, 128, False, True, 8),
    "u1024x128-nockpt-mb16": (1024, 128, False, False, 16),
    "u4096x32": (4096, 32, False, True, 8),
    "u4096x32-nockpt-mb16": (4096, 32, False, False, 16),
}
# Seconds-long code-path checks (not in the default list)
SMOKE: Dict[str, tuple] = {"smoke-f": (16, 8, True, True, 2), "smoke-u": (16, 8, False, False, 2)}
CONFIGS_ALL = {**CONFIGS, **SMOKE}


def bench(base: PPOConfig, name: str, warmup: int, iters: int, run_dir: str) -> Dict:
    B, T, freeze, ckpt, mb = CONFIGS_ALL[name]
    c = replace(base, num_envs=B, rollout_len=T, encoder_freeze=freeze, enc_checkpoint=ckpt,
                minibatches=mb, target_kl=1e9, wandb=False)
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    sync = lambda: torch.cuda.synchronize(dev)  # noqa: E731
    out: Dict = {"config": name, "num_envs": B, "rollout_len": T, "encoder_frozen": freeze,
                 "enc_checkpoint": ckpt, "minibatches": mb}
    t0 = time.time()
    tr = Trainer(c, run_dir=os.path.join(run_dir, name), build_eval=False)
    out["init_s"] = time.time() - t0
    rows: List[Dict[str, float]] = []
    for i in range(warmup + iters):
        sync()
        a = time.time()
        if tr.sampler is not None:
            tr.sampler.refresh(tr.iteration)
        sync()
        b = time.time()
        tr.rollout()
        sync()
        d = time.time()
        tr.update()
        sync()
        e = time.time()
        tr.iteration += 1
        tr.env_steps += B * T
        if i >= warmup:
            rows.append({"refresh": b - a, "rollout": d - b, "update": e - d})
        else:
            print(f"  {name} warmup {i}: refresh {b - a:.2f} s, rollout {d - b:.2f} s, "
                  f"update {e - d:.2f} s", flush=True)
    for k in ("refresh", "rollout", "update"):
        out[f"{k}_s"] = sum(r[k] for r in rows) / len(rows)
    it = out["refresh_s"] + out["rollout_s"] + out["update_s"]
    out["iteration_s"] = it
    out["steps_per_s"] = B * T / it
    out["steps_per_s_no_refresh"] = B * T / (out["rollout_s"] + out["update_s"])
    out["hours_per_1e9"] = 1e9 / out["steps_per_s"] / 3600
    out["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
    del tr
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoder-init", required=True)
    ap.add_argument("--init-from", default="")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--pool-size", type=int, default=0, help="0 = the PPOConfig default")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", required=True, help="Directory for bench.json / bench.md")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    base = PPOConfig(encoder_init=args.encoder_init, init_from=args.init_from)
    if args.pool_size:
        base = replace(base, pool_size=args.pool_size)
    gpu = torch.cuda.get_device_name()
    print(f"GPU {gpu}; configs {args.configs}", flush=True)
    results = []
    for name in args.configs:
        try:
            r = bench(base, name, args.warmup, args.iters, os.path.join(args.out, "tmp"))
        except torch.cuda.OutOfMemoryError as err:
            r = {"config": name, "error": f"OOM: {str(err).splitlines()[0]}"}
            gc.collect()
            torch.cuda.empty_cache()
        results.append(r)
        print(json.dumps(r), flush=True)
        with open(os.path.join(args.out, "bench.json"), "w") as f:
            json.dump({"gpu": gpu, "results": results}, f, indent=1)
    lines = [f"# PPO throughput, {gpu}", "",
             "| config | envs x steps | encoder | ckpt | mb | refresh s | rollout s | update s "
             "| iter s | steps/s | steps/s w/o refresh | h per 1e9 | peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['config']} | {r['error']} |")
            continue
        lines.append(
            f"| {r['config']} | {r['num_envs']} x {r['rollout_len']} | "
            f"{'frozen' if r['encoder_frozen'] else 'trained'} | {int(r['enc_checkpoint'])} | "
            f"{r['minibatches']} | {r['refresh_s']:.2f} | {r['rollout_s']:.2f} | "
            f"{r['update_s']:.2f} | {r['iteration_s']:.2f} | {r['steps_per_s']:.0f} | "
            f"{r['steps_per_s_no_refresh']:.0f} | {r['hours_per_1e9']:.1f} | {r['peak_gb']:.1f} |")
    with open(os.path.join(args.out, "bench.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
