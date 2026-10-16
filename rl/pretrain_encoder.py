"""
pretrain_encoder.py - Single-frame encoder pretraining (PLAN phase 5, Level 1)

Supervised training on online FrameData batches (randomized renders, fresh every step)
of a FrameModel (CNN or ViT encoder to z = 256, heads on z only).

Run from rl/:
    uv run python pretrain_encoder.py --arch cnn --seed 0
    uv run python pretrain_encoder.py --arch vit --patch 8 --seed 0
    uv run python pretrain_encoder.py --arch vit --patch 4 --seed 0
Outputs in runs/encoder/<name>/: config.json, train.csv, val.csv, model.pt.
Evaluation: eval_encoder.py.

Loss = w_map L_map + w_count L_count + w_ell L_ell:
    L_map    peak-weighted MSE on the square-root spot map,
             mean over pixels of (1 + PEAK_WEIGHT sqrt(t)) (p - sqrt(t))^2
    L_count  cross-entropy over the counts 0-64
    L_ell    smooth L1 on (cx, cy, mxx, myy, mxy) / ELL_SCALE_MM, valid fits only
Why weighted MSE and not a CenterNet focal loss: the target peaks carry the incident
intensity (0.98^k, 1 down to 0.08) and overlapping spots sum, and the focal loss treats
the map as a probability with unit peaks, so it cannot regress either. The square root
compresses the 13x brightness range so dim late-bounce spots are not lost against
bright early ones, and the peak weight counters the sparsity (spots cover ~2-15 % of
the pixels), which in plain MSE pulls the prediction toward an empty map.
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn.functional as F

from encoders import EncoderConfig, FrameModel, count_params, model_name
from frame_data import DataConfig, FrameData

RL_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(RL_DIR, "runs", "encoder")
PEAK_WEIGHT = 20.0  # Map loss weight 1 + PEAK_WEIGHT sqrt(t)
ELL_SCALE_MM = 10.0  # Ellipse targets are regressed in units of this
VAL_SEED = 777  # Fixed in-training validation set


# ── Loss ────────────────────────────────────────────────────────────────────


def frame_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    w_map: float = 1.0,
    w_count: float = 0.1,
    w_ell: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Loss terms (float32 scalars) and their weighted sum 'total'."""
    s = torch.sqrt(batch["spot_map"])
    p = out["map"].float()
    l_map = ((1.0 + PEAK_WEIGHT * s) * (p - s) ** 2).mean()
    l_count = F.cross_entropy(out["count"].float(), batch["count"])
    m = batch["ellipse_mask"].float()
    e = F.smooth_l1_loss(
        out["ellipse"].float(), batch["ellipse"] / ELL_SCALE_MM, beta=0.02, reduction="none"
    ).mean(1)
    l_ell = (e * m).sum() / m.sum().clamp_min(1.0)
    total = w_map * l_map + w_count * l_count + w_ell * l_ell
    return {"total": total, "map": l_map, "count": l_count, "ellipse": l_ell}


# ── Optimizer ───────────────────────────────────────────────────────────────


def param_groups(model: torch.nn.Module, weight_decay: float):
    """AdamW groups: no decay on biases, norms, embeddings, pooling queries, scalars."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "queries" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    """Linear warmup, then cosine to min_ratio."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


# ── Validation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model: FrameModel, val: Dict[str, torch.Tensor], bs: int = 256) -> Dict[str, float]:
    """Mean losses and count accuracy on the fixed validation set (in-distribution)."""
    model.eval()
    n = val["image_id"].shape[0]
    sums = {"total": 0.0, "map": 0.0, "count": 0.0, "ellipse": 0.0, "count_acc": 0.0}
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in val.items() if torch.is_tensor(v)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["image_id"])
        losses = frame_loss(out, b)
        k = b["count"].shape[0] / n
        for key in ("total", "map", "count", "ellipse"):
            sums[key] += float(losses[key]) * k
        sums["count_acc"] += float((out["count"].argmax(1) == b["count"]).float().mean()) * k
    model.train()
    return sums


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arch", choices=("cnn", "vit"), default="cnn")
    ap.add_argument("--patch", type=int, default=8, help="ViT patch size (4 or 8)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="Peak LR (default cnn 1e-3, vit 5e-4)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--w-map", type=float, default=1.0)
    ap.add_argument("--w-count", type=float, default=0.1)
    ap.add_argument("--w-ell", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="fp32 instead of bf16 autocast")
    ap.add_argument("--no-compile", action="store_true", help="eager model (no torch.compile)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=1024)
    ap.add_argument("--tag", default="", help="Suffix for the run name")
    ap.add_argument("--wandb", action="store_true", help="Log to W&B (project rl-encoder)")
    return ap.parse_args()


def run_name(args: argparse.Namespace, cfg: EncoderConfig) -> str:
    """Display / directory name from the key config, built at start."""
    k = args.steps / 1000
    steps = f"{k:g}k" if args.steps >= 1000 else str(args.steps)
    name = f"{model_name(cfg)}_w{cfg.width if cfg.arch == 'vit' else cfg.cnn_widths[-1]}" \
           f"_lr{args.lr:g}_b{args.batch}_{steps}_s{args.seed}"
    return name + (f"_{args.tag}" if args.tag else "")


def main():
    args = parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    if args.lr is None:
        args.lr = 1e-3 if args.arch == "cnn" else 5e-4
    cfg = EncoderConfig(arch=args.arch, patch=args.patch)
    name = run_name(args, cfg)
    run_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)

    data = FrameData(DataConfig(batch_size=args.batch), dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1_000_003 * (args.seed + 1))
    val = data.eval_set(args.val_n, VAL_SEED)

    model = FrameModel(cfg).to(dev)
    fwd = model if args.no_compile else torch.compile(model)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, args.steps))
    n_enc, n_heads = count_params(model.encoder), count_params(model.heads)
    info = {"name": name, "args": vars(args), "encoder": asdict(cfg),
            "params_encoder": n_enc, "params_heads": n_heads,
            "data": {"batch_size": args.batch, "cell_frac": data.cfg.cell_frac,
                     "n_cells": int(data.cells.shape[0])}}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"{name}: encoder {n_enc / 1e6:.2f} M params, heads {n_heads / 1e6:.2f} M -> {run_dir}")

    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(project="rl-encoder", name=name, config=info, dir=run_dir)

    train_f = open(os.path.join(run_dir, "train.csv"), "w", newline="")
    val_f = open(os.path.join(run_dir, "val.csv"), "w", newline="")
    tw = csv.writer(train_f)
    vw = csv.writer(val_f)
    tw.writerow(["step", "time_s", "lr", "loss", "map", "count", "ellipse", "grad_norm",
                 "grad_norm_max", "skipped"])
    vw.writerow(["step", "time_s", "loss", "map", "count", "ellipse", "count_acc"])

    acc = {k: torch.zeros((), device=dev) for k in ("total", "map", "count", "ellipse", "gn")}
    gn_max = torch.zeros((), device=dev)
    skipped = 0
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch = data.batch(gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
            out = fwd(batch["image"])
        losses = frame_loss(out, batch, args.w_map, args.w_count, args.w_ell)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if bool(torch.isfinite(losses["total"]) & torch.isfinite(gn)):
            opt.step()
        else:
            skipped += 1
        sched.step()
        for k in ("total", "map", "count", "ellipse"):
            acc[k] += losses[k].detach()
        acc["gn"] += gn
        gn_max = torch.maximum(gn_max, gn)

        if step % args.log_every == 0:
            n = args.log_every
            row = [step, round(time.time() - t0, 1), sched.get_last_lr()[0]] + [
                float(acc[k]) / n for k in ("total", "map", "count", "ellipse", "gn")
            ] + [float(gn_max), skipped]
            tw.writerow(row)
            train_f.flush()
            if wb is not None:
                wb.log(dict(zip(["lr", "loss", "map", "count", "ellipse", "grad_norm",
                                 "grad_norm_max"], row[2:9])), step=step)
            for k in acc:
                acc[k].zero_()
            gn_max.zero_()
        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, val)
            el = time.time() - t0
            vw.writerow([step, round(el, 1)] + [v[k] for k in ("total", "map", "count", "ellipse", "count_acc")])
            val_f.flush()
            if wb is not None:
                wb.log({f"val/{k}": x for k, x in v.items()}, step=step)
            print(f"step {step:6d} {el / 60:5.1f} min  val loss {v['total']:.4f} map {v['map']:.4f} "
                  f"count {v['count']:.3f} (acc {v['count_acc']:.3f}) ell {v['ellipse']:.4f}  "
                  f"skipped {skipped}", flush=True)
            torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "step": step,
                        "name": name}, os.path.join(run_dir, "model.pt"))

    elapsed = time.time() - t0
    info.update(train_time_s=round(elapsed, 1), skipped_steps=skipped,
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(info, f, indent=1)
    train_f.close()
    val_f.close()
    if wb is not None:
        wb.finish()
    print(f"done in {elapsed / 60:.1f} min, peak {info['peak_mem_gb']} GB")


if __name__ == "__main__":
    main()
