"""
probe_path.py - Probe: how well the frozen agent's core features predict the absolute path

A ppo.py checkpoint is loaded with its run config (env with its renderer, scenario pool,
agent); every weight stays frozen. Rollouts in that env (sampled actions, as in training)
record per step the core output h of the current frame (the input of the heads), the
existing path head's output (P / P_ref) and the labels of that frame: path P (mm), P_ref,
bounces. Train / val / test splits are separate rollout segments (new episodes).

A new head with the path head's shape (MLP d_model -> head_hidden -> 1) is trained on h
alone to predict log P (P in mm, floored at 1 mm), MSE, AdamW; the learning rate drops on
validation plateaus and the training stops after the last one (train to convergence).
Both heads are then compared on the test split in the same units: absolute path (the
existing head's P / P_ref times the true P_ref), relative error |P_hat - P| / P, log error,
and error in P / P_ref units (the eval metric head_path_mae), overall and per bounce bin.
A constant baseline (mean log P of train) gives the scale.

    python probe_path.py --ckpt <ppo best.pt> --out <dir> [--steps-train 400] [--steps-eval 100]
Outputs: probe_path.json, probe_path.md, head.pt (the new head's weights).
"""

import argparse
import json
import math
import os
from typing import Dict

import torch
import torch.nn.functional as F

BINS = ((0, 2), (3, 30), (31, 31), (32, 56), (57, 57), (58, 88), (89, 10**6))  # Bounces, inclusive


@torch.no_grad()
def collect(trainer, steps: int, gen: torch.Generator) -> Dict[str, torch.Tensor]:
    """steps rollout steps of every env from a reset: h (N, d), old head output, labels."""
    from agent import sample

    env, agent = trainer.env, trainer.agent
    feats = []
    hook = agent.path.register_forward_pre_hook(lambda m, inp: feats.append(inp[0].detach().float()))
    obs = env.reset()
    B, dev = env.num_envs, env.device
    first = torch.ones(B, dtype=torch.bool, device=dev)
    ring = agent.init_ring(B, dev)
    rec = {k: [] for k in ("path_old", "path_mm", "path_ref", "bounces")}
    pool = env.pool is not None
    for _ in range(steps):
        rec["path_mm"].append(env.path.clone())
        rec["path_ref"].append((env.scen_ref if pool else env.ref_of(env.setup)).clone())
        rec["bounces"].append(env.bounces.clone())
        out, ring = agent.act(obs["image"], obs["action"], first, ring)
        rec["path_old"].append(out["path"].float())
        obs, _, term, trunc, _ = env.step(sample(out["logits"], gen))
        first = term | trunc
    hook.remove()
    d = {k: torch.cat(v) for k, v in rec.items()}
    d["h"] = torch.cat(feats)
    assert d["h"].shape[0] == d["path_mm"].shape[0], "one head call per step expected"
    return d


def target(path_mm: torch.Tensor) -> torch.Tensor:
    return torch.log(path_mm.clamp_min(1.0))


def train_head(agent_cfg, tr: Dict[str, torch.Tensor], va: Dict[str, torch.Tensor], dev,
               lr: float, patience: int, factor: float, reductions: int, max_epochs: int,
               batch: int, log=print):
    """New head on h -> log P, trained until the validation loss plateaus."""
    from agent import _mlp

    head = _mlp(agent_cfg.d_model, agent_cfg.head_hidden, 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    xt, yt = tr["h"], target(tr["path_mm"])
    xv, yv = va["h"], target(va["path_mm"])
    mu, sd = yt.mean(), yt.std().clamp_min(1e-6)  # Standardized target
    best, bad, n_red, state, converged = math.inf, 0, 0, None, False
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    for ep in range(max_epochs):
        head.train()
        perm = torch.randperm(xt.shape[0], generator=gen, device=dev)
        for i in range(0, xt.shape[0], batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(head(xt[idx]).squeeze(-1), (yt[idx] - mu) / sd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = F.mse_loss(head(xv).squeeze(-1), (yv - mu) / sd).item()
        if vl < best * (1 - 1e-3):
            best, bad = vl, 0
            state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            if n_red >= reductions:
                converged = True
                log(f"converged at epoch {ep}: val loss {best:.5f} (standardized), {n_red} lr reductions")
                break
            n_red, bad = n_red + 1, 0
            for g in opt.param_groups:
                g["lr"] *= factor
            log(f"epoch {ep}: val plateau, lr x {factor:g} ({n_red} of {reductions})")
    head.load_state_dict(state)
    head.eval()
    return head, (mu, sd), {"converged": converged, "epochs": ep + 1, "val_loss_std": best,
                            "lr_reductions": n_red}


def metrics(p_hat: torch.Tensor, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Errors of an absolute-path prediction p_hat (mm) on the frames of d."""
    p, ref = d["path_mm"].clamp_min(1.0), d["path_ref"]
    p_hat = p_hat.clamp_min(1.0)
    rel = (p_hat - p).abs() / p
    return {"n": int(p.numel()),
            "rel_err_median": rel.median().item(), "rel_err_mean": rel.mean().item(),
            "rel_err_p90": rel.quantile(0.9).item() if p.numel() > 1 else rel.mean().item(),
            "log_mae": (torch.log(p_hat) - torch.log(p)).abs().mean().item(),
            "mae_mm": (p_hat - p).abs().mean().item(),
            "mae_pref_units": ((p_hat - p).abs() / ref).mean().item()}


def compare(head, norm, te: Dict[str, torch.Tensor], mu_train: float) -> Dict[str, Dict]:
    mu, sd = norm
    with torch.no_grad():
        new = torch.exp(head(te["h"]).squeeze(-1) * sd + mu)
    old = te["path_old"] * te["path_ref"]  # Existing head: P / P_ref times the true P_ref
    const = torch.full_like(new, math.exp(mu_train))
    out = {}
    masks = {"all": torch.ones_like(te["bounces"], dtype=torch.bool)}
    for lo, hi in BINS:
        masks[f"b{lo}-{hi}" if hi < 10**6 else f"b{lo}+"] = (te["bounces"] >= lo) & (te["bounces"] <= hi)
    for name, m in masks.items():
        if not bool(m.any()):
            continue
        sub = {k: v[m] for k, v in te.items() if k != "h"}
        out[name] = {"new_logP": metrics(new[m], sub), "old_path_head": metrics(old[m], sub),
                     "constant": metrics(const[m], sub)}
    return out


def table(res: Dict[str, Dict]) -> str:
    rows = ["| frames | n | new head: median rel. err | old head: median rel. err | new: mean rel. err | "
            "old: mean rel. err | new: MAE (P_ref units) | old: MAE (P_ref units) | constant: median rel. err |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        n, o, c = r["new_logP"], r["old_path_head"], r["constant"]
        rows.append(f"| {name} | {n['n']} | {100 * n['rel_err_median']:.1f} % | {100 * o['rel_err_median']:.1f} % | "
                    f"{100 * n['rel_err_mean']:.1f} % | {100 * o['rel_err_mean']:.1f} % | "
                    f"{n['mae_pref_units']:.3f} | {o['mae_pref_units']:.3f} | {100 * c['rel_err_median']:.1f} % |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps-train", type=int, default=400)
    ap.add_argument("--steps-eval", type=int, default=100, help="Steps of each of val and test")
    ap.add_argument("--num-envs", type=int, default=0, help="0: the run's")
    ap.add_argument("--pool-size", type=int, default=0, help="0: the run's (smaller: smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--factor", type=float, default=0.3)
    ap.add_argument("--reductions", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    from ppo import Trainer, resume_config

    os.makedirs(args.out, exist_ok=True)
    argv = ["--no-compile", "--no-wandb"]
    cfg = resume_config(args.ckpt, argv)
    if args.num_envs:
        cfg.num_envs = args.num_envs
    if args.pool_size:
        cfg.pool_size = args.pool_size
    trainer = Trainer(cfg, os.path.join(args.out, "trainer"), build_eval=False, load_encoder=False)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    trainer.agent.load_state_dict(ck["agent"])
    for p in trainer.agent.parameters():
        p.requires_grad_(False)
    trainer.agent.eval()
    dev = trainer.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    print(f"checkpoint {args.ckpt}: iteration {ck['iteration']}, {ck['env_steps']} env steps; "
          f"{cfg.num_envs} envs, camera {cfg.camera} {cfg.camera_set!r}", flush=True)
    tr = collect(trainer, args.steps_train, gen)
    va = collect(trainer, args.steps_eval, gen)
    te = collect(trainer, args.steps_eval, gen)
    print(f"frames: train {tr['h'].shape[0]}, val {va['h'].shape[0]}, test {te['h'].shape[0]}", flush=True)
    head, norm, info = train_head(trainer.agent.cfg, tr, va, dev, args.lr, args.patience, args.factor,
                                  args.reductions, args.max_epochs, args.batch,
                                  log=lambda s: print(s, flush=True))
    res = compare(head, norm, te, float(norm[0]))
    out = {"ckpt": os.path.abspath(args.ckpt), "iteration": ck["iteration"], "env_steps": ck["env_steps"],
           "frames": {"train": int(tr["h"].shape[0]), "val": int(va["h"].shape[0]), "test": int(te["h"].shape[0])},
           "training": info, "test": res}
    with open(os.path.join(args.out, "probe_path.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = (f"# Path probe: {os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))} "
          f"iteration {ck['iteration']}\n\nNew head (log P, trained on the frozen core features) vs the "
          f"existing path head (P / P_ref x true P_ref) on {out['frames']['test']} test frames "
          f"(training: {info}).\n\n" + table(res) + "\n")
    with open(os.path.join(args.out, "probe_path.md"), "w") as f:
        f.write(md)
    torch.save({"head": head.state_dict(), "mu": float(norm[0]), "sd": float(norm[1])},
               os.path.join(args.out, "head.pt"))
    print(md, flush=True)
    if not info["converged"]:
        print("NOT CONVERGED: max epochs reached", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
