"""
probe_path.py - Probe: how well the frozen agent's core features predict the absolute path

A ppo.py checkpoint is loaded with its run config (env with its renderer, scenario pool,
agent); every weight stays frozen. Rollouts in that env (sampled actions, as in training)
record per step the core output h of the current frame (the input of the heads), the
existing path head's output (P / P_ref) and the labels of that frame: path P (mm), P_ref,
bounces. Train / val / test splits are separate rollout segments (new episodes).

Two new heads with the path head's shape (MLP d_model -> head_hidden -> outputs) are
trained on h alone, on the same frames:
    logP          log P (P in mm, floored at 1 mm), MSE on the standardized target
    logP+bounces  the same plus a bounce-count classifier (0 .. the env's bounce cap,
                  cross-entropy x --w-bounces) from the same hidden layer
AdamW; the learning rate drops on validation plateaus (validation loss of the head's own
objective) and the training stops after the last one (train to convergence); exit code 3
if a head reaches --max-epochs first. On the test split the heads are compared with the
existing path head in the same units: absolute path (the existing head's P / P_ref times
the true P_ref), relative error |P_hat - P| / P, and error in P / P_ref units (the eval
metric head_path_mae), overall and per bounce bin; a constant (mean log P of train) gives
the scale. The bounce classifier: exact and within-2 accuracy, mean absolute error.

    python probe_path.py --ckpt <ppo best.pt> --out <dir> [--steps-train 400] [--steps-eval 100]
Outputs: probe_path.json, probe_path.md, heads.pt (the new heads' weights).
"""

import argparse
import json
import math
import os
from typing import Dict, Optional

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


def head_loss(out: torch.Tensor, y_std: torch.Tensor, bounces: Optional[torch.Tensor],
              w_bounces: float) -> torch.Tensor:
    """MSE on the standardized log P (output 0) [+ w_bounces x CE on the bounce logits]."""
    loss = F.mse_loss(out[:, 0], y_std)
    if bounces is not None:
        loss = loss + w_bounces * F.cross_entropy(out[:, 1:], bounces)
    return loss


def train_head(agent_cfg, tr: Dict[str, torch.Tensor], va: Dict[str, torch.Tensor], dev,
               n_cls: int, w_bounces: float, args, log=print):
    """New head on h: log P (n_cls = 0) or log P + bounce logits (n_cls classes), trained
    until the validation loss plateaus."""
    from agent import _mlp

    head = _mlp(agent_cfg.d_model, agent_cfg.head_hidden, 1 + n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    yt, yv = target(tr["path_mm"]), target(va["path_mm"])
    mu, sd = yt.mean(), yt.std().clamp_min(1e-6)  # Standardized target
    cls = lambda d: d["bounces"].long().clamp(0, n_cls - 1) if n_cls else None  # noqa: E731
    bt, bv = cls(tr), cls(va)
    best, bad, n_red, state, converged = math.inf, 0, 0, None, False
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    for ep in range(args.max_epochs):
        head.train()
        perm = torch.randperm(tr["h"].shape[0], generator=gen, device=dev)
        for i in range(0, perm.shape[0], args.batch):
            idx = perm[i:i + args.batch]
            loss = head_loss(head(tr["h"][idx]), (yt[idx] - mu) / sd,
                             bt[idx] if bt is not None else None, w_bounces)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = head_loss(head(va["h"]), (yv - mu) / sd, bv, w_bounces).item()
        if vl < best * (1 - 1e-3):
            best, bad = vl, 0
            state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            if n_red >= args.reductions:
                converged = True
                log(f"converged at epoch {ep}: val loss {best:.5f}, {n_red} lr reductions")
                break
            n_red, bad = n_red + 1, 0
            for g in opt.param_groups:
                g["lr"] *= args.factor
            log(f"epoch {ep}: val plateau, lr x {args.factor:g} ({n_red} of {args.reductions})")
    head.load_state_dict(state)
    head.eval()
    return head, (mu, sd), {"converged": converged, "epochs": ep + 1, "val_loss": best,
                            "lr_reductions": n_red}


def path_metrics(p_hat: torch.Tensor, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Errors of an absolute-path prediction p_hat (mm) on the frames of d."""
    p, ref = d["path_mm"].clamp_min(1.0), d["path_ref"]
    p_hat = p_hat.clamp_min(1.0)
    rel = (p_hat - p).abs() / p
    return {"n": int(p.numel()),
            "rel_err_median": rel.median().item(), "rel_err_mean": rel.mean().item(),
            "log_mae": (torch.log(p_hat) - torch.log(p)).abs().mean().item(),
            "mae_mm": (p_hat - p).abs().mean().item(),
            "mae_pref_units": ((p_hat - p).abs() / ref).mean().item()}


def bounce_metrics(b_hat: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    err = (b_hat - b.long()).abs().float()
    return {"acc": (err == 0).float().mean().item(), "acc_within2": (err <= 2).float().mean().item(),
            "mae": err.mean().item()}


def compare(heads: Dict[str, tuple], te: Dict[str, torch.Tensor], mu_train: float) -> Dict[str, Dict]:
    preds, bounce_pred = {}, {}
    with torch.no_grad():
        for name, (head, (mu, sd)) in heads.items():
            out = head(te["h"])
            preds[name] = torch.exp(out[:, 0] * sd + mu)
            if out.shape[1] > 1:
                bounce_pred[name] = out[:, 1:].argmax(1)
    preds["old_path_head"] = te["path_old"] * te["path_ref"]  # P / P_ref times the true P_ref
    preds["constant"] = torch.full_like(te["path_mm"], math.exp(mu_train))
    masks = {"all": torch.ones_like(te["bounces"], dtype=torch.bool)}
    for lo, hi in BINS:
        masks[f"b{lo}-{hi}" if hi < 10**6 else f"b{lo}+"] = (te["bounces"] >= lo) & (te["bounces"] <= hi)
    out = {}
    for name, m in masks.items():
        if not bool(m.any()):
            continue
        sub = {k: v[m] for k, v in te.items() if k != "h"}
        out[name] = {k: path_metrics(p[m], sub) for k, p in preds.items()}
        for k, bp in bounce_pred.items():
            out[name][f"{k}/bounces"] = bounce_metrics(bp[m], sub["bounces"])
    return out


def tables(res: Dict[str, Dict]) -> str:
    rows = ["### Path (relative error of the absolute path; MAE in P_ref units)", "",
            "| frames | n | logP: median | logP+bounces: median | old head: median | logP: mean | "
            "logP+bounces: mean | old head: mean | logP: MAE P_ref | logP+bounces: MAE P_ref | "
            "old head: MAE P_ref | constant: median |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        a, b, o, c = r["logP"], r["logP+bounces"], r["old_path_head"], r["constant"]
        rows.append(f"| {name} | {a['n']} | {100 * a['rel_err_median']:.1f} % | {100 * b['rel_err_median']:.1f} % | "
                    f"{100 * o['rel_err_median']:.1f} % | {100 * a['rel_err_mean']:.1f} % | "
                    f"{100 * b['rel_err_mean']:.1f} % | {100 * o['rel_err_mean']:.1f} % | "
                    f"{a['mae_pref_units']:.3f} | {b['mae_pref_units']:.3f} | {o['mae_pref_units']:.3f} | "
                    f"{100 * c['rel_err_median']:.1f} % |")
    rows += ["", "### Bounce count (logP+bounces head)", "",
             "| frames | exact | within 2 | MAE (bounces) |", "|---|---|---|---|"]
    for name, r in res.items():
        bm = r.get("logP+bounces/bounces")
        if bm:
            rows.append(f"| {name} | {100 * bm['acc']:.1f} % | {100 * bm['acc_within2']:.1f} % | {bm['mae']:.2f} |")
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
    ap.add_argument("--w-bounces", type=float, default=0.1, help="Bounce cross-entropy weight")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--factor", type=float, default=0.3)
    ap.add_argument("--reductions", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    from ppo import Trainer, resume_config

    os.makedirs(args.out, exist_ok=True)
    cfg = resume_config(args.ckpt, ["--no-compile", "--no-wandb"])
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
    n_cls = trainer.env.cfg.max_bounces + 1
    heads, info = {}, {}
    for name, nc in (("logP", 0), ("logP+bounces", n_cls)):
        print(f"== head {name}", flush=True)
        head, norm, inf = train_head(trainer.agent.cfg, tr, va, dev, nc, args.w_bounces, args,
                                     log=lambda s: print(s, flush=True))
        heads[name], info[name] = (head, norm), inf
    res = compare(heads, te, float(heads["logP"][1][0]))
    out = {"ckpt": os.path.abspath(args.ckpt), "iteration": ck["iteration"], "env_steps": ck["env_steps"],
           "frames": {"train": int(tr["h"].shape[0]), "val": int(va["h"].shape[0]), "test": int(te["h"].shape[0])},
           "w_bounces": args.w_bounces, "bounce_classes": n_cls, "training": info, "test": res}
    with open(os.path.join(args.out, "probe_path.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = (f"# Path probe: {os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))} "
          f"iteration {ck['iteration']}\n\nNew heads on the frozen core features vs the existing path head "
          f"(P / P_ref x true P_ref) on {out['frames']['test']} test frames ({out['frames']['train']} train). "
          f"Training: {info}.\n\n" + tables(res) + "\n")
    with open(os.path.join(args.out, "probe_path.md"), "w") as f:
        f.write(md)
    torch.save({k: {"head": h.state_dict(), "mu": float(n[0]), "sd": float(n[1])} for k, (h, n) in heads.items()},
               os.path.join(args.out, "heads.pt"))
    print(md, flush=True)
    if not all(v["converged"] for v in info.values()):
        print("NOT CONVERGED: max epochs reached", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
