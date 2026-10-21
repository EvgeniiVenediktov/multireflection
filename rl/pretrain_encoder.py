"""
pretrain_encoder.py - Single-frame encoder pretraining (PLAN phase 5, Level 1)

Supervised training on online FrameData batches (randomized renders, fresh every step)
of a FrameModel (CNN or ViT encoder to z = 256, heads on z only).

Run from rl/:
    uv run python pretrain_encoder.py --arch cnn --seed 0
    uv run python pretrain_encoder.py --arch vit --patch 8 --seed 0
    uv run python pretrain_encoder.py --arch vit --patch 4 --seed 0
    uv run python pretrain_encoder.py --arch cnn --steps 8000 --setups all   # 4 setups
    uv run python pretrain_encoder.py --scenarios 16384                      # grid scenarios
    uv run python pretrain_encoder.py --init <run>/model.pt --scenarios 16384 \
        --sep-range 20 400 --laser-range -8 8                    # fine-tune, continuous (D7)
Outputs in runs/encoder/<name>/ (or <--runs-dir>/<name>/): config.json, train.csv,
val.csv, model.pt.
Evaluation: eval_encoder.py.

Data: --setups measured / all (the four geometry.SETUPS); --scenarios N: N scenarios plus
the four named setups, from the atlas grid (scenarios.sample_grid; run name _grid<N>) or,
with --sep-range, continuous (D7: scenarios.sample_valid_scenarios over scenarios.ContRanges:
separation --sep-range, laser pitch / yaw --laser-range, laser offset uniform over the M1
hole disk of --laser-offset-radius; only beams that reach M2; run name _<ranges tag>x<N>,
e.g. _cont20-400x16384; laser angles from a uniform M2 landing point by default,
--laser-mode angle for uniform angles; no laser-offset jitter, the offsets already cover
the hole disk); --scenario-seed seeds the scenarios (eval_encoder.py holds out another
seed).

Fine-tuning (--init PATH): model config and weights from a pretrain_encoder.py checkpoint
(encoder and the map / count / ellipse heads, every tensor whose name and shape match;
skipped and missing tensors are printed and kept in config.json and model.pt), the
outputs equal the source's before the first step; fresh optimizer. Defaults then: peak lr
3e-4 (CNN; ViT 1.5e-4, 0.3 x the from-scratch values), warmup 100 steps, 8000 steps
(cosine to 1 %): a third of the from-scratch peak keeps the start close to the source
while 8k steps of new-range frames (half the source run's 16k) adapt it. Run name
_ft-<origin>, origin = the source run's data marker (e.g. _ft-grid16384_cont20-400x16384).

Camera mode (--camera): 3-channel inputs from render_camera.CameraRenderer (its
CameraRanges augmentation config: per-effect probabilities and ranges; the edge scatter
uses the sim's miss point) instead of the 1-channel randomized renders; targets in the
camera frame (CameraRenderer.targets: spot map of every M2 spot inside the field, count
of all M2 spots), no ellipse. The stem takes 3 channels (EncoderConfig.in_ch); a
1-channel stem from --init / --init-agent is copied to the 3 channels at 1/3, so a gray
image copied x 3 gives the source's output. --init-agent PATH takes the encoder from a
ppo.py checkpoint (the policy's encoder; --init then gives the heads and the config).
Validation on a fixed camera set (val/) and on the same states rendered without any
effect (val_clean/). Run name marker _cam; with --init-agent _ft-<--init-agent-tag>.

Convergence: --converge trains until the validation loss (augmented set in camera mode)
plateaus, --steps is only a cap: after the warmup the lr stays at its peak; after
--patience validations without a relative improvement of --plateau-tol over the best it
is multiplied by --plateau-factor; after --max-reductions reductions the next plateau
stops the run (converged). Reaching the cap first is a failure: exit code 3, so a Slurm
afterok chain does not start from an unconverged encoder. Without --converge (fixed
cosine schedule) the run ends with a check of the last quarter: a validation-loss drop of
more than 2 % there is reported as not converged (config.json convergence, exit code 0).

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
import re
import time
import uuid
from dataclasses import asdict, replace
from typing import Any, Dict

import torch
import torch.nn.functional as F

from encoders import EncoderConfig, FrameModel, count_params, model_name
from frame_data import DataConfig, FrameData
from geometry import SETUPS
from render_camera import CameraRenderer, chunks

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
    if "ellipse" in batch:
        m = batch["ellipse_mask"].float()
        e = F.smooth_l1_loss(
            out["ellipse"].float(), batch["ellipse"] / ELL_SCALE_MM, beta=0.02, reduction="none"
        ).mean(1)
        l_ell = (e * m).sum() / m.sum().clamp_min(1.0)
    else:  # Camera mode: no ellipse target
        l_ell = torch.zeros((), device=p.device)
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


# ── Fine-tuning init ────────────────────────────────────────────────────────


def load_init(path: str) -> Dict[str, Any]:
    """A pretrain_encoder.py checkpoint (CPU tensors)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def init_config(ck: Dict[str, Any]) -> EncoderConfig:
    """EncoderConfig of a pretrain_encoder.py checkpoint."""
    return EncoderConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})


def init_model(model: FrameModel, ck: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """Load every tensor of the checkpoint ck whose name and shape match into model
    (encoder and heads); returns the info: path, name, step, loaded, skipped (with the
    reason), missing (model tensors not in the checkpoint)."""
    own = model.state_dict()
    sd, loaded, skipped = {}, [], {}
    for k, v in ck["model"].items():
        if k not in own:
            skipped[k] = "not in the model"
        elif tuple(own[k].shape) != tuple(v.shape):
            skipped[k] = f"shape {tuple(v.shape)} != {tuple(own[k].shape)}"
        else:
            sd[k] = v
            loaded.append(k)
    missing = sorted(k for k in own if k not in sd)
    model.load_state_dict({**own, **sd})
    return {"path": os.path.abspath(path) if path else "", "name": ck.get("name"),
            "step": ck.get("step"), "loaded": len(loaded), "skipped": skipped, "missing": missing}


def expand_stem(sd: Dict[str, torch.Tensor], in_ch: int) -> Dict[str, torch.Tensor]:
    """State dict with a 1-channel stem / patch embedding copied to in_ch channels at
    1 / in_ch (a gray image copied x in_ch then gives the 1-channel output)."""
    out = dict(sd)
    for k in ("encoder.stem.weight", "encoder.embed.weight", "stem.weight", "embed.weight"):
        if k in out and out[k].shape[1] == 1 and in_ch > 1:
            out[k] = out[k].repeat(1, in_ch, 1, 1) / in_ch
    return out


def init_agent_encoder(model: FrameModel, path: str) -> Dict[str, Any]:
    """Encoder weights of a ppo.py checkpoint (the policy's encoder) into model.encoder,
    stem expanded to the model's channels; returns the info (path, run name, iteration,
    env steps)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = ck["agent_cfg"]["encoder"]
    own = asdict(model.cfg)
    for k in ("arch", "z_dim", "img_res", "cnn_widths", "cnn_blocks", "n_keypoints",
              "gn_groups", "mlp_hidden"):
        v = tuple(enc[k]) if isinstance(enc[k], list) else enc[k]
        assert v == own[k], f"--init-agent encoder {k}: {v} != {own[k]}"
    sd = {k[len("encoder."):]: v for k, v in ck["agent"].items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(expand_stem(sd, model.cfg.in_ch))
    return {"path": os.path.abspath(path), "name": ck.get("name"), "iteration": ck.get("iteration"),
            "env_steps": ck.get("env_steps"), "tensors": len(sd)}


def init_origin(name: str) -> str:
    """Data marker of a source run name (after _s<seed>_, '_' -> '-'), e.g. grid16384,
    set4; meas for a measured-cell run."""
    m = re.search(r"_s\d+_(.+)$", name or "")
    return m.group(1).replace("_", "-") if m else "meas"


# ── Camera mode data ────────────────────────────────────────────────────────


def camera_sim(data: FrameData, gen: torch.Generator, n: int) -> Dict[str, torch.Tensor]:
    """Sim outputs (with the per-hit outputs and the miss point) of n fresh FrameData
    states: setups, tilts, physical jitter."""
    setup = data.sample_setups(n, gen)
    st = data.states(data.sample_tilts(n, gen, setup=setup), gen, setup=setup)
    return data.sim.simulate(st, store_world=False)


def camera_frames(cam: CameraRenderer, out: Dict[str, torch.Tensor], p: Dict[str, torch.Tensor],
                  gen: torch.Generator, chunk: int) -> Dict[str, torch.Tensor]:
    """CameraRenderer images (n, 3, H, W) uint8 and camera-frame targets (spot_map,
    count) of the sim outputs out under the parameters p, in chunks of chunk frames."""
    hm, lxy = out["hit_mirror"], out["hit_local_xy"]
    miss = (out["miss_mirror"], out["miss_local_xy"], out["intensity"])
    parts: Dict[str, list] = {"image": [], "spot_map": [], "count": []}
    for s in chunks(hm.shape[0], chunk):
        ps = {k: v[s] for k, v in p.items()}
        parts["image"].append(cam.render(hm[s], lxy[s], ps, gen, tuple(t[s] for t in miss)))
        for k, v in cam.targets(hm[s], lxy[s], ps).items():
            parts[k].append(v)
    return {k: torch.cat(v) for k, v in parts.items()}


# ── Validation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model: FrameModel, val: Dict[str, torch.Tensor], bs: int = 256,
             image_key: str = "image_id") -> Dict[str, float]:
    """Mean losses and count accuracy on the fixed validation set (in-distribution)."""
    model.eval()
    n = val[image_key].shape[0]
    sums = {"total": 0.0, "map": 0.0, "count": 0.0, "ellipse": 0.0, "count_acc": 0.0}
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in val.items() if torch.is_tensor(v)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b[image_key])
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
    ap.add_argument("--steps", type=int, default=None, help="Default 6000 (--init: 8000)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None,
                    help="Peak LR (default cnn 1e-3, vit 5e-4; --init: cnn 3e-4, vit 1.5e-4)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=None, help="Default 300 (--init: 100)")
    ap.add_argument("--init", default="", help="Fine-tune from this pretrain_encoder.py "
                    "checkpoint (model config and weights)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--w-map", type=float, default=1.0)
    ap.add_argument("--w-count", type=float, default=0.1)
    ap.add_argument("--w-ell", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--setups", choices=("measured", "all"), default="measured",
                    help="Frames of the measured cell only, or of all geometry.SETUPS "
                         "(uniform; run name suffix _set4)")
    ap.add_argument("--scenarios", type=int, default=0,
                    help="Frames of this many scenarios sampled uniformly from the atlas grid "
                         "(or continuously with --sep-range) plus the four named setups "
                         "(overrides --setups); 0 = --setups")
    ap.add_argument("--scenario-seed", type=int, default=0)
    ap.add_argument("--sep-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="--scenarios: continuous scenarios with this separation range (mm) "
                         "instead of grid points")
    ap.add_argument("--laser-range", type=float, nargs=2, default=[-8.0, 8.0],
                    metavar=("MIN", "MAX"), help="Continuous: laser pitch and yaw range (deg)")
    ap.add_argument("--laser-offset-radius", type=float, default=1.5,
                    help="Continuous: laser offset disk radius around the hole centre (mm); "
                         "0 = the measured offsets")
    ap.add_argument("--laser-mode", choices=("landing", "angle"), default="landing",
                    help="Continuous: laser angles from a uniform M2 landing point (bounded by "
                         "--laser-range) or uniform in angle")
    ap.add_argument("--no-amp", action="store_true", help="fp32 instead of bf16 autocast")
    ap.add_argument("--no-compile", action="store_true", help="eager model (no torch.compile)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=1024)
    ap.add_argument("--tag", default="", help="Suffix for the run name")
    ap.add_argument("--run-tag", default="",
                    help="Short name e<N>-<short>: prefix of the run name and W&B id, a W&B tag")
    ap.add_argument("--camera", action="store_true",
                    help="Camera mode: 3-channel CameraRenderer inputs, camera-frame targets")
    ap.add_argument("--init-agent", default="",
                    help="Encoder from this ppo.py checkpoint (the policy's encoder)")
    ap.add_argument("--init-agent-tag", default="agent",
                    help="Short name of that run for the run name, e.g. r6")
    ap.add_argument("--render-chunk", type=int, default=256,
                    help="Camera mode: frames per render call")
    ap.add_argument("--converge", action="store_true",
                    help="Train until the validation loss plateaus; --steps is the cap (exit 3 if reached)")
    ap.add_argument("--patience", type=int, default=6,
                    help="--converge: validations without improvement before a plateau")
    ap.add_argument("--plateau-tol", type=float, default=0.005,
                    help="--converge: relative val-loss improvement that counts")
    ap.add_argument("--plateau-factor", type=float, default=0.3, help="--converge: lr factor on a plateau")
    ap.add_argument("--max-reductions", type=int, default=3,
                    help="--converge: lr reductions; the next plateau stops the run")
    ap.add_argument("--runs-dir", default=RUNS_DIR, help="Parent of the run directory")
    ap.add_argument("--wandb", action="store_true", help="Log to W&B (project rl-encoder)")
    return ap.parse_args()


def cont_ranges_of(args: argparse.Namespace):
    """scenarios.ContRanges of the continuous mode (--sep-range given), else None."""
    if args.scenarios <= 0 or args.sep_range is None:
        return None
    from scenarios import cont_ranges

    return cont_ranges(args.sep_range, args.laser_range, args.laser_offset_radius,
                       args.laser_mode)


def run_name(args: argparse.Namespace, cfg: EncoderConfig, init_name: str = "") -> str:
    """Display / directory name from the key config, built at start:
    <model>_w<width>_lr<lr>_b<batch>_<steps>_s<seed>[_ft-<origin>][_<data>][_cam][_<tag>]
    (<steps>: conv<cap> with --converge),
    data grid<N>, <ranges tag>x<N> (continuous) or set4; _ft-<origin> with --init
    (_ft-<--init-agent-tag> with --init-agent); _cam in camera mode; prefixed with
    <--run-tag>_ when given."""
    k = args.steps / 1000
    steps = f"{k:g}k" if args.steps >= 1000 else str(args.steps)
    if getattr(args, "converge", False):
        steps = "conv" + steps  # The cap; the run stops at the plateau
    name = f"{model_name(cfg)}_w{cfg.width if cfg.arch == 'vit' else cfg.cnn_widths[-1]}" \
           f"_lr{args.lr:g}_b{args.batch}_{steps}_s{args.seed}"
    if getattr(args, "init_agent", ""):  # Optional: callers may pass a partial Namespace
        name += f"_ft-{args.init_agent_tag}"
    elif args.init:
        name += f"_ft-{init_origin(init_name)}"
    rng = cont_ranges_of(args)
    if rng is not None:
        name += f"_{rng.tag()}x{args.scenarios}"
    elif args.scenarios > 0:
        name += f"_grid{args.scenarios}"
    elif args.setups == "all":
        name += f"_set{len(SETUPS)}"
    if getattr(args, "camera", False):
        name += "_cam"
    name += f"_{args.tag}" if args.tag else ""
    rt = getattr(args, "run_tag", "")  # Optional: callers may pass a partial Namespace
    if rt:
        assert re.fullmatch(r"[A-Za-z0-9-]+", rt), f"--run-tag {rt!r}: letters, digits, '-'"
        name = f"{rt}_{name}"
    return name


def data_config(setups: str, batch_size: int = 256) -> DataConfig:
    """DataConfig of the --setups choice."""
    return DataConfig(batch_size=batch_size, setups=SETUPS if setups == "all" else ())


def main():
    args = parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    ck_init = load_init(args.init) if args.init else None
    if ck_init is not None:
        cfg = init_config(ck_init)
        if cfg.arch != args.arch or (cfg.arch == "vit" and cfg.patch != args.patch):
            print(f"--init: model config from the checkpoint ({model_name(cfg)}), not --arch "
                  f"{args.arch}", flush=True)
        args.arch = cfg.arch
    else:
        cfg = EncoderConfig(arch=args.arch, patch=args.patch)
    if args.camera:
        cfg = replace(cfg, in_ch=3)
    ft = ck_init is not None or bool(args.init_agent)
    if args.lr is None:
        args.lr = (3e-4 if ft else 1e-3) if args.arch == "cnn" else (1.5e-4 if ft else 5e-4)
    if args.steps is None:
        args.steps = 8000 if ft else 6000
    if args.warmup is None:
        args.warmup = 100 if ft else 300
    name = run_name(args, cfg, ck_init.get("name", "") if ck_init is not None else "")
    run_dir = os.path.join(args.runs_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    data_info: Dict[str, Any] = {}
    if args.scenarios > 0:
        rng = cont_ranges_of(args)
        if rng is not None:
            from scenarios import sample_valid_scenarios, scenario_geometry

            pts, rej = sample_valid_scenarios(args.scenarios, rng, args.scenario_seed, device=dev)
            setups = tuple(SETUPS) + tuple(scenario_geometry(p) for p in pts)
            stages = {k: v for k, v in rej.items() if k not in ("positions", "next")}
            data_info = {"ranges": asdict(rng), "scenarios": args.scenarios,
                         "scenario_seed": args.scenario_seed, "stages": stages}
            kind = (f"continuous scenarios ({rng.tag()}, {rng.mode} mode, laser bounds "
                    f"{rng.pitch} deg, offsets r {rng.offset_radius} mm; {rej['candidates']} "
                    f"candidates: angles out of bounds "
                    f"{100 * rej['angle_rejected'] / rej['candidates']:.1f} %, clipped "
                    f"{100 * rej['clipped'] / rej['candidates']:.1f} %, missed M2 "
                    f"{100 * rej['missed'] / rej['candidates']:.1f} %)")
        else:
            from find_setups import setup_geometry
            from scenarios import sample_grid

            pts = sample_grid(args.scenarios, args.scenario_seed)
            setups = tuple(SETUPS) + tuple(setup_geometry(*p) for p in pts)
            kind = "grid scenarios"
        # Continuous offsets cover the hole disk: no offset jitter (it would leave the disk)
        jit = {"jitter_laser_offset": 0.0} if rng is not None and rng.offset_radius > 0 else {}
        data = FrameData(replace(data_config("measured", args.batch), setups=setups,
                                 scan_step=0.05, **jit), dev)
        print(f"{args.scenarios} {kind} (seed {args.scenario_seed}) + {len(SETUPS)} named "
              f"setups; {int((data.cell_count == 0).sum())} without >= 4 bounce cells", flush=True)
    else:
        data = FrameData(data_config(args.setups, args.batch), dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1_000_003 * (args.seed + 1))
    cam = val_clean = None
    if args.camera:
        cam = CameraRenderer(data.base)
        if not args.no_compile:
            cam.compile()
        gen_v = torch.Generator(device=dev)
        gen_v.manual_seed(VAL_SEED)
        out_v = camera_sim(data, gen_v, args.val_n)
        val = camera_frames(cam, out_v, cam.sample_params(args.val_n, gen_v), gen_v, args.render_chunk)
        val_clean = camera_frames(cam, out_v, cam.neutral_params(args.val_n), gen_v, args.render_chunk)
        del out_v
    else:
        val = data.eval_set(args.val_n, VAL_SEED)

    model = FrameModel(cfg).to(dev)
    init_info = None
    if ck_init is not None:
        init_info = init_model(model, {**ck_init, "model": expand_stem(ck_init["model"], cfg.in_ch)},
                               args.init)
        print(f"init from {init_info['path']} ({init_info['name']}, step {init_info['step']}): "
              f"{init_info['loaded']} tensors loaded; skipped {sorted(init_info['skipped'])}; "
              f"missing {init_info['missing']}", flush=True)
        del ck_init
    if args.init_agent:
        agent_info = init_agent_encoder(model, args.init_agent)
        init_info = {**(init_info or {}), "agent": agent_info}
        print(f"encoder from {agent_info['path']} ({agent_info['name']}, iteration "
              f"{agent_info['iteration']}, {agent_info['env_steps']} env steps): "
              f"{agent_info['tensors']} tensors", flush=True)
    fwd = model if args.no_compile else torch.compile(model)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95), fused=True)
    conv = {"mult": 1.0, "best": math.inf, "bad": 0, "reductions": 0, "converged": False}
    if args.converge:  # Warmup, then the peak lr times the plateau reductions
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / args.warmup) * conv["mult"])
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, args.steps))
    val_hist = []  # (step, val loss)
    n_enc, n_heads = count_params(model.encoder), count_params(model.heads)
    info = {"name": name, "args": vars(args), "encoder": asdict(cfg),
            "params_encoder": n_enc, "params_heads": n_heads,
            "data": {"batch_size": args.batch, "cell_frac": data.cfg.cell_frac,
                     "n_cells": int(data.cells.shape[0]),
                     "setups": [g.name for g in data.setups],
                     "n_cells_per_setup": data.cell_count.tolist(), **data_info},
            "init": init_info, "camera": asdict(cam.ranges) if cam is not None else None}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"{name}: encoder {n_enc / 1e6:.2f} M params, heads {n_heads / 1e6:.2f} M -> {run_dir}")

    wb = None
    if args.wandb:
        import wandb

        rt = args.run_tag
        wb = wandb.init(project="rl-encoder", name=name, config=info, dir=run_dir,
                        id=f"{rt}-{uuid.uuid4().hex[:8]}" if rt else None, tags=[rt] if rt else None)

    train_f = open(os.path.join(run_dir, "train.csv"), "w", newline="")
    val_f = open(os.path.join(run_dir, "val.csv"), "w", newline="")
    tw = csv.writer(train_f)
    vw = csv.writer(val_f)
    tw.writerow(["step", "time_s", "lr", "loss", "map", "count", "ellipse", "grad_norm",
                 "grad_norm_max", "skipped"])
    VAL_KEYS = ("total", "map", "count", "ellipse", "count_acc")
    vw.writerow(["step", "time_s", "loss", "map", "count", "ellipse", "count_acc"]
                + (["clean_" + k for k in ("loss", "map", "count", "ellipse", "count_acc")]
                   if cam is not None else []))

    acc = {k: torch.zeros((), device=dev) for k in ("total", "map", "count", "ellipse", "gn")}
    gn_max = torch.zeros((), device=dev)
    skipped = 0
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        if cam is not None:
            batch = camera_frames(cam, camera_sim(data, gen, args.batch),
                                  cam.sample_params(args.batch, gen), gen, args.render_chunk)
        else:
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
            key = "image" if cam is not None else "image_id"
            v = validate(model, val, image_key=key)
            vc = validate(model, val_clean, image_key=key) if val_clean is not None else None
            el = time.time() - t0
            vw.writerow([step, round(el, 1)] + [v[k] for k in VAL_KEYS]
                        + ([vc[k] for k in VAL_KEYS] if vc is not None else []))
            val_f.flush()
            if wb is not None:
                wb.log({f"val/{k}": x for k, x in v.items()}, step=step)
                if vc is not None:
                    wb.log({f"val_clean/{k}": x for k, x in vc.items()}, step=step)
            clean = (f"  clean: map {vc['map']:.4f} count acc {vc['count_acc']:.3f}"
                     if vc is not None else "")
            print(f"step {step:6d} {el / 60:5.1f} min  val loss {v['total']:.4f} map {v['map']:.4f} "
                  f"count {v['count']:.3f} (acc {v['count_acc']:.3f}) ell {v['ellipse']:.4f}{clean}  "
                  f"skipped {skipped}", flush=True)
            torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "step": step,
                        "name": name, "data_setups": [g.name for g in data.setups],
                        "data": data_info, "init": init_info,
                        "camera": asdict(cam.ranges) if cam is not None else None},
                       os.path.join(run_dir, "model.pt"))
            val_hist.append((step, v["total"]))
            if args.converge:
                if v["total"] < conv["best"] * (1.0 - args.plateau_tol):
                    conv["best"], conv["bad"] = v["total"], 0
                else:
                    conv["bad"] += 1
                if conv["bad"] >= args.patience:
                    if conv["reductions"] >= args.max_reductions:
                        conv["converged"] = True
                        print(f"converged at step {step}: no {100 * args.plateau_tol:g} % val-loss "
                              f"improvement in {args.patience} validations after "
                              f"{conv['reductions']} lr reductions", flush=True)
                        break
                    conv["mult"] *= args.plateau_factor
                    conv["reductions"] += 1
                    conv["bad"] = 0
                    print(f"step {step}: val-loss plateau, lr x {args.plateau_factor:g} (reduction "
                          f"{conv['reductions']} of {args.max_reductions})", flush=True)

    elapsed = time.time() - t0
    if args.converge:
        converged = conv["converged"]
        verdict = {"mode": "plateau", "converged": converged, "lr_reductions": conv["reductions"],
                   "best_val_loss": conv["best"], "steps": val_hist[-1][0] if val_hist else 0}
    else:  # Fixed schedule: the val-loss drop over the last quarter of the steps
        q = [x for s_, x in val_hist if s_ >= 0.75 * args.steps]
        drop = (q[0] - q[-1]) / q[0] if len(q) > 1 and q[0] > 0 else 0.0
        converged = drop <= 0.02
        verdict = {"mode": "fixed", "converged": converged, "last_quarter_drop": drop}
    info.update(train_time_s=round(elapsed, 1), skipped_steps=skipped,
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), convergence=verdict)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(info, f, indent=1)
    train_f.close()
    val_f.close()
    if wb is not None:
        wb.finish()
    print(f"done in {elapsed / 60:.1f} min, peak {info['peak_mem_gb']} GB; convergence {verdict}",
          flush=True)
    if not converged:
        print("NOT CONVERGED: " + ("the step cap was reached before the validation loss plateaued"
                                   if args.converge else
                                   f"val loss fell {100 * verdict['last_quarter_drop']:.1f} % over the "
                                   f"last quarter"), flush=True)
        if args.converge:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
