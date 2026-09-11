"""Deployment-oriented comparison table for ResNet-18 and MLP (SimpleFC) tilt regressors.

One row per model from a JSON manifest (a list of objects):
  label, checkpoint (optional), resolution, split ("random" | "holdout"), eval_dir,
  train_summary (optional), wandb_run (optional W&B run id, e.g. "l40s-3870308"),
  arch (optional, "resnet18" | "mlp"; only used without a checkpoint, else detected from its keys).

Relative paths are resolved against the current directory, then against the repo root.

Columns: params (trainable), weights_MB (float32), RAM_peak_MB (measured peak RSS increase for
CPU float32 batch-1 inference including checkpoint loading, fresh subprocess, 4 threads, cached
per resolution), RAM_forward_MB (peak RSS increase during the forward passes only),
best_val_mse, and clean / noisy / occlusion / combined_harsh errors (mean ± std over
perturbation seeds) from eval_dir/sweep.csv (random split, utils/eval_sweep.py) or
eval_dir/metrics.csv (holdout split, holdout/eval_test.py).

Usage:
  python utils/model_table.py --manifest eval_results/model_table_manifest.json \
      --out eval_results/model_table.md
"""
import argparse
import csv
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WANDB_PATH = "e-venediktov-university-of-pittsburgh/multireflection"
THREADS = 4
MIB = 2 ** 20

METRIC_NAME = {"random": "closed-loop final error, 0.1 deg grid",
               "holdout": "open-loop TEST-hole error"}
# split -> (file, count column, mean column, std column)
EVAL_FILE = {"random": ("sweep.csv", "starts", "final_angular_error_mean", "final_angular_error_mean_std"),
             "holdout": ("metrics.csv", "n", "ang_err_mean", "ang_err_mean_std")}

# Runs in a fresh interpreter; prints one JSON line.
RAM_PROBE = r"""
import gc, json, os, sys
def status():
    out = {}
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(('VmHWM:', 'VmRSS:')):
                k, v = line.split(':')
                out[k] = int(v.split()[0])  # kB
    return out
import torch
torch.set_num_threads(int(sys.argv[3]))
sys.path.insert(0, sys.argv[4])
from app.inference import resnet18, SimpleFC, model_from_state_dict  # model definitions (module also imports cv2/skimage/config)
gc.collect()
try:
    with open('/proc/self/clear_refs', 'w') as f:
        f.write('5')  # reset VmHWM to the current RSS
    hwm_reset = True
except OSError:
    hwm_reset = False
base = status()
res, ckpt, arch = int(sys.argv[1]), sys.argv[2], sys.argv[5]
if ckpt:
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    model, arch, _ = model_from_state_dict(sd)
    del sd
else:
    model = SimpleFC(res * res, 2) if arch == 'mlp' else resnet18(output_dim=2)
model = model.float().eval()
gc.collect()
loaded = status()  # peak so far = build + load
if hwm_reset:
    with open('/proc/self/clear_refs', 'w') as f:
        f.write('5')  # second reset: isolate the forward-pass peak
    pre_fwd = status()
x = torch.rand(1, 1, res, res, dtype=torch.float32)
with torch.inference_mode():
    for _ in range(5):
        y = model(x)
after = status()
peak = max(loaded['VmHWM'], after['VmHWM'])
print(json.dumps({'base_rss_kb': base['VmRSS'], 'base_hwm_kb': base['VmHWM'], 'peak_kb': peak,
                  'load_peak_kb': loaded['VmHWM'], 'loaded_rss_kb': loaded['VmRSS'],
                  'fwd_delta_kb': (after['VmHWM'] - pre_fwd['VmRSS']) if hwm_reset else None,
                  'end_rss_kb': after['VmRSS'], 'hwm_reset': hwm_reset, 'threads': torch.get_num_threads(),
                  'torch': torch.__version__, 'out_shape': list(y.shape)}))
"""


def resolve(path):
    if not path:
        return None
    for cand in (path, os.path.join(REPO, path)):
        if os.path.exists(cand):
            return cand
    return None


def fmt_sig(v, digits=3):
    return f"{v:.{digits}g}"


def entry_arch(entry, ckpt):
    """Architecture of a manifest row: detected from the checkpoint keys, else the optional "arch" field."""
    if ckpt:
        import torch
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        return "mlp" if any(k.startswith("layers.") for k in sd) else "resnet18"
    return entry.get("arch", "resnet18")


def count_params(ckpt, arch, resolution):
    """-> (trainable parameters, buffer numel, float32 state dict MB, source)."""
    import torch
    sys.path.insert(0, REPO)
    from app.inference import resnet18, SimpleFC, model_from_state_dict
    if ckpt:
        model, arch, _ = model_from_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        src = f"{arch} with checkpoint loaded"
    else:
        model = SimpleFC(resolution * resolution, 2) if arch == "mlp" else resnet18(output_dim=2)
        src = f"{arch} architecture (no checkpoint)"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(b.numel() for b in model.buffers())
    float_bytes = sum(t.numel() * 4 for t in model.state_dict().values() if t.is_floating_point())
    return n_params, n_buffers, float_bytes / MIB, src


def measure_ram(resolution, ckpt, arch="resnet18"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(THREADS))
    proc = subprocess.run([sys.executable, "-c", RAM_PROBE, str(resolution), ckpt or "", str(THREADS), REPO, arch],
                          capture_output=True, text=True, env=env, cwd=REPO)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"RAM probe failed for {arch} r{resolution}: {proc.stderr.strip()[-500:]}")
    m = json.loads(lines[-1])
    m["delta_mb"] = (m["peak_kb"] - m["base_rss_kb"]) / 1024
    m["fwd_mb"] = m["fwd_delta_kb"] / 1024 if m["fwd_delta_kb"] is not None else None
    m["checkpoint"] = ckpt
    return m


def best_val(entry, ckpt, eval_dir):
    cands = [resolve(entry.get("train_summary"))]
    if ckpt:
        cands.append(os.path.join(os.path.dirname(ckpt), "train_summary.json"))
    if eval_dir:
        cands.append(os.path.join(eval_dir, "train_summary.json"))
    for c in cands:
        if c and os.path.isfile(c):
            with open(c) as f:
                v = json.load(f).get("best_val_loss")
            if v is not None:
                return float(v), f"{os.path.relpath(c, REPO)} best_val_loss"
    run_id = entry.get("wandb_run")
    if run_id:
        try:
            import wandb
            run = wandb.Api(timeout=60).run(f"{WANDB_PATH}/{run_id}")
            for key in ("best_loss", "best_val_loss"):
                v = run.summary.get(key)
                if v is not None:
                    return float(v), f"W&B {run_id} summary {key}"
            return None, f"W&B {run_id}: no best_loss/best_val_loss in summary"
        except Exception as e:
            return None, f"W&B {run_id} lookup failed ({type(e).__name__}: {e})"
    return None, "no train_summary.json and no wandb_run"


def read_eval(split, eval_dir, conditions):
    fname, count_col, mean_col, std_col = EVAL_FILE[split]
    path = os.path.join(eval_dir, fname) if eval_dir else None
    if not path or not os.path.isfile(path):
        return None, f"missing {fname}"
    with open(path) as f:
        rows = {r["condition"]: r for r in csv.DictReader(f)}
    out, problems = {}, []
    for key, cond in conditions.items():
        r = rows.get(cond)
        if r is None:
            problems.append(f"condition {cond} not in {fname}")
            continue
        mean = float(r[mean_col])
        std = float(r[std_col]) if r.get(std_col) not in (None, "") else None
        out[key] = {"mean": mean, "std": std, "seeds": int(r["seeds"]), "count": int(r[count_col])}
    return out, "; ".join(problems)


def cell_err(d, key):
    v = d.get(key) if d else None
    if v is None:
        return "n/a"
    if key == "clean" or v["std"] is None:
        return f"{v['mean']:.5f}"
    return f"{v['mean']:.5f} ± {v['std']:.2g}"


def join_unique(vals):
    vals = [str(v) for v in vals]
    uniq = list(dict.fromkeys(vals))
    return uniq[0] if len(uniq) == 1 else "/".join(vals) if vals else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="markdown path; a .csv with the same stem is written too")
    ap.add_argument("--noise-condition", default="noise_0.20")
    ap.add_argument("--occlusion-condition", default="occlusion_2x30")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(*a):
        if not args.quiet:
            print(*a, flush=True)

    with open(args.manifest) as f:
        entries = json.load(f)
    conditions = {"clean": "clean", "noisy": args.noise_condition,
                  "occlusion": args.occlusion_condition, "combined_harsh": "combined_harsh"}

    # RAM: one probe per (arch, resolution), preferring an entry with an available checkpoint.
    archs = [entry_arch(e, resolve(e.get("checkpoint"))) for e in entries]
    ram_ckpt = {}
    for e, a in zip(entries, archs):
        key = (a, int(e["resolution"]))
        ck = resolve(e.get("checkpoint"))
        if key not in ram_ckpt or (ram_ckpt[key] is None and ck):
            ram_ckpt[key] = ck
    ram = {}
    for a, r in sorted(ram_ckpt):
        log(f"measuring RAM {a} r{r} ...")
        ram[(a, r)] = measure_ram(r, ram_ckpt[(a, r)], a)

    header = ["label", "arch", "split", "resolution", "params", "weights_MB", "RAM_peak_MB", "RAM_forward_MB", "best_val_mse",
              "metric", "runs", "starts", "clean", f"noisy ({args.noise_condition})",
              f"occlusion ({args.occlusion_condition})", "combined_harsh"]
    rows, row_notes = [], []
    for e, arch in zip(entries, archs):
        label, split, r = e["label"], e["split"], int(e["resolution"])
        if split not in EVAL_FILE:
            raise ValueError(f"{label}: split must be random or holdout, got {split!r}")
        ck = resolve(e.get("checkpoint"))
        eval_dir = resolve(e.get("eval_dir"))
        params, n_buffers, weights_mb, psrc = count_params(ck, arch, r)
        mse, msrc = best_val(e, ck, eval_dir)
        ev, eprob = read_eval(split, eval_dir, conditions) if eval_dir else (None, f"eval_dir not found: {e.get('eval_dir')}")
        perturbed = [ev[k]["seeds"] for k in ("noisy", "occlusion", "combined_harsh") if ev and k in ev]
        counts = [ev[k]["count"] for k in conditions if ev and k in ev]
        fwd = ram[(arch, r)]["fwd_mb"]
        rows.append([label, arch, split, r, f"{params / 1e6:.2f} M", f"{weights_mb:.1f}",
                     f"{ram[(arch, r)]['delta_mb']:.1f}",
                     f"{fwd:.1f}" if fwd is not None else "n/a",
                     fmt_sig(mse) if mse is not None else "n/a", METRIC_NAME[split],
                     join_unique(perturbed), join_unique(counts),
                     cell_err(ev, "clean"), cell_err(ev, "noisy"), cell_err(ev, "occlusion"),
                     cell_err(ev, "combined_harsh")])
        n = f"{label}: params {params} (buffers {n_buffers}) from {psrc}; best_val_mse from {msrc}"
        if e.get("checkpoint") and not ck:
            n += f"; checkpoint not found: {e['checkpoint']}"
        if eprob:
            n += f"; {eprob}"
        row_notes.append(n)
        log(f"{label}: done")

    probe0 = next(iter(ram.values()))
    ram_desc = ", ".join(
        f"{a} r{r}: peak {m['peak_kb'] / 1024:.0f} MB absolute (baseline {m['base_rss_kb'] / 1024:.0f} MB, "
        f"load peak {m['load_peak_kb'] / 1024:.0f} MB, RSS after load {m['loaded_rss_kb'] / 1024:.0f} MB, "
        f"forward-only peak increase {(m['fwd_delta_kb'] or 0) / 1024:.1f} MB, "
        f"RSS after inference {m['end_rss_kb'] / 1024:.0f} MB, weights {'loaded from checkpoint' if m['checkpoint'] else 'random init'})"
        for (a, r), m in sorted(ram.items()))
    hwm_note = ("VmHWM was reset to the baseline RSS via /proc/self/clear_refs before building the model"
                if all(m["hwm_reset"] for m in ram.values())
                else "VmHWM could not be reset for some probes, so import peaks may be included")
    notes = (
        f"**Notes.** Units: MB = 2^20 bytes; errors in degrees. `arch` is detected from the checkpoint keys "
        f"(resnet18, or mlp = the 4-layer fully connected SimpleFC on the flattened r*r input). `params` is the number "
        f"of trainable parameters (ResNet-18: independent of input size because of the adaptive average pool; MLP: "
        f"dominated by the first layer, r*r*1024); the state dicts additionally hold BatchNorm buffer values "
        f"(running mean/var, num_batches_tracked, per row below), not counted in `params`. "
        f"`weights_MB` is the float32 size of the state dict's floating-point tensors (parameters plus BatchNorm "
        f"running statistics). RAM columns are measured, not estimated: for each "
        f"resolution a fresh Python subprocess (CPU only, torch {probe0['torch']}, torch.set_num_threads({THREADS}) "
        f"to match the 4-core Raspberry Pi 4) imports torch and the model definitions (app/inference.py), then records "
        f"VmRSS from /proc/self/status as the baseline; {hwm_note}. It then builds the model, loads the checkpoint "
        f"(state dict freed after loading), runs 5 forwards in torch.inference_mode on a float32 (1, 1, r, r) input "
        f"and records VmHWM. `RAM_peak_MB` = overall peak VmHWM (build, load and forwards) minus baseline VmRSS; at "
        f"batch 1 up to r256 it is set by weight loading (model plus the transient state dict copy), so it barely "
        f"depends on resolution. `RAM_forward_MB` = peak increase during the forward passes only: after loading, "
        f"VmHWM is reset again and the column is VmHWM after the forwards minus VmRSS before them; this is the "
        f"activation/workspace part that scales with resolution. "
        f"Absolute values: {ram_desc}. The probe ran "
        f"on this x86-64 machine with a CUDA build of torch, so absolute RSS (dominated by the torch import) is not "
        f"representative of a Pi; the increase is the model-specific part, though ARM CPU kernels may use different "
        f"workspace memory. `best_val_mse` is the best validation MSE (training loss on normalized targets) from "
        f"train_summary.json, else the W&B run summary. Evaluation cells are mean ± std, std over perturbation seeds "
        f"(`runs`); clean has a single run and shows only the mean. Random split: closed-loop replay over the 0.1 deg "
        f"grid (utils/eval_sweep.py, eval_dir/sweep.csv), metric = final angular error after the loop, averaged over "
        f"`starts` start positions. Holdout split: open-loop angular error of single predictions on the TEST hole "
        f"positions (holdout/eval_test.py, eval_dir/metrics.csv), averaged over `starts` TEST positions. The two "
        f"metrics are not directly comparable. Conditions: noisy = {args.noise_condition}, occlusion = "
        f"{args.occlusion_condition}, combined_harsh. Per-row sources: " + "; ".join(row_notes) + "."
    )

    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out_md = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n\n" + notes + "\n")
    out_csv = os.path.splitext(out_md)[0] + ".csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    if args.quiet:
        print(out_md)
    else:
        print(f"wrote {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
