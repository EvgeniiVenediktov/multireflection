"""
tests/test_encoder_schedule.py - encoder training schedule of the PPO trainer (ppo.py
--encoder-epochs, --encoder-train-until), the pool-refresh default, run names, init-from
without --encoder-init

Run from rl/:  uv run python tests/test_encoder_schedule.py

    1. default settings (encoder_epochs 0, encoder_train_until 1) bit-identical to the
       trainer before the schedule (ppo.py of commit REF_COMMIT, from git): frozen,
       --no-encoder-freeze (bf16) and --unfreeze-at 1, three iterations each: agent,
       optimizer state, rollout buffers, ring, entropy coefficient, value norm, stats
    2. --encoder-epochs 1: the encoder after the full update equals the encoder after
       epoch 1 (captured when the buffer is re-encoded, once) while the core changed in
       the later epochs; the encoder's AdamW state is untouched after epoch 1 (no step,
       no decay); the recomputed z = a fresh no-grad encode by the updated encoder;
       target-KL stop after epoch 1: no re-encode; encoder_epochs >= epochs = default
    3. --encoder-train-until: encoder_trains() switches off exactly at the threshold
       (exact and rounded boundaries, with --unfreeze-at); across the switch (with
       --encoder-epochs 1): encoder parameters and AdamW state frozen afterwards, the core
       still trains, the stale ring is re-encoded once at the first frozen rollout and
       never after; stored z = the current encoder's z of the stored frames and update /
       rollout log-probs agree in every iteration
    4. resume just before, at and after the threshold reproduces the uninterrupted run
       bit for bit
    5. pool refresh default 1/16 (checkpoints keep their own value), run-name suffixes
       (_ee, _eu, _pr), resume overrides
    6. --init-from without --encoder-init: the encoder (and every other tensor) equals the
       checkpoint's, the run name's encoder origin comes from its encoder_info; with
       --encoder-init the encoder is replaced
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

from encoders import CNNEncoder, EncoderConfig  # noqa: E402
from ppo import RESUME_OVERRIDES, PPOConfig, Trainer, config_from_dict, run_name  # noqa: E402

DEV = torch.device("cuda")
REF_COMMIT = "1bdc7ff"  # ppo.py before the encoder schedule and the 1/16 pool refresh
FT_ENC = "cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384"


def small_cfg(mod=None, **kw):
    base = dict(setups="all", num_envs=16, rollout_len=24, max_steps=6, history_len=6,
                d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False, compile=False,
                wandb=False, minibatches=4, eval_every=0, seed=3, total_env_steps=10**6)
    base.update(kw)
    return (mod.PPOConfig if mod is not None else PPOConfig)(**base)


def stop_bias(tr, b: float = 1.5):
    """Raise the zero-step logit of both factors: P(not move) ~ 0.1 per step."""
    with torch.no_grad():
        nb = tr.agent.cfg.n_bins
        tr.agent.pi[-1].bias[0] += b
        tr.agent.pi[-1].bias[nb] += b


def make(cfg, mod=None):
    tr = (mod.Trainer if mod is not None else Trainer)(cfg, tempfile.mkdtemp(), build_eval=False)
    stop_bias(tr)
    return tr


def iterate(tr, n: int = 1, check=None):
    """n trainer iterations as in Trainer.train (refresh, rollout, update, counters);
    check(tr, stats) after each update. Returns the update stats."""
    S = tr.cfg.num_envs * tr.cfg.rollout_len
    out = []
    for _ in range(n):
        if tr.sampler is not None:
            tr.sampler.refresh(tr.iteration)
        tr.rollout()
        st = tr.update()
        if check is not None:
            check(tr, st)
        out.append(st)
        tr.iteration += 1
        tr.env_steps += S
    return out


def enc_params(tr):
    return [p for n, p in tr.agent.named_parameters() if n.startswith("encoder.")]


def core_params(tr):
    return [p for n, p in tr.agent.named_parameters() if not n.startswith("encoder.")]


def clone(ps):
    return [p.detach().clone() for p in ps]


def opt_state(tr, ps):
    return [{k: v.clone() for k, v in tr.opt.state.get(p, {}).items()} for p in ps]


def same(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def same_state(a, b) -> bool:
    return len(a) == len(b) and all(
        x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(a, b))


def snapshot(tr):
    """Every tensor of the trainer's state: agent, optimizer, buffers, ring, entropy
    coefficient, value norm."""
    d = {f"agent.{k}": v.detach().clone() for k, v in tr.agent.state_dict().items()}
    for gi, g in enumerate(tr.opt.param_groups):
        for pi, p in enumerate(g["params"]):
            for k, v in tr.opt.state.get(p, {}).items():
                d[f"opt.{gi}.{pi}.{k}"] = v.clone()
    d.update({f"buf.{k}": v.clone() for k, v in vars(tr.buf).items() if torch.is_tensor(v)})
    used = tr.buf.slot_t >= 0  # Unused truncation slots keep older rollouts' frames: not state
    d["buf.slot_frame"] = d["buf.slot_frame"] * used[..., None, None, None]
    d["buf.slot_a"] = d["buf.slot_a"] * used[..., None]
    d.update({f"ring.{k}": v.clone() for k, v in tr.ring._asdict().items()})
    d.update({f"vnorm.{k}": v.clone() for k, v in tr.vnorm.state_dict().items()})
    d["log_alpha"] = tr.log_alpha.clone()
    d["next_first"] = tr.next_first.clone()
    return d


def diff_keys(a, b):
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
    return [k for k in a if not torch.equal(a[k], b[k])]


def z_dev(tr) -> float:
    """max |z stored - encode(stored frame)| over the real positions of the buffer."""
    b = tr.buf
    with torch.no_grad():
        zf = tr.agent.encode(b.frames.flatten(0, 1)).view_as(b.z)
    return float(((zf - b.z).abs().amax(-1) * b.real.float()).max())


def ring_dev(tr) -> float:
    """max |ring z - encode(ring frame)| over the valid ring entries."""
    r = tr.ring
    with torch.no_grad():
        zf = tr.agent.encode(r.frame.flatten(0, 1)).view_as(r.z)
    return float(((zf - r.z).abs().amax(-1) * r.valid.float()).max())


# ── 1. Default settings vs the previous trainer ─────────────────────────────


def load_ref(tmp: str):
    """ppo.py of REF_COMMIT as module ppo_ref, or None without git / the commit."""
    try:
        src = subprocess.run(["git", "show", f"{REF_COMMIT}:rl/ppo.py"], cwd=RL_DIR, check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    path = os.path.join(tmp, "ppo_ref.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("ppo_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_identical(tmp: str):
    print("\n== 1. default settings vs the trainer of commit", REF_COMMIT, "==")
    ref = load_ref(tmp)
    if ref is None:
        print(f"  skipped: ppo.py of {REF_COMMIT} not available (git)")
        return
    for label, kw, trains in (("frozen", {}, (0, 0, 0)),
                              ("--no-encoder-freeze, bf16", dict(encoder_freeze=False, amp=True), (1, 1, 1)),
                              ("--unfreeze-at 1", dict(unfreeze_at=1), (0, 1, 1))):
        res = []
        for mod in (ref, None):
            tr = make(small_cfg(mod, seed=13, **kw), mod)
            st = iterate(tr, 3)
            res.append((snapshot(tr), st))
            del tr
        (sa, sta), (sb, stb) = res
        bad = diff_keys(sa, sb)
        common = [k for k in sta[0] if k in stb[0]]
        bad_st = [(i, k) for i in range(3) for k in common if not torch.equal(
            torch.as_tensor(sta[i][k]).cpu(), torch.as_tensor(stb[i][k]).cpu())]
        print(f"  {label}: 3 iterations, {len(sa)} state tensors identical: {not bad}; "
              f"{len(common)} update stats identical: {not bad_st}; new stats "
              f"{sorted(set(stb[0]) - set(sta[0]))}")
        assert not bad, bad[:10]
        assert not bad_st, bad_st[:10]
        assert [float(s["enc_epochs"]) for s in stb] == [t * float(s["epochs"]) for t, s in zip(trains, stb)]


# ── 2. --encoder-epochs ─────────────────────────────────────────────────────


def test_encoder_epochs():
    print("\n== 2. --encoder-epochs 1 ==")
    torch.backends.cudnn.allow_tf32 = False
    cfg = small_cfg(encoder_freeze=False, encoder_epochs=1, epochs=4, target_kl=1e9, seed=41)
    tr = make(cfg)
    iterate(tr)  # Context in the buffer, AdamW state of every tensor
    tr.rollout()
    enc_ps, core_ps = enc_params(tr), core_params(tr)
    enc0, core0 = clone(enc_ps), clone(core_ps)
    st_enc0, st_core0 = opt_state(tr, enc_ps), opt_state(tr, core_ps)
    cap = {"n": 0}
    orig = tr.reencode_buffer

    def hook():
        cap["n"] += 1
        cap["enc"], cap["core"] = clone(enc_ps), clone(core_ps)
        cap["st_enc"] = opt_state(tr, enc_ps)
        orig()

    tr.reencode_buffer = hook
    st = tr.update()
    tr.reencode_buffer = orig
    mb = cfg.minibatches
    enc_after_ep1 = same(clone(enc_ps), cap["enc"])
    enc_trained = not same(enc0, cap["enc"])
    n_core_moved = sum(not torch.equal(a, b) for a, b in zip(clone(core_ps), cap["core"]))
    st_enc_same = same_state(opt_state(tr, enc_ps), cap["st_enc"])
    steps_enc = {float(s["step"]) - float(s0["step"]) for s, s0 in zip(opt_state(tr, enc_ps), st_enc0)}
    steps_core = {float(s["step"]) - float(s0["step"]) for s, s0 in zip(opt_state(tr, core_ps), st_core0)}
    core_moved_total = sum(not torch.equal(a, b) for a, b in zip(clone(core_ps), core0))
    b = tr.buf
    Bm = b.B // mb
    with torch.no_grad():
        z_full = tr.agent.encode(b.frames.flatten(0, 1)).view_as(b.z)
        z_blk = tr.agent.encode(b.frames[:Bm].flatten(0, 1)).view_as(b.z[:Bm])
    dz_full = float((z_full - b.z).abs().max())
    print(f"  re-encodes {cap['n']}; epochs {int(st['epochs'])}, encoder epochs {int(st['enc_epochs'])}; "
          f"encoder after the update == after epoch 1: {enc_after_ep1} (changed in epoch 1: "
          f"{enc_trained}); core tensors changed after epoch 1: {n_core_moved} of {len(core_ps)}; "
          f"encoder AdamW state untouched after epoch 1: {st_enc_same}, steps this update: encoder "
          f"{steps_enc}, core {steps_core}")
    print(f"  recomputed z (all {b.z.shape[1]} positions incl. {b.C} context) vs a fresh no-grad "
          f"encode: same block {torch.equal(z_blk, b.z[:Bm])}, whole batch max |d| {dz_full:.1e}")
    assert cap["n"] == 1 and float(st["epochs"]) == 4 and float(st["enc_epochs"]) == 1
    assert enc_after_ep1 and enc_trained and st_enc_same
    assert n_core_moved > 0.9 * len(core_ps) and core_moved_total > 0.9 * len(core_ps)
    assert steps_enc == {float(mb)} and steps_core == {4.0 * mb}
    assert torch.equal(z_blk, b.z[:Bm]) and dz_full < 1e-5
    assert tr.ring_stale

    # Next rollout: the ring re-encoded, the stored z = the current encoder's
    iterate(tr, 1)
    dz = z_dev(tr)
    print(f"  next iteration: stored z vs the current encoder max |d| {dz:.1e}")
    assert dz < 1e-5

    # Target-KL stop after epoch 1: no re-encode
    tk = make(replace(cfg, target_kl=0.0))
    iterate(tk)
    tk.rollout()
    calls = []
    tk.reencode_buffer = lambda: calls.append(1)
    s2 = tk.update()
    print(f"  target-KL stop: epochs {int(s2['epochs'])}, encoder epochs {int(s2['enc_epochs'])}, "
          f"re-encodes {len(calls)}")
    assert float(s2["epochs"]) == 1 and float(s2["enc_epochs"]) == 1 and not calls

    # encoder_epochs >= epochs: the default (all epochs)
    snaps = []
    for ee in (0, 4, 7):
        t = make(replace(cfg, encoder_epochs=ee, seed=43))
        iterate(t, 2)
        snaps.append(snapshot(t))
    ok = not diff_keys(snaps[0], snaps[1]) and not diff_keys(snaps[0], snaps[2])
    print(f"  encoder_epochs 4 / 7 (>= epochs) identical to 0: {ok}")
    assert ok
    torch.backends.cudnn.allow_tf32 = True


# ── 3. --encoder-train-until ────────────────────────────────────────────────


def sched_cfg(**kw):
    """8 iterations of 384 env steps; encoder trained (--encoder-epochs 1) until 0.5 ->
    frozen from iteration 4."""
    base = dict(encoder_freeze=False, encoder_epochs=1, encoder_train_until=0.5,
                total_env_steps=8 * 16 * 24, seed=61, lr_warmup=1, encoder_lr_scale=1.0)
    base.update(kw)
    return small_cfg(**base)


def test_train_until():
    print("\n== 3. --encoder-train-until ==")
    torch.backends.cudnn.allow_tf32 = False
    S = 16 * 24
    for kw, stop, want in (
            (dict(), 4, [1, 1, 1, 1, 0, 0, 0, 0]),
            (dict(encoder_train_until=0.375), 3, [1, 1, 1, 0, 0, 0, 0, 0]),  # exactly 3 S
            (dict(encoder_train_until=0.45), 4, [1, 1, 1, 1, 0, 0, 0, 0]),  # 3.6 S
            (dict(encoder_train_until=0.3, total_env_steps=10 * S), 3, [1, 1, 1, 0, 0, 0]),  # 3 S (float 0.3)
            (dict(encoder_freeze=True, unfreeze_at=2), 4, [0, 0, 1, 1, 0, 0, 0, 0]),
            (dict(encoder_train_until=1.0), None, [1] * 8),
            (dict(encoder_freeze=True), 4, [0] * 8)):
        tr = Trainer(sched_cfg(**kw), tempfile.mkdtemp(), build_eval=False)
        got = [int(tr.encoder_trains(i)) for i in range(len(want))]
        assert tr.encoder_stop_iteration() == stop and got == want, (kw, tr.encoder_stop_iteration(), got)
        del tr
    print("  encoder_trains(): off from env steps >= round(F x total): 0.5 x 8 S -> iteration 4, "
          "0.375 -> 3 (exact), 0.45 -> 4, 0.3 x 10 S -> 3; with --unfreeze-at 2: 2..3; F 1: never")

    tr = make(sched_cfg())
    enc_ps, core_ps = enc_params(tr), core_params(tr)
    n_reenc = {"n": 0}
    orig_ring = tr.agent.reencode_ring

    def count_ring(ring):
        n_reenc["n"] += 1
        return orig_ring(ring)

    tr.agent.reencode_ring = count_ring
    rows = []
    for it in range(7):
        e0, c0, s0 = clone(enc_ps), clone(core_ps), opt_state(tr, enc_ps)
        stale_before = tr.ring_stale
        rdev = ring_dev(tr) if it > 0 else 0.0
        n0 = n_reenc["n"]
        tr.rollout()
        dz = z_dev(tr)
        st = tr.update()
        tr.iteration += 1
        tr.env_steps += S
        rows.append(dict(it=it, trains=tr.encoder_trains(it), stale=stale_before, ring_dev=rdev,
                         reenc=n_reenc["n"] - n0, dz=dz, ratio=float(st["ratio_dev_first_mb"]),
                         enc_ep=int(st["enc_epochs"]), enc_moved=not same(e0, clone(enc_ps)),
                         enc_state=same_state(s0, opt_state(tr, enc_ps)),
                         core_moved=not same(c0, clone(core_ps))))
    for r in rows:
        print(f"  it {r['it']}: trains {int(r['trains'])}, ring stale {int(r['stale'])} (ring z vs encoder "
              f"{r['ring_dev']:.1e}), ring re-encoded {r['reenc']}; stored z vs encoder {r['dz']:.1e}, "
              f"|ratio - 1| first mb {r['ratio']:.1e}; encoder epochs {r['enc_ep']}, encoder changed "
              f"{int(r['enc_moved'])}, encoder AdamW state same {int(r['enc_state'])}, core changed "
              f"{int(r['core_moved'])}")
    for r in rows:
        tr_ = r["it"] < 4
        assert r["trains"] == tr_ and r["enc_ep"] == int(tr_) and r["enc_moved"] == tr_
        assert r["enc_state"] == (not tr_) and r["core_moved"]
        assert r["dz"] < 1e-5 and r["ratio"] < 1e-4
        assert r["reenc"] == (1 if tr_ or r["it"] == 4 else 0)
        assert r["stale"] == (1 <= r["it"] <= 4)
        if r["it"] in (1, 2, 3, 4):
            assert r["ring_dev"] > 1e-5, "the ring is stale after an encoder update"
        if r["it"] > 4:
            assert r["ring_dev"] < 1e-5
    print("  encoder frozen from iteration 4 (parameters and AdamW state), the core trains on; the "
          "stale ring re-encoded once at iteration 4, never after; z consistent in every iteration")
    torch.backends.cudnn.allow_tf32 = True


# ── 4. Resume across the threshold ──────────────────────────────────────────


def test_resume_threshold():
    print("\n== 4. resume across the threshold ==")
    d = tempfile.mkdtemp()
    try:
        cfg = sched_cfg(seed=71)
        a = Trainer(cfg, d, build_eval=False)
        stop_bias(a)
        for it in range(6):
            if it in (3, 4, 5):
                a.save(f"it{it}.pt")
            iterate(a)
        ref = snapshot(a)
        for it in (3, 4, 5):
            bt = Trainer(cfg, d, build_eval=False)
            bt.load(os.path.join(d, f"it{it}.pt"))
            stale, trains = bt.ring_stale, bt.encoder_trains()
            iterate(bt, 6 - it)
            bad = diff_keys(ref, snapshot(bt))
            print(f"  resumed at iteration {it} (encoder trains {int(trains)}, ring stale {int(stale)}) "
                  f"-> iteration 6: {len(ref)} state tensors identical to the uninterrupted run: {not bad}")
            assert trains == (it < 4) and stale == (1 <= it <= 4)
            assert not bad, bad[:10]
            del bt
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── 5. Pool refresh default, run names ──────────────────────────────────────


def test_names():
    print("\n== 5. pool refresh default, run names, resume overrides ==")
    assert PPOConfig().pool_refresh == 0.0625
    old = config_from_dict({"setups": "full", "scenarios": "pool", "pool_refresh": 0.125})
    assert old.pool_refresh == 0.125
    assert {"encoder_epochs", "encoder_train_until"} <= set(RESUME_OVERRIDES)
    assert "pool_refresh" not in RESUME_OVERRIDES
    job = os.environ.get("SLURM_JOB_ID", "local")
    base = PPOConfig(encoder_freeze=False, init_from="/x/4407578/best_snapshot_it3800.pt")
    info = {"name": FT_ENC}
    stem = "ppo_cnn-ftcont_k16_cont20-400-plr_b1024_s0_ft4407578_e2e"
    for kw, sfx in ((dict(), ""), (dict(encoder_epochs=1, encoder_train_until=0.3), "_ee1_eu0.3"),
                    (dict(encoder_epochs=1, encoder_train_until=0.8), "_ee1_eu0.8"),
                    (dict(pool_refresh=0.125), "_pr0.125"),
                    (dict(encoder_epochs=2, pool_refresh=0.125, name_tag="r4b"), "_ee2_pr0.125_r4b")):
        nm = run_name(replace(base, **kw), info)
        assert nm == f"{stem}{sfx}_{job}", nm
    nm = run_name(PPOConfig(setups="all", pool_refresh=0.125), None)
    assert "_pr" not in nm, nm  # Not a pool run
    print(f"  default pool_refresh 0.0625, a checkpoint's 0.125 kept; e.g. {stem}_ee1_eu0.3_{job}; "
          f"_pr<F> for a non-default refresh (pool runs only)")


# ── 6. init-from without encoder-init ───────────────────────────────────────


def test_init_from_encoder():
    print("\n== 6. --init-from without --encoder-init ==")
    d = tempfile.mkdtemp()
    try:
        src_dir = os.path.join(d, "4410759")
        a = Trainer(small_cfg(seed=81), src_dir, build_eval=False)
        with torch.no_grad():
            for p in a.agent.parameters():
                p.add_(0.01 * torch.randn_like(p))
        a.encoder_info = {"path": "/x/enc.pt", "name": FT_ENC, "step": 8000, "n_setups": 1}
        a.save("best.pt")
        path = os.path.join(src_dir, "best.pt")
        b = Trainer(small_cfg(seed=82, init_from=path), os.path.join(d, "b"), build_eval=False)
        sa, sb = a.agent.state_dict(), b.agent.state_dict()
        enc_same = all(torch.equal(sa[k], sb[k]) for k in sa if k.startswith("encoder."))
        all_same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        print(f"  encoder tensors equal to the checkpoint's: {enc_same}; all tensors: {all_same}; "
              f"name {b.name}")
        assert enc_same and all_same and b.encoder_info["name"] == FT_ENC
        assert b.name.startswith("ppo_cnn-ftcont_k6_set4_b16_s82_ft4410759_"), b.name
        # With --encoder-init the encoder is replaced (applied after init-from)
        torch.manual_seed(5)
        enc = CNNEncoder(EncoderConfig())
        ep = os.path.join(d, "enc.pt")
        torch.save({"cfg": asdict(EncoderConfig()), "name": FT_ENC, "step": 8000,
                    "model": {f"encoder.{k}": v for k, v in enc.state_dict().items()}}, ep)
        c = Trainer(small_cfg(seed=82, init_from=path, encoder_init=ep), os.path.join(d, "c"),
                    build_eval=False)
        se, sc = enc.state_dict(), c.agent.state_dict()
        replaced = all(torch.equal(se[k].to(DEV), sc[f"encoder.{k}"]) for k in se)
        rest = all(torch.equal(sa[k], sc[k]) for k in sa if not k.startswith("encoder."))
        print(f"  with --encoder-init: encoder replaced {replaced}, the rest from the checkpoint {rest}")
        assert replaced and rest
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    torch.backends.cudnn.deterministic = True  # Bitwise comparisons of encoder backward passes
    torch.backends.cudnn.benchmark = False
    tmp = tempfile.mkdtemp()
    try:
        test_default_identical(tmp)
        test_encoder_epochs()
        test_train_until()
        test_resume_threshold()
        test_names()
        test_init_from_encoder()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nall tests passed")
