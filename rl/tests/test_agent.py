"""
tests/test_agent.py - Agent (agent.py) and PPO trainer (ppo.py) checks

Run from rl/:  uv run python tests/test_agent.py

    1. act() vs evaluate_windows() across episode boundaries (terminations, truncations,
       the context carried over from the previous rollout): the rollout's own z (frozen
       encoder) gives the same logits / values / log-probs to fp32 round-off; z
       recomputed from the stored frames (the unfrozen path) agrees within fp tolerance;
       final_windows() == act() pushing the final frame; bf16 deviation reported
    2. factorized log-prob / entropy vs the joint 121-way categorical; Gumbel-max
       sampling frequencies; uniform start (small policy init)
    3. GAE on a hand-made sequence with a termination and a truncation vs a scalar loop
    4. label alignment: every stored frame re-rendered from its stored (tilt, scenario
       parameters) labels, path label = re-simulated path / P_ref (post-reset labels at
       resets)
    5. checkpoint save / resume: the resumed trainer continues the same rollout and gives
       an identical next update on the same data
    6. no host syncs in rollout() (torch.cuda.set_sync_debug_mode("error")), eager and
       compiled; syncs left in update(): one per epoch (KL early stop)
"""

import math
import os
import shutil
import sys
import tempfile

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

from agent import Agent, AgentConfig, entropy, log_prob, sample  # noqa: E402
from ppo import PPOConfig, Trainer, gae  # noqa: E402

DEV = torch.device("cuda")


def small_cfg(**kw) -> PPOConfig:
    base = dict(setups="all", num_envs=16, rollout_len=24, max_steps=6, history_len=6,
                d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False, compile=False,
                wandb=False, minibatches=4, eval_every=0, seed=3, total_env_steps=10**6)
    base.update(kw)
    return PPOConfig(**base)


def stop_bias(tr: Trainer, b: float = 1.5):
    """Raise the zero-step logit of both factors: P(not move) ~ 0.1 per step."""
    with torch.no_grad():
        nb = tr.agent.cfg.n_bins
        tr.agent.pi[-1].bias[0] += b
        tr.agent.pi[-1].bias[nb] += b


class Recorder:
    """Wraps agent.act to record outputs and rings of every call."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.orig = agent.act
        self.calls = []
        agent.act = self

    def __call__(self, image, prev, first, ring):
        out, new = self.orig(image, prev, first, ring)
        self.calls.append(({k: v.clone() for k, v in out.items()}, new))
        return out, new

    def restore(self):
        self.agent.act = self.orig


# ── Tests ───────────────────────────────────────────────────────────────────


def test_agreement():
    print("\n== 1. act() vs evaluate_windows() ==")
    torch.backends.cudnn.allow_tf32 = False  # fp32 convolutions (the default is TF32)
    tr = Trainer(small_cfg(), tempfile.mkdtemp(), build_eval=False)
    stop_bias(tr)
    ag, b = tr.agent, tr.buf
    n_term = n_trunc = 0
    for r in range(3):
        rec = Recorder(ag)
        tr.rollout()
        rec.restore()
        T = b.T
        assert len(rec.calls) == T + 1  # T steps + bootstrap peek
        with torch.no_grad():
            ev = ag.evaluate_windows(b.z, b.a_in, b.real, b.first)
            lg = torch.stack([rec.calls[t][0]["logits"] for t in range(T)], 1)
            vv = torch.stack([rec.calls[t][0]["value"] for t in range(T)], 1)
            d_logit = float((ev["logits"] - lg).abs().max())
            d_value = float((ev["value"] - vv).abs().max())
            d_logp = float((log_prob(ev["logits"], b.actions) - b.logp).abs().max())
            for k in ("path", "tilt_mu", "scen_mu", "scen_logvar"):
                a = torch.stack([rec.calls[t][0][k] for t in range(T)], 1)
                assert float((ev[k] - a).abs().max()) < 1e-5, k
            # Unfrozen path: z recomputed from the stored frames
            z2 = ag.encode(b.frames.flatten(0, 1)).view_as(b.z)
            ev2 = ag.evaluate_windows(z2, b.a_in, b.real, b.first)
            d2 = float((ev2["logits"] - lg).abs().max())
            dv2 = float((ev2["value"] - vv).abs().max())
            # Final windows vs act() pushing the final frame onto the ring of step t
            nf, d_fin = 0, 0.0
            for s in range(b.S):
                for e in range(b.B):
                    t = int(b.slot_t[e, s])
                    if t < 0:
                        continue
                    ring_t = rec.calls[t][1]
                    fr = b.slot_frame[:, s]
                    out, _ = rec.orig(fr, b.slot_a[:, s], torch.zeros(b.B, dtype=torch.bool, device=DEV),
                                      ring_t)
                    d_fin = max(d_fin, abs(float(out["value"][e]) - float(b.v_final[e, t])))
                    nf += 1
            # v_last: the peek
            d_last = float((rec.calls[T][0]["value"] - b.v_last).abs().max())
        nt, nr = int(b.term.sum()), int(b.trunc.sum())
        n_term += nt
        n_trunc += nr
        ctx = int(b.real[:, :b.C].sum())
        print(f"  rollout {r}: {nt} terminations, {nr} truncations, {ctx} valid context entries; "
              f"max |d| logits {d_logit:.1e}, value {d_value:.1e}, logp {d_logp:.1e}; "
              f"recomputed z: logits {d2:.1e}, value {dv2:.1e}; final windows ({nf}) {d_fin:.1e}; "
              f"v_last {d_last:.1e}")
        assert d_logit < 1e-5 and d_value < 1e-5 and d_logp < 1e-5
        assert d2 < 1e-4 and dv2 < 1e-4
        assert d_fin < 1e-5 and d_last < 1e-6
        if r > 0:
            assert ctx > 0, "context from the previous rollout"
    assert n_term > 0 and n_trunc > 0
    torch.backends.cudnn.allow_tf32 = True

    # bf16 deviation (production setting), frozen z
    tr2 = Trainer(small_cfg(amp=True, seed=5), tempfile.mkdtemp(), build_eval=False)
    stop_bias(tr2)
    tr2.rollout()
    tr2.rollout()
    b2 = tr2.buf
    with torch.no_grad():
        ev = tr2.agent.evaluate_windows(b2.z, b2.a_in, b2.real, b2.first)
        d = float((log_prob(ev["logits"], b2.actions) - b2.logp).abs().max())
    print(f"  bf16 autocast: max |logp(update) - logp(rollout)| {d:.1e} (frozen z)")
    assert d < 2e-2


def test_distribution():
    print("\n== 2. factorized categorical ==")
    g = torch.Generator(device=DEV)
    g.manual_seed(0)
    nb = 11
    logits = torch.randn(64, 2, nb, generator=g, device=DEV) * 2
    joint = (torch.log_softmax(logits[:, 0], -1)[:, :, None]
             + torch.log_softmax(logits[:, 1], -1)[:, None, :]).flatten(1)  # (64, 121)
    cat = torch.distributions.Categorical(logits=joint)
    acts = torch.randint(0, nb, (64, 2), generator=g, device=DEV)
    lp_ref = cat.log_prob(acts[:, 0] * nb + acts[:, 1])
    d_lp = float((log_prob(logits, acts) - lp_ref).abs().max())
    d_ent = float((entropy(logits) - cat.entropy()).abs().max())
    print(f"  vs joint 121-way Categorical: max |d log-prob| {d_lp:.1e}, |d entropy| {d_ent:.1e}")
    assert d_lp < 1e-5 and d_ent < 1e-4
    n = 200_000
    one = logits[:1].expand(n, -1, -1)
    s = sample(one, g)
    freq = torch.bincount(s[:, 0] * nb + s[:, 1], minlength=nb * nb).float() / n
    p = joint[0].exp()
    err = float(((freq - p).abs() / (p * (1 - p) / n).sqrt().clamp_min(1e-12)).max())
    print(f"  Gumbel-max: {n} samples, max |freq - p| / sigma {err:.2f}")
    assert err < 5.0
    ag = Agent(AgentConfig(amp=False)).to(DEV)
    img = torch.randint(0, 256, (8, 1, 64, 64), dtype=torch.uint8, device=DEV)
    with torch.no_grad():
        out, _ = ag.act(img, torch.zeros(8, 2, device=DEV),
                        torch.ones(8, dtype=torch.bool, device=DEV), ag.init_ring(8, DEV))
    ent = float(entropy(out["logits"]).min())
    print(f"  fresh agent: min joint entropy {ent:.4f} (uniform {2 * math.log(nb):.4f})")
    assert ent > 2 * math.log(nb) - 1e-3


def test_gae():
    print("\n== 3. GAE ==")
    gamma, lam = 0.9, 0.8
    r = [0.1, -0.2, 0.5, 0.3, 0.4, -0.1, 0.2]
    v = [1.0, 0.5, -0.3, 0.2, 0.7, 0.1, 0.6]
    term = [0, 0, 1, 0, 0, 0, 0]
    trunc = [0, 0, 0, 0, 1, 0, 0]
    vfin = [0, 0, 0, 0, 2.5, 0, 0]
    vlast = -0.4
    T = len(r)
    ref = [0.0] * T
    nxt_a = 0.0
    for t in reversed(range(T)):
        if term[t]:
            nv, cont = 0.0, 0.0
        elif trunc[t]:
            nv, cont = vfin[t], 0.0
        else:
            nv, cont = (v[t + 1] if t + 1 < T else vlast), 1.0
        delta = r[t] + gamma * nv - v[t]
        nxt_a = delta + gamma * lam * cont * nxt_a
        ref[t] = nxt_a
    f = lambda x, dt=torch.float32: torch.tensor([x], dtype=dt, device=DEV)
    adv, ret = gae(f(r), f(v), f(term, torch.bool), f(trunc, torch.bool), f(vfin),
                   torch.tensor([vlast], device=DEV), gamma, lam)
    d = max(abs(float(a) - b) for a, b in zip(adv[0], ref))
    dr = max(abs(float(a) - (b + c)) for a, b, c in zip(ret[0], ref, v))
    print(f"  T = {T}, termination at 2, truncation at 4 (V_final 2.5), bootstrap -0.4: max |d adv| "
          f"{d:.1e}, |d ret| {dr:.1e}")
    print("  adv " + " ".join(f"{x:+.4f}" for x in ref))
    assert d < 1e-6 and dr < 1e-6
    # The termination and truncation cut the trace: adv[2] and adv[4] are one-step
    assert abs(ref[2] - (r[2] - v[2])) < 1e-12
    assert abs(ref[4] - (r[4] + gamma * vfin[4] - v[4])) < 1e-12


def test_labels():
    print("\n== 4. label alignment ==")
    tr = Trainer(small_cfg(seed=11), tempfile.mkdtemp(), build_eval=False)
    stop_bias(tr)
    tr.rollout()
    tr.rollout()
    b, env = tr.buf, tr.env
    B, T, C = b.B, b.T, b.C
    tilt = b.tilt.flatten(0, 1)
    params = tr.agent.scen_denorm(b.scen.flatten(0, 1))
    sp = env.setup_params[:, :3]  # The agent's three scenario parameters
    setup = (params[:, None] - sp[None]).abs().amax(-1).argmin(1)
    assert float((sp[setup] - params).abs().max()) < 1e-4
    out = env.sim.simulate(env.scenario_states(tilt, params), store_world=False)
    img = env._render(out)
    frames = b.frames[:, C:].flatten(0, 1)
    same = (img == frames).flatten(1).all(1)
    pn = out["total_path_length"] / env.path_refs[setup]
    d_path = float((pn - b.path.flatten()).abs().max())
    resets = b.first[:, C:].flatten()
    print(f"  {B * T} steps ({int(resets.sum())} at episode starts): image re-rendered from the "
          f"labels identical for {int(same.sum())}; max |path label - re-simulated| {d_path:.1e}")
    assert bool(same.all()) and d_path < 1e-6
    # Labels differ from the previous step's at resets (new episode, new labels)
    lab_prev = torch.cat((b.tilt[:, :1], b.tilt[:, :-1]), 1).flatten(0, 1)
    moved = (tilt - lab_prev).abs().sum(1) > 0
    print(f"  tilt label changes at {int((moved & resets).sum())} of {int(resets.sum())} resets")


def _params(tr: Trainer):
    return torch.cat([p.detach().flatten() for p in tr.agent.parameters()])


def test_resume():
    print("\n== 5. checkpoint save / resume ==")
    d = tempfile.mkdtemp()
    try:
        cfg = small_cfg(seed=21)
        a = Trainer(cfg, d, build_eval=False)
        stop_bias(a)
        a.rollout()
        a.update()
        a.iteration += 1
        a.save("latest.pt")
        a.rollout()
        snap = {k: getattr(a.buf, k).clone() for k in ("frames", "z", "actions", "logp", "rewards",
                                                       "values", "v_final", "v_last", "first")}
        bt = Trainer(cfg, d, build_eval=False)
        bt.load(os.path.join(d, "latest.pt"))
        bt.rollout()
        same = {k: bool(torch.equal(v, getattr(bt.buf, k))) for k, v in snap.items()}
        print("  next rollout after resume identical: " + ", ".join(f"{k} {v}" for k, v in same.items()))
        assert all(same.values())
        for k in vars(a.buf):
            v = getattr(a.buf, k)
            if torch.is_tensor(v):
                getattr(bt.buf, k).copy_(v)
        sa = a.update()
        sb = bt.update()
        dp = float((_params(a) - _params(bt)).abs().max())
        same_m = all(x.device == y.device and torch.equal(x, y) for x, y in zip(
            [s for st in a.opt.state.values() for s in st.values() if torch.is_tensor(s)],
            [s for st in bt.opt.state.values() for s in st.values() if torch.is_tensor(s)]))
        print(f"  next update on the same data: max |d param| {dp:.1e}, optimizer state identical "
              f"{same_m}, loss {float(sa['loss']):.6f} / {float(sb['loss']):.6f}")
        assert dp == 0.0 and same_m
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_sync():
    print("\n== 6. no host syncs in rollout() ==")
    for comp in (False, True):
        tr = Trainer(small_cfg(seed=31, compile=comp, amp=True), tempfile.mkdtemp(), build_eval=False)
        stop_bias(tr)
        tr.rollout()  # Warm-up (compilation)
        tr.update()
        tr.rollout()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            tr.rollout()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        print(f"  {'compiled' if comp else 'eager'}: a full rollout (24 steps, bootstrap, "
              f"final windows) under set_sync_debug_mode('error'): no sync")
    print("  syncs left: update() one per epoch (KL early stop), logging once per iteration, "
          "evaluation (evaluation.evaluate, per step)")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    test_agreement()
    test_distribution()
    test_gae()
    test_labels()
    test_resume()
    test_no_sync()
    print("\nall tests passed")
