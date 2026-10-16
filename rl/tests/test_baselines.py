"""
tests/test_baselines.py - Move-count BFS, oracle plans, evaluate() and the baselines

Run from rl/:  uv run python tests/test_baselines.py

    1. BFS move counts: hand-checked cases, symmetry, decompositions, larger window
    2. axis walk: realized tilt = start + d * unit, no clamping, starts at the limits
    3. oracle at hand-checkable starts (nominal, inside the trapped region, one move
       from it, the (-2, -2) corner)
    4. evaluate(): always_stop gives return 0 and length 1; waves with padding keep the
       episode order
    5. realized oracle return equals the planned return
    6. homing ends exactly at its corner plan's target; scripted emits valid actions
    7. four setups: oracle per setup at the nominal tilts (targets the setup's best
       bounce count), realized oracle return equals the plan for every setup (with and
       without auto-reset), planned return uses the setup's P_ref
    8. four setups: setup-aware homing ends at its setup's route target, setup-blind
       homing at one tilt for all setups
"""

import os
import sys

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

from baselines import (  # noqa: E402
    HomingPolicy,
    MoveTable,
    OraclePolicy,
    ScriptedPolicy,
    always_stop,
    axis_walk,
    ladder_units,
    plan_oracle,
    sim_tilts,
)
from env import AlignEnv, EnvConfig  # noqa: E402
from evaluation import eval_initial_tilts, eval_setups, evaluate  # noqa: E402
from geometry import SETUP_INFO, SETUPS  # noqa: E402

DEV = torch.device("cuda")
CAP = 128


def tilts(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32, device=DEV)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_bfs(env: AlignEnv) -> MoveTable:
    print("\n== 1. BFS move counts ==")
    lad = ladder_units(env)
    assert lad == [0, 1, -1, 3, -3, 10, -10, 30, -30, 100, -100]
    t = MoveTable(lad, 400)
    # 2 = 3 - 1, 29 = 30 - 1, 37 = 30 + 10 - 3, 99 = 100 - 1, 137 = 100 + 30 + 10 - 3
    # (no 3-step sum: 137 - 100 = 37 needs 3, 137 - 200 = -63 and 137 - 30 = 107 need 2+)
    want = {0: 0, 1: 1, 2: 2, 29: 2, 37: 3, 99: 2, 137: 4, 400: 4}
    got = {d: t(d) for d in want}
    assert got == want, got
    assert all(t(d) == t(-d) for d in range(401)), "symmetry"
    lad_t = torch.tensor(lad)
    assert torch.equal((t.counts * lad_t).sum(1), torch.arange(-400, 401)), "decompositions"
    big = MoveTable(lad, 2000)
    assert torch.equal(big.moves[1600:2401], t.moves), "window [-500, 500] is exact"
    print(f"  {got}; max over |d| <= 400: {t.max_moves}; same counts with a 4x larger window")
    return MoveTable(lad, 400, DEV)


def test_axis_walk(env: AlignEnv, table: MoveTable):
    print("\n== 2. axis walk ==")
    L, u = env.limit, env.cfg.step_unit
    starts = tilts(-2.0, -1.995, -1.0, -0.38, 0.0, 0.27, 1.0, 1.999, 2.0, 0.123456)
    d = torch.arange(-400, 401, device=DEV)
    worst, n_ok = 0.0, 0
    for s in starts:
        sd = s.double()
        ok = (sd + d * u >= -L - 1e-9) & (sd + d * u <= L + 1e-9)
        seq, x, clamped = axis_walk(s.view(1), d, table, env.ladder, L)
        assert not bool(clamped[ok].any()), (float(s), d[ok & clamped][:5])
        err = (x.double() - (sd + d * u))[ok].abs().max()
        worst = max(worst, float(err))
        n_steps = (seq != 0).sum(1)
        assert torch.equal(n_steps, table.moves_dev[d + 400]), "walk length = BFS moves"
        n_ok += int(ok.sum())
    assert worst < 1e-5, worst
    print(f"  {n_ok} (start, d) pairs inside the box: no clamping, max |x - (s + d u)| "
          f"{worst:.1e} deg, walk length = BFS count")


def test_oracle_hand(env: AlignEnv, table: MoveTable):
    print("\n== 3. oracle at hand-checkable starts ==")
    starts = tilts([-0.38, 0.27], [0.10, 1.04], [-0.20, 1.04], [-2.0, -2.0])
    b0 = sim_tilts(env, starts)["hit_counts"].tolist()
    assert b0 == [31, CAP, 57, 1], b0
    p = plan_oracle(env, starts, chunk=4, table=table)
    assert p.n_clamped == 0

    # Nominal: the trapped cells lie in pitch [0.02, 0.21] x yaw [0.94, 1.09], i.e.
    # displacements [40, 59] x [67, 82]. No 1-move value per axis in either range; the only
    # <= 2-move pair is (40, 70) = (30 + 10, 100 - 30), and it is not trapped, so the
    # optimum needs 3 moves (59 = 30 + 30 - 1, 80 = 100 - 30 + 10).
    two = [(dx, dy) for dx in range(40, 60) for dy in range(67, 83)
           if max(table(dx), table(dy)) <= 2]
    assert two == [(40, 70)], two
    s = starts[0].double()
    b2 = sim_tilts(env, (s + 0.01 * torch.tensor([[40, 70]], device=DEV)).float())["hit_counts"]
    assert int(b2) != CAP
    assert int(p.moves[0]) == 3 and int(p.target_bounces[0]) == CAP
    assert p.disp[0].tolist() == [59, 80], p.disp[0].tolist()

    # Inside the trapped region: stop at once
    assert int(p.moves[1]) == 0 and p.disp[1].tolist() == [0, 0] and float(p.ret[1]) == 0.0
    assert p.actions[1, 0].tolist() == [0, 0]

    # 57 island, 0.30 deg below the trapped region in pitch: one joint move (+0.30, +0.03)
    assert int(p.moves[2]) == 1 and int(p.target_bounces[2]) == CAP
    assert p.disp[2].tolist() == [30, 3] and p.actions[2, :2].tolist() == [[7, 3], [0, 0]]

    # Corner (-2, -2): 3 moves, 210 = 100 + 100 + 10 and 300 = 3 x 100
    assert int(p.moves[3]) == 3 and int(p.target_bounces[3]) == CAP
    assert p.disp[3].tolist() == [210, 300], p.disp[3].tolist()
    assert p.actions[3].tolist()[:4] == [[9, 9], [9, 9], [5, 9], [0, 0]]

    for i, name in enumerate(("nominal", "trapped", "57 island", "corner (-2, -2)")):
        want = (float(p.target_path_mm[i]) - float(p.path0_mm[i])) / env.p_nom
        want -= env.cfg.step_cost * int(p.moves[i])
        assert abs(float(p.ret[i]) - want) < 1e-9
        print(f"  {name:16s} disp {p.disp[i].tolist()}, {int(p.moves[i])} moves, target "
              f"{int(p.target_bounces[i])} bounces ({float(p.target_path_mm[i]) / 1000:.2f} m), "
              f"planned return {float(p.ret[i]):.4f}")


def test_evaluate_always_stop():
    print("\n== 4. evaluate() with always_stop ==")
    env = AlignEnv(EnvConfig(num_envs=4, history_len=2))
    init = eval_initial_tilts(10, seed=3, device=DEV)  # 3 waves, the last one padded
    assert torch.equal(init, eval_initial_tilts(10, seed=3, device=DEV))
    assert bool((init.abs() <= 2.0).all())
    ep, met = evaluate(env, always_stop, init)
    assert torch.equal(ep["init_tilt"], init)
    assert bool((ep["ep_return"] == 0).all()) and bool((ep["ep_length"] == 1).all())
    assert bool((ep["ep_moves"] == 0).all()) and bool(ep["terminated"].all())
    assert not bool(ep["truncated"].any())
    assert torch.equal(ep["final_path_mm"], ep["init_path_mm"])
    assert torch.equal(ep["final_tilt"], init)
    a = met["all"]
    assert a["n"] == 10 and a["return_mean"] == 0.0 and a["length_mean"] == 1.0
    assert a["terminated"] == 1.0 and met["blind"]["n"] + met["informative"]["n"] == 10
    print(f"  10 episodes in 3 waves of 4: return 0, length 1, terminated, final = initial; "
          f"blind {met['blind']['n']}, informative {met['informative']['n']}")


def test_oracle_realized(table: MoveTable):
    print("\n== 5. realized oracle return equals the plan ==")
    env = AlignEnv(EnvConfig(num_envs=48))
    init = eval_initial_tilts(96, seed=5, device=DEV)
    plan = plan_oracle(env, init, chunk=8, table=table)
    pol = OraclePolicy(plan)
    ep, met = evaluate(env, pol, init, ref_path_mm=plan.target_path_mm)
    diff = (ep["ep_return"].double() - plan.ret).abs().max()
    assert pol.start_mismatch == 0
    assert float(diff) < 1e-4, float(diff)
    assert torch.equal(ep["init_path_mm"], plan.path0_mm)
    assert torch.equal(ep["final_path_mm"], plan.target_path_mm)
    assert torch.equal(ep["final_tilt"], plan.target_tilt)
    assert torch.equal(ep["ep_moves"], plan.moves) and bool(ep["terminated"].all())
    assert torch.equal(ep["ep_length"], plan.moves + 1)
    print(f"  96 episodes: max |return - planned| {float(diff):.1e}; final tilt, path and "
          f"moves exactly as planned; mean return {met['all']['return_mean']:.3f}")


def test_homing_scripted():
    print("\n== 6. homing and scripted ==")
    env = AlignEnv(EnvConfig(num_envs=64))
    init = eval_initial_tilts(64, seed=7, device=DEV)
    hom = HomingPolicy(env)
    ep, _ = evaluate(env, hom, init)
    tgt = torch.tensor(hom.corner_plan["target_tilt"], device=DEV)
    assert bool((ep["final_tilt"] == tgt).all()), "homing ends at the corner plan's target"
    assert bool((ep["final_path_mm"] == hom.corner_plan["target_path_mm"]).all())
    pushes = ep["ep_moves"] - hom.corner_plan["moves"]
    assert bool((pushes >= 1).all()) and bool((pushes <= 5).all()), pushes
    print(f"  homing: corner {hom.corner_plan['corner']}, all 64 end at "
          f"{[round(v, 2) for v in tgt.tolist()]} ({hom.corner_plan['target_bounces']} bounces), "
          f"homing pushes {pushes.min().item()}-{pushes.max().item()}")

    sp = ScriptedPolicy(env)
    seen = []

    def checked(obs, info, state):
        a, state = sp(obs, info, state)
        seen.append(a[info["active"]])
        return a, state

    ep, met = evaluate(env, checked, init)
    acts = torch.cat(seen)
    assert bool((acts >= 0).all()) and bool((acts < env.n_bins).all())
    assert bool(torch.isfinite(ep["ep_return"]).all())
    print(f"  scripted: {acts.shape[0]} actions, all valid ladder indices; mean return "
          f"{met['all']['return_mean']:.3f}")


def test_multi_oracle(table: MoveTable):
    print("\n== 7. oracle with four setups ==")
    env = AlignEnv(EnvConfig(num_envs=64, setups=SETUPS, history_len=0))
    refs = env.path_refs.double()
    # From each setup's nominal tilt the oracle goes to that setup's best region
    nom = tilts(*[(g.m2_pitch, g.m2_yaw) for g in SETUPS])
    s4 = torch.arange(4, device=DEV)
    p = plan_oracle(env, nom, chunk=4, table=table, setups=s4)
    assert p.n_clamped == 0 and torch.equal(p.setup, s4)
    for i, g in enumerate(SETUPS):
        assert int(p.target_bounces[i]) == SETUP_INFO[g.name].target_bounces, g.name
        want = (float(p.target_path_mm[i]) - float(p.path0_mm[i])) / float(refs[i])
        want -= env.cfg.step_cost * int(p.moves[i])
        assert abs(float(p.ret[i]) - want) < 1e-9
        assert float(p.target_path_mm[i]) >= 0.99 * float(refs[i])
        print(f"  {g.name:8s} nominal -> disp {p.disp[i].tolist()}, {int(p.moves[i])} moves, "
              f"{int(p.target_bounces[i])} bounces, P / P_max {float(p.target_path_mm[i]) / float(refs[i]):.4f}, "
              f"planned return {float(p.ret[i]):.4f}")

    # Realized == planned for every setup, with and without auto-reset
    init = eval_initial_tilts(96, seed=9, device=DEV)
    setups = eval_setups(96, 4, seed=9, device=DEV)
    assert torch.equal(torch.bincount(setups, minlength=4).cpu(), torch.full((4,), 24))
    plan = plan_oracle(env, init, chunk=8, table=table, setups=setups)
    for auto in (True, False):
        e = AlignEnv(EnvConfig(num_envs=40, setups=SETUPS, auto_reset=auto, history_len=2))
        pol = OraclePolicy(plan)
        ep, met = evaluate(e, pol, init, ref_path_mm=plan.target_path_mm, ref_return=plan.ret,
                           init_setups=setups)
        assert torch.equal(ep["setup"], setups) and pol.start_mismatch == 0
        diff = (ep["ep_return"].double() - plan.ret).abs()
        for i in range(4):
            assert float(diff[setups == i].max()) < 1e-4
        assert torch.equal(ep["init_path_mm"], plan.path0_mm)
        assert torch.equal(ep["final_path_mm"], plan.target_path_mm)
        assert torch.equal(ep["final_tilt"], plan.target_tilt)
        assert torch.equal(ep["ep_moves"], plan.moves) and bool(ep["terminated"].all())
        assert torch.equal(ep["path_ref"], e.path_refs[setups])
        assert all(met[f"setup:{g.name}"]["n"] == 24 for g in SETUPS)
        print(f"  96 episodes (24 per setup), auto_reset {auto}: max |return - planned| "
              f"{float(diff.max()):.1e}, final tilt / path / moves as planned; mean return "
              + ", ".join(f"{g.name} {met['setup:' + g.name]['return_mean']:.3f}" for g in SETUPS))


def test_multi_homing():
    print("\n== 8. homing with four setups ==")
    env = AlignEnv(EnvConfig(num_envs=64, setups=SETUPS, auto_reset=False))
    init = eval_initial_tilts(64, seed=11, device=DEV)
    setups = eval_setups(64, 4, seed=11, device=DEV)
    aware = HomingPolicy(env, variant="aware")
    ep, _ = evaluate(env, aware, init, init_setups=setups)
    for i, cp in enumerate(aware.corner_plans):
        m = setups == i
        tgt = torch.tensor(cp["target_tilt"], device=DEV)
        assert bool((ep["final_tilt"][m] == tgt).all()), SETUPS[i].name
        assert bool((ep["final_path_mm"][m] == cp["target_path_mm"]).all())
        assert int(cp["target_bounces"]) == SETUP_INFO[SETUPS[i].name].target_bounces
    blind = HomingPolicy(env, variant="blind")
    ep, _ = evaluate(env, blind, init, init_setups=setups)
    tgt = torch.tensor(blind.corner_plan["target_tilt"], device=DEV)
    assert bool((ep["final_tilt"] == tgt).all())
    for i in range(4):
        m = setups == i
        assert bool((ep["final_path_mm"][m] == blind.corner_plan["target_path_mm"][i]).all())
    print("  aware: " + "; ".join(
        f"{g.name} -> {[round(v, 2) for v in cp['target_tilt']]} ({cp['target_bounces']})"
        for g, cp in zip(SETUPS, aware.corner_plans)))
    print(f"  blind: all -> {[round(v, 2) for v in blind.corner_plan['target_tilt']]}, bounces "
          f"{blind.corner_plan['target_bounces']} per setup")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    env = AlignEnv(EnvConfig(num_envs=4, history_len=0))
    table = test_bfs(env)
    test_axis_walk(env, table)
    test_oracle_hand(env, table)
    test_evaluate_always_stop()
    test_oracle_realized(table)
    test_homing_scripted()
    test_multi_oracle(table)
    test_multi_homing()
    print("\nall baseline tests passed")
