"""
baselines.py - Phase 4 baselines on the fixed evaluation set (PLAN phase 4)

Run from rl/:  uv run python baselines.py [--n 4096] [--seed 0] [--num-envs 1024] [--variants]
               uv run python baselines.py --multisetup   (4 setups of geometry.SETUPS)

Policies (evaluation.Policy interface):
    random          uniform over the 121 joint actions; (0, 0) has probability 1/121
    random_no_stop  uniform over the 120 moves, never stops (always truncated)
    always_stop     (0, 0) at step 0
    oracle          privileged: knows the true initial tilt; walks the minimal number of
                    ladder steps to the tilt that maximizes the return, then stops
    scripted        image only: spiral search while one spot is visible, then hill-climb on
                    the image brightness with decreasing step sizes, then stop
    homing          image free (extra, not in the PLAN): pushes both axes into a corner
                    until the clamp shows in obs["action"], then replays the best plan from
                    that corner (prior knowledge of the fixed stage-1 landscape)

Multi-setup run (--multisetup): env with geometry.SETUPS, P_ref = P_max per setup, no
auto-reset; eval set = (initial tilt, setup) pairs, balanced over the setups
(evaluation.eval_setups); metrics also per setup. The oracle knows the setup and the tilt
(per-episode P_ref in its value). Homing in two variants: homing_aware (privileged: the
route of the episode's setup) and homing_blind (one fixed route maximizing the mean value
over the setups; the best an image-free agent can do with homing).

Oracle. From the start tilt s the reachable tilts are s + step_unit * (i, j) inside the
box. Tilts reachable only through clamping at the limits (e.g. pushing into the limit to
land exactly on it) are ignored, so the oracle is the exact optimum over the
clamp-free strategies; the homing baseline shows what clamping adds. Per axis the
minimal number of moves for a displacement of d units is an exact BFS over the integers
with the signed ladder steps; both axes move in the same step, so a target costs
max(moves_x, moves_y). Each episode scans its whole shifted grid (401 x 401 states, one
Triton call per chunk of episodes) with the env's float32 tilt arithmetic, and picks
argmax P(target) / P_nom - step_cost * moves (stop at once: P_0 / P_nom, moves 0), ties
to fewer moves. The planned return is (P_target - P_0) / P_nom - step_cost * moves.
With several setups P_nom is the episode's P_ref.

Writes reports/baselines.md and reports/baselines.json (multi-setup:
reports/baselines_multisetup.md / .json).
"""

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from env import AlignEnv, EnvConfig
from evaluation import (
    LEVELS,
    SUCCESS_FRAC,
    bounce_bins,
    eval_initial_tilts,
    eval_setups,
    evaluate,
)
from geometry import SETUPS

RL_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Move counts (exact BFS) ─────────────────────────────────────────────────


def ladder_units(env: AlignEnv) -> List[int]:
    """The step ladder as integers in step_unit."""
    out = [int(round(s)) for s in env.cfg.step_ladder]
    assert all(abs(a - s) < 1e-9 for a, s in zip(out, env.cfg.step_ladder)), "integer ladder"
    return out


class MoveTable:
    """Minimal number of ladder steps per axis for every displacement in [-span, span].

    BFS from 0 over the integers with the signed nonzero ladder steps. It is exact on
    the window [-(span + M), span + M] (M = largest step): any step multiset summing to d
    can be ordered (positive while the partial sum is <= d, negative while it is > d) so
    that every partial sum stays within [min(0, d) - M, max(0, d) + M].

    moves (2 span + 1,) int64; counts (2 span + 1, n_bins) int64: one minimal step
    multiset per displacement (BFS parents, larger steps tried first); index d + span.
    """

    def __init__(self, ladder: Sequence[int], span: int, device: Optional[torch.device] = None):
        self.ladder = list(ladder)
        self.span = span
        steps = sorted(
            [(i, s) for i, s in enumerate(self.ladder) if s != 0], key=lambda p: (-abs(p[1]), -p[1])
        )
        M = max(abs(s) for _, s in steps)
        R = span + M
        dist = [-1] * (2 * R + 1)
        via = [0] * (2 * R + 1)
        dist[R] = 0
        queue = deque([0])
        while queue:
            x = queue.popleft()
            for i, s in steps:
                y = x + s
                if -R <= y <= R and dist[y + R] < 0:
                    dist[y + R] = dist[x + R] + 1
                    via[y + R] = i
                    queue.append(y)
        counts = torch.zeros(2 * span + 1, len(self.ladder), dtype=torch.long)
        for d in range(-span, span + 1):
            x = d
            while x != 0:
                i = via[x + R]
                counts[d + span, i] += 1
                x -= self.ladder[i]
        self.moves = torch.tensor(dist[R - span : R + span + 1], dtype=torch.long)
        assert bool((self.moves >= 0).all()) and torch.equal(counts.sum(1), self.moves)
        self.counts = counts
        self.max_moves = int(self.moves.max())
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.moves_dev = self.moves.to(dev)
        self.counts_dev = counts.to(dev)
        lad = torch.tensor(self.ladder)
        order = torch.argsort(-lad.abs(), stable=True)
        self.pos_order = order[lad[order] > 0].to(dev)  # Ladder indices, largest step first
        self.neg_order = order[lad[order] < 0].to(dev)

    def __call__(self, d: int) -> int:
        return int(self.moves[d + self.span])


def axis_walk(
    start: torch.Tensor, d: torch.Tensor, table: MoveTable, ladder: torch.Tensor, limit: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the minimal step multiset of displacement d (units) from start on one axis.

    Same float32 arithmetic as AlignEnv.step: x = clamp(x + ladder[a], -limit, limit).
    Order: the largest remaining positive step if it stays inside the limit, else the
    largest remaining negative step. With the target inside the box this never clamps
    (a positive step that does not fit means x > limit - M, so any negative step lands
    above -limit; once one sign is used up the partial sums are monotone toward the
    target).

    start broadcastable to d.shape (float32), d int64. Returns seq (*d.shape, T) ladder
    indices (zero-padded), final tilt (*d.shape,), clamped (*d.shape,) bool (any step
    clamped by more than 1e-6 deg).
    """
    T = table.max_moves
    cnt = table.counts_dev[d + table.span].clone()
    x = start.to(torch.float32).expand(d.shape).clone()
    seq = torch.zeros(*d.shape, T, dtype=torch.long, device=d.device)
    clamped = torch.zeros(d.shape, dtype=torch.bool, device=d.device)
    pos, neg = table.pos_order, table.neg_order
    for t in range(T):
        ap = cnt[..., pos] > 0
        an = cnt[..., neg] > 0
        ip = pos[ap.int().argmax(-1)]
        ineg = neg[an.int().argmax(-1)]
        has_p, has_n = ap.any(-1), an.any(-1)
        fits = (x + ladder[ip]) <= limit
        take_p = has_p & (fits | ~has_n)
        idx = torch.where(take_p, ip, torch.where(has_n, ineg, 0))
        raw = x + ladder[idx]
        clamped |= (raw > limit + 1e-6) | (raw < -limit - 1e-6)
        x = raw.clamp(-limit, limit)
        cnt.scatter_add_(-1, idx.unsqueeze(-1), -(idx != 0).long().unsqueeze(-1))
        seq[..., t] = idx
    assert bool((cnt == 0).all())
    return seq, x, clamped


# ── Oracle ──────────────────────────────────────────────────────────────────


@dataclass
class OraclePlan:
    """Per-episode oracle plans (n leading, on the env device)."""

    start: torch.Tensor  # (n, 2) initial tilt (deg)
    disp: torch.Tensor  # (n, 2) target displacement (units)
    moves: torch.Tensor  # (n,) moves to the target (the episode has moves + 1 steps)
    actions: torch.Tensor  # (n, T + 1, 2) ladder indices, zero after the last move
    ret: torch.Tensor  # (n,) float64 planned return
    path0_mm: torch.Tensor  # (n,) path at the start
    target_path_mm: torch.Tensor  # (n,)
    target_bounces: torch.Tensor  # (n,)
    target_tilt: torch.Tensor  # (n, 2)
    max_path_mm: torch.Tensor  # (n,) longest path on the reachable grid (cost ignored)
    max_moves_axis: int  # T
    n_clamped: int  # grid targets whose walk clamped (must be 0)
    plan_s: float  # wall time
    setup: Optional[torch.Tensor] = None  # (n,) setup index per episode


def sim_tilts(env: AlignEnv, tilts: torch.Tensor, setups: Optional[torch.Tensor] = None,
              params: Optional[torch.Tensor] = None) -> dict:
    """Sim of (n, 2) controlled tilts of setups (n,) (default 0), the env's states, or
    of explicit scenario params (n, 3 | 5) (env.scenario_states); no per-hit outputs."""
    st = env.states(tilts, setups) if params is None else env.scenario_states(tilts, params)
    return env.sim.simulate(st, store_world=False, store_hits=False)


@dataclass
class ReachGrid:
    """Clamp-free reachable grid from E start tilts: start + step_unit * (i, j) inside
    the box, W = span + 1 displacements per axis (span = box width in units), walked with
    the minimal step multiset."""

    ix: torch.Tensor  # (E,) displacement (units) of grid index 0, pitch
    iy: torch.Tensor  # (E,) the same, yaw
    dx: torch.Tensor  # (E, W) displacements, pitch
    dy: torch.Tensor  # (E, W)
    seqx: torch.Tensor  # (E, W, T) ladder indices of the pitch walk
    seqy: torch.Tensor  # (E, W, T)
    xx: torch.Tensor  # (E, W) realized pitch (float32, env arithmetic)
    xy: torch.Tensor  # (E, W)
    m: torch.Tensor  # (E, W, W) moves = max(moves_x, moves_y)
    ok: torch.Tensor  # (E, W, W) inside the box
    n_clamped: int  # Grid targets inside the box whose walk clamped (must be 0)

    @property
    def tilts(self) -> torch.Tensor:
        """(E * W * W, 2) realized grid tilts, pitch-major."""
        E, W = self.xx.shape
        return torch.stack(
            (self.xx[:, :, None].expand(E, W, W), self.xy[:, None, :].expand(E, W, W)), dim=-1
        ).reshape(-1, 2)


def reach_grid(env: AlignEnv, starts: torch.Tensor, table: MoveTable) -> ReachGrid:
    """ReachGrid of (E, 2) float32 start tilts."""
    dev = env.device
    u, L = env.cfg.step_unit, env.limit
    span = table.span  # Largest displacement (units) = box width; W = span + 1
    k = torch.arange(span + 1, device=dev)
    ax, n_clamped = [], 0
    for a in range(2):
        sa = starts[:, a].double()
        i_min = torch.ceil((-L - sa) / u - 1e-6).long()
        i_max = torch.floor((L - sa) / u + 1e-6).long()
        d = i_min[:, None] + k  # (E, W)
        ok = d <= i_max[:, None]
        seq, x, clamped = axis_walk(starts[:, a, None], d, table, env.ladder, L)
        n_clamped += int((clamped & ok).sum())
        ax.append((i_min, d, ok, seq, x, table.moves_dev[d + span]))
    (ix, dx, okx, seqx, xx, mx), (iy, dy, oky, seqy, xy, my) = ax
    return ReachGrid(
        ix=ix, iy=iy, dx=dx, dy=dy, seqx=seqx, seqy=seqy, xx=xx, xy=xy,
        m=torch.maximum(mx[:, :, None], my[:, None, :]),
        ok=okx[:, :, None] & oky[:, None, :], n_clamped=n_clamped,
    )


def _best_cell(val: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(kx, ky) per row of val (E, W, W): first of the fewest-move maxima."""
    W = val.shape[-1]
    vmax = val.flatten(1).amax(1)
    tie = val == vmax[:, None, None]
    key = torch.where(tie, m, torch.iinfo(torch.long).max).flatten(1)
    j = key.argmin(1)
    return j // W, j % W


@torch.no_grad()
def plan_oracle(
    env: AlignEnv,
    init_tilts: torch.Tensor,
    chunk: int = 8,
    table: Optional[MoveTable] = None,
    setups: Optional[torch.Tensor] = None,
    scenarios: Optional[torch.Tensor] = None,
    path_ref: Optional[torch.Tensor] = None,
) -> OraclePlan:
    """Exact best clamp-free plan per (initial tilt, setup), see the module docstring.
    setups (n,): default 0; the value uses each episode's P_ref. Any scenario instead:
    scenarios (n, 3 | 5) explicit (separation, laser pitch, laser yaw[, dx, dy]) per episode (every
    other DOF as the env's setup 0) with path_ref (n,) their P_ref; setups is then only
    recorded in the plan."""
    dev = env.device
    c = env.cfg
    u, L = c.step_unit, env.limit
    span = int(round(2 * L / u))
    W = span + 1
    table = table or MoveTable(ladder_units(env), span, dev)
    T = table.max_moves
    starts = init_tilts.to(device=dev, dtype=torch.float32).clamp(-L, L)
    n = starts.shape[0]
    single = env.n_setups == 1
    setups = (torch.zeros(n, dtype=torch.long, device=dev) if setups is None
              else setups.to(device=dev, dtype=torch.long))
    if scenarios is not None:
        assert path_ref is not None and path_ref.shape[0] == n and scenarios.shape[0] == n
        assert scenarios.shape[1] in (3, 5), "scenario parameters (n, 3) or (n, 5)"
        scen = scenarios.to(device=dev, dtype=torch.float32)
        pref = path_ref.to(device=dev).double()
        single = False
    fields: Dict[str, List[torch.Tensor]] = {
        f: [] for f in ("disp", "moves", "actions", "ret", "path0", "tp", "tb", "tt", "maxp")
    }
    n_clamped = 0
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for e0 in range(0, n, chunk):
        s = starts[e0 : e0 + chunk]
        E = s.shape[0]
        su = setups[e0 : e0 + E]
        G = reach_grid(env, s, table)
        n_clamped += G.n_clamped
        if scenarios is not None:
            ref_e = pref[e0 : e0 + E]
            out = sim_tilts(env, G.tilts, params=scen[e0 : e0 + E].repeat_interleave(W * W, 0))
        else:
            ref_e = env.p_nom if single else env.path_refs.double()[su]
            out = sim_tilts(env, G.tilts, None if single else su.repeat_interleave(W * W))
        P = out["total_path_length"].view(E, W, W)
        nb = out["hit_counts"].view(E, W, W)
        del out
        m = G.m
        ref = ref_e if single else ref_e[:, None, None]
        val = torch.where(G.ok, P.double() / ref - c.step_cost * m.double(), -math.inf)
        kx, ky = _best_cell(val, m)
        e = torch.arange(E, device=dev)
        p0 = P[e, -G.ix, -G.iy]  # d = (0, 0)
        mt = m[e, kx, ky]
        fields["disp"].append(torch.stack((G.dx[e, kx], G.dy[e, ky]), 1))
        fields["moves"].append(mt)
        acts = torch.stack((G.seqx[e, kx], G.seqy[e, ky]), -1)  # (E, T, 2)
        fields["actions"].append(F.pad(acts, (0, 0, 0, 1)))
        fields["ret"].append(
            (P[e, kx, ky].double() - p0.double()) / ref_e - c.step_cost * mt.double()
        )
        fields["path0"].append(p0)
        fields["tp"].append(P[e, kx, ky])
        fields["tb"].append(nb[e, kx, ky])
        fields["tt"].append(torch.stack((G.xx[e, kx], G.xy[e, ky]), 1))
        fields["maxp"].append(torch.where(G.ok, P, -math.inf).flatten(1).amax(1))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    plan_s = time.perf_counter() - t0

    cat = {f: torch.cat(v) for f, v in fields.items()}
    return OraclePlan(
        start=starts,
        disp=cat["disp"],
        moves=cat["moves"],
        actions=cat["actions"],
        ret=cat["ret"],
        path0_mm=cat["path0"],
        target_path_mm=cat["tp"],
        target_bounces=cat["tb"],
        target_tilt=cat["tt"],
        max_path_mm=cat["maxp"],
        max_moves_axis=T,
        n_clamped=n_clamped,
        plan_s=plan_s,
        setup=setups,
    )


@torch.no_grad()
def plan_route(
    env: AlignEnv,
    start: torch.Tensor,
    setups: Sequence[int],
    weights: Sequence[float],
    table: Optional[MoveTable] = None,
) -> Dict[str, Any]:
    """Best fixed clamp-free route from one start tilt (2,) for a weighted set of setups:
    argmax over the reachable grid of sum_s w_s P_s / P_ref,s - step_cost * moves (ties to
    fewer moves). With one setup this is the oracle plan of that start."""
    dev = env.device
    c = env.cfg
    span = int(round(2 * env.limit / c.step_unit))
    W = span + 1
    table = table or MoveTable(ladder_units(env), span, dev)
    s = start.to(device=dev, dtype=torch.float32).view(1, 2)
    G = reach_grid(env, s, table)
    single = env.n_setups == 1
    val = torch.zeros(1, W, W, dtype=torch.float64, device=dev)
    Ps, Bs = [], []
    for si, w in zip(setups, weights):
        sv = None if single else torch.full((W * W,), int(si), dtype=torch.long, device=dev)
        out = sim_tilts(env, G.tilts, sv)
        P = out["total_path_length"].view(1, W, W)
        Ps.append(P)
        Bs.append(out["hit_counts"].view(1, W, W))
        del out
        ref = env.p_nom if single else float(env.path_refs[int(si)])
        val = val + float(w) * (P.double() / ref)
    val = torch.where(G.ok, val - c.step_cost * G.m.double(), -math.inf)
    kx, ky = _best_cell(val, G.m)
    kx, ky = int(kx[0]), int(ky[0])
    acts = torch.stack((G.seqx[0, kx], G.seqy[0, ky]), -1)
    return {
        "start": [float(v) for v in s[0].tolist()],
        "target_tilt": [float(G.xx[0, kx]), float(G.xy[0, ky])],
        "moves": int(G.m[0, kx, ky]),
        "value": float(val[0, kx, ky]),
        "actions": F.pad(acts, (0, 0, 0, 1)),  # (T + 1, 2)
        "target_path_mm": [float(P[0, kx, ky]) for P in Ps],
        "target_bounces": [int(B_[0, kx, ky]) for B_ in Bs],
        "n_clamped": G.n_clamped,
    }


# ── Policies ────────────────────────────────────────────────────────────────


class RandomPolicy:
    """Uniform over the joint actions; allow_stop False excludes (0, 0)."""

    def __init__(self, env: AlignEnv, allow_stop: bool = True, seed: int = 0):
        self.n = env.n_bins
        self.lo = 0 if allow_stop else 1
        self.gen = torch.Generator(device=env.device)
        self.gen.manual_seed(seed)

    def __call__(self, obs, info, state):
        B = obs["image"].shape[0]
        j = torch.randint(
            self.lo, self.n * self.n, (B,), generator=self.gen, device=obs["image"].device
        )
        return torch.stack((j // self.n, j % self.n), 1), state


def always_stop(obs, info, state):
    """(0, 0) at step 0."""
    B = obs["image"].shape[0]
    return torch.zeros(B, 2, dtype=torch.long, device=obs["image"].device), state


class OraclePolicy:
    """Replays the oracle plan of each env's episode (privileged: episode -> start tilt)."""

    def __init__(self, plan: OraclePlan):
        self.plan = plan
        self.start_mismatch = 0  # Envs whose start tilt differs from the plan's

    def __call__(self, obs, info, state):
        ep = info["episode"].clamp(min=0)
        t = info["step"].clamp(max=self.plan.actions.shape[1] - 1)
        a = self.plan.actions[ep, t]
        a = torch.where(info["active"][:, None], a, 0)
        chk = info["reset"] & info["active"]
        self.start_mismatch += int((chk & (info["tilt"] != self.plan.start[ep]).any(1)).sum())
        return a, state


class ScriptedPolicy:
    """Image-only heuristic, vectorized over envs; uses obs["image"] and its own history.

    Scores of the frame: M2 spot count = 3x3 local maxima >= peak_thr, brightness =
    sum of pixels / 255 (monotone in the bounce count: ~11 for one spot, ~144 at 31,
    ~218 at 57, ~250 at 89, ~273 trapped; the spot count saturates near 30 beyond 57
    because of overlap, so the climb uses brightness).
        search  while the spot count is <= 1: expanding square spiral (legs 1, 1, 2, 2,
                3, 3, ... of search_step units, directions +x, +y, -x, -y)
        climb   from the first frame with >= 2 spots: probe one of n_dirs directions at
                step size sizes[i]; brightness up by > eps: keep it and probe the same
                direction again; within +-eps: keep the position, next direction; down by
                > eps: move back (one step), next direction. n_dirs probes in a row
                without a gain move to the next smaller size; after the last size, stop.
    The episode may truncate first (64 steps). Defaults (1 deg search step, sizes 1 and
    0.3 deg, 8 directions, eps 2) were the best of a few variants on the eval set, incl.
    the PLAN's 0.3 deg search with sizes 0.3 / 0.1 / 0.03 deg (main --variants).
    """

    SEARCH, CLIMB, STOP = 0, 1, 2

    def __init__(
        self,
        env: AlignEnv,
        search_step: int = 100,
        sizes: Sequence[int] = (100, 30),
        n_dirs: int = 8,
        eps: float = 2.0,
        peak_thr: int = 20,
    ):
        dev = env.device
        lad = ladder_units(env)
        M = max(abs(s) for s in lad)
        self.M = M
        self.lut = torch.full((2 * M + 1,), -1, dtype=torch.long, device=dev)
        for i, s in enumerate(lad):
            self.lut[s + M] = i
        for s in (search_step, *sizes):
            assert s in lad and -s in lad, f"step {s} not on the ladder"
        self.sizes = torch.tensor(sizes, device=dev)
        dirs8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        dirs = dirs8 if n_dirs == 8 else dirs8[::2]
        self.dirs = torch.tensor(dirs, device=dev)
        self.n_dirs = len(dirs)
        legs, leg = [], 1
        while len(legs) < env.cfg.max_steps:
            for dvec in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                legs += [dvec] * leg
                if dvec[1] != 0:
                    leg += 1
        self.spiral = torch.tensor(legs[: env.cfg.max_steps], device=dev) * search_step
        self.eps = eps
        self.peak_thr = peak_thr
        self.params = dict(
            search_step=search_step, sizes=list(sizes), n_dirs=self.n_dirs, eps=eps,
            peak_thr=peak_thr,
        )

    def scores(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(spot count, brightness) per frame of (B, 1, H, W) uint8 images."""
        img = image.float()
        peak = (img == F.max_pool2d(img, 3, 1, 1)) & (img >= self.peak_thr)
        return peak.flatten(1).sum(1), img.flatten(1).sum(1) / 255.0

    def __call__(self, obs, info, state):
        n, f = self.scores(obs["image"])
        B = f.shape[0]
        dev = f.device
        if state is None:
            z = torch.zeros(B, dtype=torch.long, device=dev)
            state = dict(mode=z, k=z, size=z, dir=z, tries=z, best=f,
                         probing=z.bool(), pending=torch.zeros(B, 2, dtype=torch.long, device=dev))
        S = dict(state)
        r = info["reset"]
        start_mode = torch.where(n >= 2, self.CLIMB, self.SEARCH)
        S["mode"] = torch.where(r, start_mode, S["mode"])
        for key in ("k", "size", "dir", "tries"):
            S[key] = torch.where(r, 0, S[key])
        S["best"] = torch.where(r, f, S["best"])
        S["probing"] = S["probing"] & ~r

        # Search -> climb at the first frame with >= 2 spots
        to_climb = (S["mode"] == self.SEARCH) & (n >= 2)
        S["mode"] = torch.where(to_climb, self.CLIMB, S["mode"])
        S["best"] = torch.where(to_climb, f, S["best"])

        # Outcome of the last probe
        climb = S["mode"] == self.CLIMB
        ev = climb & S["probing"]
        better = ev & (f > S["best"] + self.eps)
        worse = ev & (f < S["best"] - self.eps)
        same = ev & ~better & ~worse
        S["best"] = torch.where(better | same, f, S["best"])
        S["tries"] = torch.where(better, 0, torch.where(worse | same, S["tries"] + 1, S["tries"]))
        S["dir"] = torch.where(worse | same, (S["dir"] + 1) % self.n_dirs, S["dir"])
        S["probing"] = S["probing"] & ~ev

        # Step-size schedule and stop
        shrink = climb & ~worse & (S["tries"] >= self.n_dirs)
        S["size"] = torch.where(shrink, S["size"] + 1, S["size"])
        S["tries"] = torch.where(shrink, 0, S["tries"])
        S["mode"] = torch.where(climb & (S["size"] >= len(self.sizes)), self.STOP, S["mode"])

        # Actions (units per axis)
        mode = S["mode"]
        size = self.sizes[S["size"].clamp(max=len(self.sizes) - 1)]
        probe_move = self.dirs[S["dir"]] * size[:, None]
        search_move = self.spiral[S["k"].clamp(max=self.spiral.shape[0] - 1)]
        probe = (mode == self.CLIMB) & ~worse
        move = torch.zeros(B, 2, dtype=torch.long, device=dev)
        move = torch.where((mode == self.SEARCH)[:, None], search_move, move)
        move = torch.where(worse[:, None], -S["pending"], move)
        move = torch.where(probe[:, None], probe_move, move)
        S["pending"] = torch.where(probe[:, None], probe_move, S["pending"])
        S["probing"] = S["probing"] | probe
        S["k"] = torch.where(mode == self.SEARCH, S["k"] + 1, S["k"])
        return self.lut[move + self.M], S


class HomingPolicy:
    """Image-free: home to a corner through the mount limit, then replay a fixed route.

    Each axis pushes the largest ladder step toward the corner until obs["action"] shows
    a clamped push (|applied| < 0.999 of the step), then idles; once both axes are home
    (tilt exactly at the corner) it replays the route from that corner and stops. Corner
    and route: the best plan_route over the four corners. The landscape is known prior
    knowledge, not per-episode privileged information, except as noted:
        variant "aware"  one route per setup, selected by info["setup"] (privileged: knows
                         the episode's setup); with one setup the stage-1 homing policy
        variant "blind"  one route for all setups, maximizing the setup-weighted mean
                         value (env.setup_probs): what an agent that ignores the images can
                         do at best with homing
    """

    def __init__(self, env: AlignEnv, chunk: int = 4, variant: str = "aware"):
        assert variant in ("aware", "blind"), variant
        del chunk  # Kept for the old signature
        dev = env.device
        L = env.limit
        self.variant = variant
        S = env.n_setups
        corners = torch.tensor(
            [[-L, -L], [-L, L], [L, -L], [L, L]], dtype=torch.float32, device=dev
        )
        span = int(round(2 * L / env.cfg.step_unit))
        table = MoveTable(ladder_units(env), span, dev)
        if variant == "aware":
            groups = [([s], [1.0]) for s in range(S)]
        else:
            groups = [(list(range(S)), env.setup_probs.tolist())]
        lad = ladder_units(env)
        M = max(abs(s) for s in lad)
        self.corner_plans: List[Dict[str, Any]] = []
        seqs, pushes, cs = [], [], []
        for setups, weights in groups:
            plans = [plan_route(env, cc, setups, weights, table) for cc in corners]
            b = max(range(4), key=lambda i: (plans[i]["value"], -i))
            p = plans[b]
            self.corner_plans.append({
                "setups": setups,
                "corner": corners[b].tolist(),
                "target_tilt": p["target_tilt"],
                "target_path_mm": p["target_path_mm"][0] if len(setups) == 1
                else p["target_path_mm"],
                "target_bounces": p["target_bounces"][0] if len(setups) == 1
                else p["target_bounces"],
                "moves": p["moves"],
                "value": p["value"],
                "all_corners": {
                    str(tuple(round(v, 2) for v in corners[i].tolist())): {
                        "moves": plans[i]["moves"], "target_bounces": plans[i]["target_bounces"],
                        "value": plans[i]["value"],
                    }
                    for i in range(4)
                },
            })
            seqs.append(p["actions"])
            cs.append(corners[b])
            sgn = torch.sign(corners[b]).long()
            pushes.append(torch.tensor([lad.index(int(M * v)) for v in sgn.tolist()], device=dev))
        self.corner_plan = self.corner_plans[0]
        self.corners = torch.stack(cs)  # (G, 2)
        self.corner = self.corners[0]
        self.seqs = torch.stack(seqs)  # (G, T + 1, 2), zero after the last move
        self.seq = self.seqs[0]
        self.pushes = torch.stack(pushes)  # (G, 2)
        self.push = self.pushes[0]

    def __call__(self, obs, info, state):
        B = obs["image"].shape[0]
        dev = obs["image"].device
        if state is None:
            z = torch.zeros(B, 2, dtype=torch.bool, device=dev)
            state = dict(home=z, pushed=z, k=torch.zeros(B, dtype=torch.long, device=dev))
        grp = info["setup"] if (self.variant == "aware" and self.seqs.shape[0] > 1) else (
            torch.zeros(B, dtype=torch.long, device=dev))
        r = info["reset"]
        home = state["home"] & ~r[:, None]
        pushed = state["pushed"] & ~r[:, None]
        k = torch.where(r, 0, state["k"])
        home = home | (pushed & (obs["action"].abs() < 0.999))
        at = home.all(1)
        a_home = torch.where(home, 0, self.pushes[grp])
        a_plan = self.seqs[grp, k.clamp(max=self.seqs.shape[1] - 1)]
        a = torch.where(at[:, None], a_plan, a_home)
        state = dict(home=home, pushed=~home & ~at[:, None], k=torch.where(at, k + 1, k))
        return a, state


# ── Report ──────────────────────────────────────────────────────────────────


def _f(x: Optional[float], fmt: str = ".3f") -> str:
    return "-" if x is None else format(x, fmt)


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def region_of(bounces: torch.Tensor, moves: torch.Tensor, cap: int) -> Dict[str, int]:
    """Oracle target regions: stop at once / trapped (cap) / 89 / 57 / 31 / other."""
    stay = moves == 0
    cats = {
        "stop at once": stay,
        "trapped (cap)": ~stay & (bounces >= cap),
        "89": ~stay & (bounces == 89),
        "57": ~stay & (bounces == 57),
        "31": ~stay & (bounces == 31),
    }
    other = ~stay
    for key in list(cats)[1:]:
        other = other & ~cats[key]
    cats["other"] = other
    return {key: int(v.sum()) for key, v in cats.items()}


def write_report(res: Dict[str, Any], path_md: str):
    """Markdown report from the results dict (see main)."""
    cfg, orc = res["config"], res["oracle"]
    pol = res["policies"]
    cap = cfg["max_bounces"]
    L = []
    w = L.append
    w("# Phase 4 baselines\n")
    w(f"Generated by `uv run python baselines.py` ({res['date']}). Evaluation set: "
      f"{cfg['n']} initial M2 tilts uniform in the +-{cfg['limit']:g} deg box "
      f"(`eval_initial_tilts(n={cfg['n']}, seed={cfg['seed']})`), one episode each, default "
      f"`EnvConfig` ({cfg['img_res']} px, max_steps {cfg['max_steps']}, step_cost "
      f"{cfg['step_cost']}, cap {cap}). Blind start: 1-2 bounces at the initial tilt "
      f"({_pct(cfg['blind_share'])} % of the set); informative: >= 3. "
      "Raw numbers: `baselines.json`.\n")
    w("Policies: `random` (uniform over the 121 joint actions), `random_no_stop` (uniform over "
      "the 120 moves), `always_stop`, `oracle` (privileged, exact clamp-free optimum, see "
      "below), `scripted` (image only: spiral search, then hill-climb on brightness), "
      "`homing` (extra, image-free: home into a corner via the mount limit, then a fixed "
      "plan from the stage-1 map).\n")

    # Main table
    w("## Main metrics\n")
    w("Return = (P_final - P_0) / P_nom - 0.01 * moves. Path in m. End >= k: fraction of "
      "episodes whose final frame has >= k bounces. Term / trunc: ended by (0, 0) / by the "
      "64-step limit. Length in steps (incl. the stop step). Regret: oracle return minus "
      "return, per episode (the fair comparison across start regions, since the return is "
      "relative to P_0). P / P_orc: final path over the oracle's target path, per episode.\n")
    w("| policy | start | n | return | regret | path mean | path med | end >=31 % | >=57 % "
      "| >=89 % | cap % | term % | trunc % | len mean | len med | len p90 | P / P_orc mean |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in pol.items():
        for split in ("all", "blind", "informative"):
            s = m[split]
            w(f"| {name if split == 'all' else ''} | {split} | {s['n']} | "
              f"{_f(s['return_mean'])} | {_f(s['regret_mean'])} | "
              f"{_f(s['final_path_m_mean'], '.2f')} | "
              f"{_f(s['final_path_m_median'], '.2f')} | {_pct(s['end_ge31'])} | "
              f"{_pct(s['end_ge57'])} | {_pct(s['end_ge89'])} | {_pct(s['end_cap'])} | "
              f"{_pct(s['terminated'])} | {_pct(s['truncated'])} | {_f(s['length_mean'], '.1f')} | "
              f"{_f(s['length_median'], '.0f')} | {_f(s['length_p90'], '.0f')} | "
              f"{_f(s['path_frac_ref_mean'])} |")
    w("")

    # First reach
    w("## Steps to first reach\n")
    w("Fraction of episodes whose trajectory ever reaches >= k bounces (incl. the initial "
      "frame, step 0) and, among those, the mean / median step of the first such frame.\n")
    names = [str(v) for v in LEVELS] + ["cap"]
    w("| policy | start | " + " | ".join(f">={k} % | steps mean / med" for k in names) + " |")
    w("|---|---|" + "---|---|" * len(names))
    for name, m in pol.items():
        for split in ("all", "blind", "informative"):
            fr = m[split]["first_reach"]
            cells = []
            for k in names:
                r = fr[k]
                cells.append(f"{_pct(r['frac'])} | {_f(r['steps_mean'], '.1f')} / "
                             f"{_f(r['steps_median'], '.0f')}")
            w(f"| {name if split == 'all' else ''} | {split} | " + " | ".join(cells) + " |")
    w("")

    # Final bounces
    labels = [b[0] for b in bounce_bins(cap)]
    w("## Final bounce distribution (all starts, %)\n")
    w("| policy | " + " | ".join(labels) + " |")
    w("|---|" + "---|" * len(labels))
    for name, m in pol.items():
        bins = m["all"]["final_bounce_bins"]
        w(f"| {name} | " + " | ".join(_pct(bins[k]) for k in labels) + " |")
    w("")

    # Oracle
    w("## Oracle\n")
    w("From the start tilt s, the reachable tilts are s + 0.01 deg * (i, j) inside the box; "
      "tilts reachable only by clamping at the +-2 deg limits are ignored. Moves per axis: "
      "exact BFS over the integers with the signed ladder steps (e.g. 29 = 30 - 1: 2 moves); "
      "a target costs max(moves_x, moves_y). Each episode scans its full shifted grid "
      f"({orc['grid']} states, float32 tilt arithmetic as in the env, steps ordered so that "
      "no intermediate tilt clamps) and takes argmax P / P_nom - 0.01 * moves, ties to fewer "
      "moves.\n")
    w(f"- Planning: {orc['plan_s']:.1f} s for {cfg['n']} episodes "
      f"({1e3 * orc['plan_s'] / cfg['n']:.2f} ms per episode, {orc['grid']} states each, "
      f"chunks of {orc['chunk']} episodes, peak GPU memory {orc['peak_mem_gb']:.2f} GB).")
    w(f"- Max moves per axis over |d| <= 400: {orc['max_moves_axis']}. Grid targets whose "
      f"step walk would clamp: {orc['n_clamped']}.")
    w(f"- Realized vs planned return: max |diff| {orc['ret_diff_max']:.2e}; final path equal "
      f"to the planned target path in {orc['path_equal']} / {cfg['n']} episodes; start "
      f"mismatches {orc['start_mismatch']}.")
    w(f"- Mean planned return {orc['ret_mean']:.3f}. The longest path on the reachable grid "
      f"(cost ignored) exceeds the target path by {orc['max_minus_target_mm']:.2f} mm on "
      f"average (max {orc['max_minus_target_mm_max']:.2f} mm): trapped paths differ by "
      "< 0.5 mm, less than one step cost (27 mm).")
    w("")
    w("Moves to the target:\n")
    mv = orc["moves_hist"]
    keys = sorted(mv, key=int)
    w("| moves | " + " | ".join(keys) + " |")
    w("|---|" + "---|" * len(keys))
    w("| episodes | " + " | ".join(str(mv[k]) for k in keys) + " |")
    w(f"\nMean {orc['moves_mean']:.2f}, by start: blind {orc['moves_mean_blind']:.2f}, "
      f"informative {orc['moves_mean_informative']:.2f}.\n")
    w("Target region:\n")
    reg = orc["target_region"]
    w("| " + " | ".join(reg) + " |")
    w("|" + "---|" * len(reg))
    w("| " + " | ".join(f"{v} ({100 * v / cfg['n']:.1f} %)" for v in reg.values()) + " |")
    w("")

    # Scripted and homing
    w("## Scripted and homing\n")
    sp = res["scripted_params"]
    w(f"`scripted` (image only, uses `obs['image']` and its own action history): M2 spot "
      f"count = 3x3 local maxima >= {sp['peak_thr']}/255; brightness = sum of pixels / 255. "
      "While <= 1 spot is visible, an expanding square spiral of "
      f"{sp['search_step'] / 100:.2f} deg steps (legs 1, 1, 2, 2, 3, ...). From the first "
      f"frame with >= 2 spots, a hill-climb on brightness over {sp['n_dirs']} directions with "
      f"step sizes {', '.join(f'{s / 100:.2f}' for s in sp['sizes'])} deg: a gain > "
      f"{sp['eps']:g} keeps the move and repeats the direction, a loss > {sp['eps']:g} is "
      "reverted (one step), otherwise the move is kept and the next direction is tried; "
      f"{sp['n_dirs']} probes without a gain shrink the step, after the last size it stops. "
      "Brightness instead of the spot count because the count saturates near 30 from 57 "
      "bounces up (spots overlap at 64 px), while brightness keeps rising: ~11 (one spot), "
      "~144 (31), ~218 (57), ~250 (89), ~273 (trapped).\n")
    if "scripted_variants" in res:
        w("Variants (all starts, same eval set; the first row is `scripted`, the second the "
          "PLAN's suggestion):\n")
        w("| search step | sizes | dirs | eps | return | end >=31 % | >=57 % | >=89 % | cap % "
          "| len mean |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for v in res["scripted_variants"]:
            s = v["all"]
            w(f"| {v['params']['search_step'] / 100:.2f} | "
              f"{', '.join(f'{x / 100:.2f}' for x in v['params']['sizes'])} | "
              f"{v['params']['n_dirs']} | {v['params']['eps']:g} | {_f(s['return_mean'])} | "
              f"{_pct(s['end_ge31'])} | {_pct(s['end_ge57'])} | {_pct(s['end_ge89'])} | "
              f"{_pct(s['end_cap'])} | {_f(s['length_mean'], '.1f')} |")
        w("")
    hp = res["homing"]
    w(f"`homing` (image-free): pushes both axes by 1 deg toward the corner "
      f"{tuple(hp['corner'])} until `obs['action']` shows a clamped push, then replays the "
      f"corner's best plan ({hp['moves']} moves to "
      f"{tuple(round(v, 2) for v in hp['target_tilt'])}, "
      f"{hp['target_bounces']} bounces, {hp['target_path_mm'] / 1000:.2f} m) and stops. "
      f"Return above the oracle's by > 1e-4 in {_pct(hp['beats_oracle'])} % of episodes "
      f"(max difference {hp['beats_oracle_max']:.1e}), so ignoring clamp-only targets in the "
      "oracle costs nothing measurable here.\n")

    w("## Interpretation\n")
    w(res["interpretation"])
    w("")
    with open(path_md, "w") as fh:
        fh.write("\n".join(L))


def interpretation(res: Dict[str, Any]) -> str:
    """Short interpretation for the report, numbers from res."""
    pol, orc = res["policies"], res["oracle"]
    o, s, h = pol["oracle"], pol["scripted"], pol["homing"]
    sa, sb, si = s["all"], s["blind"], s["informative"]
    reg = orc["target_region"]
    return " ".join([
        f"The oracle returns {o['all']['return_mean']:.2f} on average and goes to the trapped "
        f"region ({orc['cap_path_m']:.2f} m) from {reg['trapped (cap)']} of {res['config']['n']} "
        f"starts (stops at once: {reg['stop at once']}, of which {orc['stop_trapped']} start "
        "inside it), in "
        f"{orc['moves_mean']:.1f} moves "
        f"(at most {max(int(k) for k in orc['moves_hist'])}): the 0.2 deg trapped spot is a "
        "few ladder steps from anywhere in the box, so the 31 / 57 / 89 patterns are never "
        "worth stopping at for a fully informed agent.",
        f"The image-only scripted policy returns {sa['return_mean']:.2f} "
        f"({100 * sa['return_mean'] / o['all']['return_mean']:.0f} % of the oracle, regret "
        f"{sa['regret_mean']:.2f}): it always ends at >= 31 bounces and at >= 57 in "
        f"{_pct(sa['end_ge57'])} %, but trapped in only {_pct(sa['end_cap'])} %, and spends "
        f"{sa['length_mean']:.0f} steps ({0.01 * (sa['moves_mean']):.2f} of return in step "
        "costs); brightness is nearly constant within the 31 plateau (141-146) and the 57 "
        "island (216-219), so the 0.2 deg 89 / trapped spots are found only when a probe "
        "happens to land in them.",
        f"For this heuristic the blind region is cheap: the 1 deg spiral reaches >= 31 bounces "
        f"after {sb['first_reach']['31']['steps_mean']:.1f} steps on average from blind starts, "
        f"and the regret is {sb['regret_mean']:.2f} (blind) vs {si['regret_mean']:.2f} "
        f"(informative); the oracle needs {orc['moves_mean_blind']:.2f} vs "
        f"{orc['moves_mean_informative']:.2f} moves (returns are higher from blind starts only "
        "because P_0 is lower).",
        f"The image-free homing policy returns {h['all']['return_mean']:.2f} "
        f"({100 * h['all']['return_mean'] / o['all']['return_mean']:.0f} % of the oracle): with "
        "one fixed geometry the mount limit gives the absolute tilt (the clamp shows in "
        "obs['action'], and 4 pushes force it anyway), so stage 1 is solvable open-loop from a "
        "memorized map; PPO can learn exactly that, so stage-1 success alone does not show "
        "that the policy reads the images.",
        f"Random policies have negative returns ({pol['random']['all']['return_mean']:.2f} with "
        f"stop, {pol['random_no_stop']['all']['return_mean']:.2f} without): step costs and "
        "wandering off the plateau outweigh the rare visits to long paths.",
    ])


# ── Multi-setup report ──────────────────────────────────────────────────────


MULTI_POLICIES = ("random", "random_no_stop", "always_stop", "oracle", "scripted",
                  "homing_aware", "homing_blind")


def write_report_multi(res: Dict[str, Any], path_md: str):
    """Markdown report of the multi-setup run (see main_multisetup)."""
    cfg, orc, pol = res["config"], res["oracle"], res["policies"]
    names = [s["name"] for s in res["setups"]]
    L: List[str] = []
    w = L.append
    w("# Phase 4 baselines, four setups\n")
    w(f"Generated by `uv run python baselines.py --multisetup` ({res['date']}). Env: "
      f"`EnvConfig(setups=geometry.SETUPS, auto_reset=False)`, {cfg['img_res']} px, max_steps "
      f"{cfg['max_steps']}, step_cost {cfg['step_cost']}, cap {cfg['max_bounces']}, P_ref = "
      "P_max of each setup (0.01 deg scan of the +-2 deg box), so the reward is "
      "dP / P_max(setup) - step_cost and the best return is about 1 - P_0 / P_max. "
      f"Evaluation set: {cfg['n']} (initial tilt, setup) pairs, tilts "
      f"`eval_initial_tilts(n={cfg['n']}, seed={cfg['seed']})`, setups "
      f"`eval_setups(n, 4, seed={cfg['seed']})` ({cfg['n'] // len(names)} per setup). "
      "Raw numbers: `baselines_multisetup.json`; single-setup report: `baselines.md`; "
      "setups: `setups.md`.\n")
    w("| setup | sep (mm) | laser pitch / yaw (deg) | nominal M2 (deg) | P_ref = P_max (m) | "
      "blind starts % |")
    w("|---|---|---|---|---|---|")
    for s in res["setups"]:
        w(f"| {s['name']} | {s['separation']:g} | {s['laser_pitch']:.2f} / {s['laser_yaw']:.2f} | "
          f"({s['m2_pitch']:+.2f}, {s['m2_yaw']:+.2f}) | {s['path_ref_mm'] / 1000:.2f} | "
          f"{_pct(s['blind_share'])} |")
    w("")

    o_all = pol["oracle"]["all"]["return_mean"]
    hb, ha = pol["homing_blind"]["all"], pol["homing_aware"]["all"]
    w("## Key number\n")
    w(f"Setup-blind homing returns {hb['return_mean']:.3f} vs the oracle's {o_all:.3f} "
      f"({100 * hb['return_mean'] / o_all:.0f} % of the oracle, regret "
      f"{hb['regret_mean']:.3f}); setup-aware homing (privileged setup) returns "
      f"{ha['return_mean']:.3f} ({100 * ha['return_mean'] / o_all:.0f} %). In the "
      "single-setup env homing got 99 % of the oracle (`baselines.md`).\n")
    w("| policy | " + " | ".join(names) + " | all |")
    w("|---|" + "---|" * (len(names) + 1))
    for p in ("oracle", "homing_aware", "homing_blind", "scripted"):
        cells = []
        for nm in names + ["all"]:
            key = nm if nm == "all" else f"setup:{nm}"
            r = pol[p][key]["return_mean"]
            ro = pol["oracle"][key]["return_mean"]
            cells.append(f"{r:.3f} ({100 * r / ro:.0f} %)")
        w(f"| {p} | " + " | ".join(cells) + " |")
    w("\nMean return per setup (in brackets: % of the oracle's mean return on the same "
      "episodes).\n")

    w("## Main metrics\n")
    w(f"Return = (P_final - P_0) / P_max(setup) - {cfg['step_cost']} * moves. Regret: oracle "
      "planned return minus return, per episode. P / P_max: final path over the setup's "
      f"P_max; success: P_final >= {SUCCESS_FRAC:g} P_max. P / P_orc: final path over the "
      "oracle's target path. Term / trunc: ended by (0, 0) / by the 64-step limit. Path in m.\n")
    w("| policy | split | n | return | regret | P / P_max | success % | P / P_orc | path mean "
      "| term % | trunc % | len mean | len p90 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    splits = ["all", "blind", "informative"] + [f"setup:{nm}" for nm in names]
    for name in MULTI_POLICIES:
        m = pol[name]
        for split in splits:
            s = m[split]
            w(f"| {name if split == 'all' else ''} | {split.replace('setup:', '')} | {s['n']} | "
              f"{_f(s['return_mean'])} | {_f(s['regret_mean'])} | "
              f"{_f(s['path_frac_pref_mean'])} | {_pct(s['success'])} | "
              f"{_f(s['path_frac_ref_mean'])} | {_f(s['final_path_m_mean'], '.2f')} | "
              f"{_pct(s['terminated'])} | {_pct(s['truncated'])} | "
              f"{_f(s['length_mean'], '.1f')} | {_f(s['length_p90'], '.0f')} |")
    w("")

    w("## Oracle\n")
    w("Per episode the exact clamp-free optimum of P / P_max(setup) - step_cost * moves over "
      "the 401^2 reachable grid (same construction as in `baselines.md`), knowing the setup "
      "and the initial tilt.\n")
    w(f"- Planning: {orc['plan_s']:.1f} s for {cfg['n']} episodes (peak GPU memory "
      f"{orc['peak_mem_gb']:.2f} GB). Grid targets whose walk would clamp: {orc['n_clamped']}.")
    w(f"- Realized vs planned return: max |diff| {orc['ret_diff_max']:.2e}; final path equal "
      f"to the planned target in {orc['path_equal']} / {cfg['n']}; start mismatches "
      f"{orc['start_mismatch']}.")
    w("")
    w("| setup | planned return | moves mean | stop at once % | target = best bounces % | "
      "target bounces (episodes) |")
    w("|---|---|---|---|---|---|")
    for nm in names:
        o = orc["per_setup"][nm]
        hist = ", ".join(f"{k}: {v}" for k, v in sorted(o["target_bounces"].items(),
                                                         key=lambda kv: -kv[1])[:5])
        w(f"| {nm} | {o['ret_mean']:.3f} | {o['moves_mean']:.2f} | {_pct(o['stop_frac'])} | "
          f"{_pct(o['at_best_frac'])} | {hist} |")
    w("")

    w("## Homing routes\n")
    w("Both variants push into a corner until the clamp shows in `obs['action']`, then replay "
      "a fixed route and stop. Value = mean over the route's setups of P / P_max - step_cost "
      "* moves at the target (the homing pushes are not included).\n")
    w("| variant | setup | corner | target M2 (deg) | moves | target bounces | P / P_max at "
      "target |")
    w("|---|---|---|---|---|---|---|")
    for cp, nm in zip(res["homing_aware"], names):
        w(f"| aware | {nm} | {tuple(cp['corner'])} | ({cp['target_tilt'][0]:+.2f}, "
          f"{cp['target_tilt'][1]:+.2f}) | {cp['moves']} | {cp['target_bounces']} | "
          f"{cp['target_path_mm'] / res['setups'][names.index(nm)]['path_ref_mm']:.3f} |")
    bp = res["homing_blind"]
    for i, nm in enumerate(names):
        w(f"| blind | {nm} | {tuple(bp['corner'])} | ({bp['target_tilt'][0]:+.2f}, "
          f"{bp['target_tilt'][1]:+.2f}) | {bp['moves']} | {bp['target_bounces'][i]} | "
          f"{bp['target_path_mm'][i] / res['setups'][i]['path_ref_mm']:.3f} |")
    w("")
    w("## Interpretation\n")
    w(res["interpretation"])
    w("")
    with open(path_md, "w") as fh:
        fh.write("\n".join(L))


def interpretation_multi(res: Dict[str, Any]) -> str:
    pol = res["policies"]
    names = [s["name"] for s in res["setups"]]
    o = pol["oracle"]["all"]["return_mean"]
    hb, ha, sc = (pol[p]["all"] for p in ("homing_blind", "homing_aware", "scripted"))
    bp = res["homing_blind"]
    fr = [bp["target_path_mm"][i] / res["setups"][i]["path_ref_mm"] for i in range(len(names))]
    per = {nm: pol["homing_blind"][f"setup:{nm}"]["return_mean"]
           / pol["oracle"][f"setup:{nm}"]["return_mean"] for nm in names}
    return " ".join([
        f"With four setups the image-free homing policy that ignores the setup returns "
        f"{hb['return_mean']:.2f}, {100 * hb['return_mean'] / o:.0f} % of the oracle's "
        f"{o:.2f} (single setup: 99 %): its one route ends at M2 "
        f"({bp['target_tilt'][0]:+.2f}, {bp['target_tilt'][1]:+.2f}), where the path is "
        + ", ".join(f"{f:.2f}" for f in fr) + " of P_max in " + ", ".join(names)
        + " (per setup " + ", ".join(f"{100 * v:.0f} %" for v in per.values())
        + " of the oracle).",
        f"Knowing the setup restores homing to {100 * ha['return_mean'] / o:.0f} % "
        f"({ha['return_mean']:.2f}): the gap is the value of identifying the setup, which an "
        "image-free agent cannot do and an image-reading agent can (the first M2 spot "
        "already differs between the setups, `setups.md`).",
        f"The image-only scripted heuristic returns {sc['return_mean']:.2f} "
        f"({100 * sc['return_mean'] / o:.0f} %), success {_pct(sc['success'])} % vs the oracle's "
        f"{_pct(pol['oracle']['all']['success'])} %.",
    ])


def main_multisetup(args):
    """Baselines on the 4-setup env (see the module docstring)."""
    env = AlignEnv(EnvConfig(num_envs=args.num_envs, seed=args.seed, setups=SETUPS,
                             auto_reset=False))
    dev = env.device
    c = env.cfg
    cap = c.max_bounces
    S = env.n_setups
    names = [g.name for g in env.setups]
    tilts = eval_initial_tilts(args.n, args.seed, env.limit, dev)
    setups = eval_setups(args.n, S, args.seed, dev)

    torch.cuda.reset_peak_memory_stats(dev)
    plan_oracle(env, tilts[: args.chunk], chunk=args.chunk, setups=setups[: args.chunk])
    plan = plan_oracle(env, tilts, chunk=args.chunk, setups=setups)
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    print(f"oracle plan: {plan.plan_s:.1f} s, peak {peak:.2f} GB, clamped {plan.n_clamped}")

    oracle = OraclePolicy(plan)
    hom_aware = HomingPolicy(env, variant="aware")
    hom_blind = HomingPolicy(env, variant="blind")
    scripted = ScriptedPolicy(env)
    policies = {
        "random": RandomPolicy(env, True, seed=1),
        "random_no_stop": RandomPolicy(env, False, seed=2),
        "always_stop": always_stop,
        "oracle": oracle,
        "scripted": scripted,
        "homing_aware": hom_aware,
        "homing_blind": hom_blind,
    }
    results: Dict[str, Any] = {}
    episodes: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, pol in policies.items():
        ep, met = evaluate(env, pol, tilts, plan.target_path_mm, plan.ret, init_setups=setups)
        episodes[name], results[name] = ep, met
        a = met["all"]
        per = " ".join(f"{nm} {met['setup:' + nm]['return_mean']:+.3f}" for nm in names)
        print(f"{name:15s} return {a['return_mean']:+.3f} (P/Pmax {a['path_frac_pref_mean']:.3f}, "
              f"success {a['success']:.3f}, len {a['length_mean']:.1f}) | {per}  "
              f"({met['wall_s']:.1f} s)")

    eo = episodes["oracle"]
    ret_diff = (eo["ep_return"].double() - plan.ret).abs()
    per_setup = {}
    for i, nm in enumerate(names):
        m = setups == i
        tb = plan.target_bounces[m]
        vals, cnt = torch.unique(tb, return_counts=True)
        per_setup[nm] = {
            "n": int(m.sum()),
            "ret_mean": float(plan.ret[m].mean()),
            "ret_diff_max": float(ret_diff[m].max()),
            "moves_mean": float(plan.moves[m].double().mean()),
            "stop_frac": float((plan.moves[m] == 0).double().mean()),
            "at_best_frac": float((plan.target_path_mm[m] >= 0.99 * env.path_refs[i]).double().mean()),
            "target_bounces": {str(int(v)): int(k) for v, k in zip(vals.tolist(), cnt.tolist())},
        }
    oracle_stats = {
        "plan_s": plan.plan_s, "chunk": args.chunk, "peak_mem_gb": peak,
        "n_clamped": plan.n_clamped, "ret_diff_max": float(ret_diff.max()),
        "path_equal": int((eo["final_path_mm"] == plan.target_path_mm).sum()),
        "start_mismatch": oracle.start_mismatch, "ret_mean": float(plan.ret.mean()),
        "per_setup": per_setup,
    }
    blind0 = episodes["oracle"]["init_bounces"] <= 2
    setups_info = []
    for i, g in enumerate(env.setups):
        m = setups == i
        setups_info.append({
            "name": g.name, "separation": g.separation, "laser_pitch": g.laser_pitch,
            "laser_yaw": g.laser_yaw, "m2_pitch": g.m2_pitch, "m2_yaw": g.m2_yaw,
            "path_ref_mm": float(env.path_refs[i]),
            "blind_share": float(blind0[m].double().mean()),
        })
    peak_total = torch.cuda.max_memory_allocated(dev) / 2**30
    print(f"oracle realized vs planned: max |diff| {oracle_stats['ret_diff_max']:.2e}, path "
          f"equal {oracle_stats['path_equal']}/{args.n}; peak GPU memory {peak_total:.2f} GB")
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "config": {
            "n": args.n, "seed": args.seed, "num_envs": args.num_envs, "limit": env.limit,
            "img_res": c.img_res, "max_steps": c.max_steps, "step_cost": c.step_cost,
            "max_bounces": cap, "path_ref": env.path_ref_mode, "auto_reset": c.auto_reset,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "peak_mem_gb": peak_total,
        },
        "setups": setups_info,
        "policies": results,
        "oracle": oracle_stats,
        "homing_aware": hom_aware.corner_plans,
        "homing_blind": hom_blind.corner_plan,
        "scripted_params": scripted.params,
    }
    res["interpretation"] = interpretation_multi(res)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baselines_multisetup.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_report_multi(res, os.path.join(args.out, "baselines_multisetup.md"))
    print(f"wrote {os.path.join(args.out, 'baselines_multisetup.md')} and .json")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=4096, help="evaluation episodes")
    ap.add_argument("--seed", type=int, default=0, help="evaluation set seed")
    ap.add_argument("--num-envs", type=int, default=1024, help="parallel envs per wave")
    ap.add_argument("--chunk", type=int, default=8, help="oracle: episodes per Triton call")
    ap.add_argument("--variants", action="store_true", help="also evaluate scripted variants")
    ap.add_argument("--out", default=os.path.join(RL_DIR, "reports"))
    ap.add_argument("--multisetup", action="store_true",
                    help="4-setup env (geometry.SETUPS), writes baselines_multisetup.*")
    args = ap.parse_args()
    if args.multisetup:
        main_multisetup(args)
        return

    env = AlignEnv(EnvConfig(num_envs=args.num_envs, seed=args.seed))
    dev = env.device
    c = env.cfg
    tilts = eval_initial_tilts(args.n, args.seed, env.limit, dev)
    cap = c.max_bounces

    # ── Oracle plan ──
    torch.cuda.reset_peak_memory_stats(dev)
    plan_oracle(env, tilts[: args.chunk], chunk=args.chunk)  # Warm-up (Triton compile)
    plan = plan_oracle(env, tilts, chunk=args.chunk)
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    print(f"oracle plan: {plan.plan_s:.1f} s ({1e3 * plan.plan_s / args.n:.2f} ms / episode), "
          f"peak {peak:.2f} GB, clamped {plan.n_clamped}")

    # ── Policies ──
    oracle = OraclePolicy(plan)
    scripted = ScriptedPolicy(env)
    homing = HomingPolicy(env)
    policies = {
        "random": RandomPolicy(env, True, seed=1),
        "random_no_stop": RandomPolicy(env, False, seed=2),
        "always_stop": always_stop,
        "oracle": oracle,
        "scripted": scripted,
        "homing": homing,
    }
    results: Dict[str, Any] = {}
    episodes: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, pol in policies.items():
        ep, met = evaluate(env, pol, tilts, plan.target_path_mm, plan.ret)
        episodes[name], results[name] = ep, met
        a = met["all"]
        print(f"{name:15s} return {a['return_mean']:+.3f}  path {a['final_path_m_mean']:.2f} m  "
              f">=31 {a['end_ge31']:.3f}  >=57 {a['end_ge57']:.3f}  cap {a['end_cap']:.3f}  "
              f"len {a['length_mean']:.1f}  ({met['wall_s']:.1f} s)")

    variants = []
    if args.variants:
        for kw in (
            dict(search_step=30, sizes=(30, 10, 3), eps=4.0),
            dict(search_step=30, sizes=(30, 10, 3), eps=4.0, n_dirs=4),
            dict(search_step=30, sizes=(100, 30)),
            dict(search_step=100, sizes=(100, 30, 10)),
            dict(search_step=100, sizes=(30,)),
            dict(search_step=100, sizes=(100, 30), n_dirs=4),
        ):
            sp = ScriptedPolicy(env, **kw)
            _, met = evaluate(env, sp, tilts, plan.target_path_mm, plan.ret)
            variants.append({"params": sp.params, **met})
            print(f"scripted {sp.params}: return {met['all']['return_mean']:+.3f}")
        variants.insert(0, {"params": scripted.params, **results["scripted"]})

    # ── Oracle checks ──
    eo = episodes["oracle"]
    ret_diff = (eo["ep_return"].double() - plan.ret).abs()
    blind = eo["init_bounces"] <= 2
    mv_vals, mv_cnt = torch.unique(plan.moves, return_counts=True)
    eh = episodes["homing"]
    beat = eh["ep_return"].double() - eo["ep_return"].double()
    oracle_stats = {
        "plan_s": plan.plan_s,
        "chunk": args.chunk,
        "grid": f"{int(round(2 * env.limit / c.step_unit)) + 1}^2",
        "peak_mem_gb": peak,
        "max_moves_axis": plan.max_moves_axis,
        "n_clamped": plan.n_clamped,
        "ret_diff_max": float(ret_diff.max()),
        "path_equal": int((eo["final_path_mm"] == plan.target_path_mm).sum()),
        "path0_equal": int((eo["init_path_mm"] == plan.path0_mm).sum()),
        "start_mismatch": oracle.start_mismatch,
        "ret_mean": float(plan.ret.mean()),
        "max_minus_target_mm": float((plan.max_path_mm - plan.target_path_mm).mean()),
        "max_minus_target_mm_max": float((plan.max_path_mm - plan.target_path_mm).max()),
        "moves_mean": float(plan.moves.double().mean()),
        "moves_mean_blind": float(plan.moves[blind].double().mean()),
        "moves_mean_informative": float(plan.moves[~blind].double().mean()),
        "moves_hist": {str(int(v)): int(k) for v, k in zip(mv_vals.tolist(), mv_cnt.tolist())},
        "target_region": region_of(plan.target_bounces, plan.moves, cap),
        "stop_trapped": int(((plan.moves == 0) & (plan.target_bounces >= cap)).sum()),
        "cap_path_m": float(plan.target_path_mm[plan.target_bounces >= cap].mean() / 1000),
    }
    homing_stats = {
        **homing.corner_plan,
        "beats_oracle": float((beat > 1e-4).double().mean()),
        "beats_oracle_max": float(beat.max()),
    }
    peak_total = torch.cuda.max_memory_allocated(dev) / 2**30
    print(f"peak GPU memory over the run: {peak_total:.2f} GB")
    print(f"oracle realized vs planned return: max |diff| {oracle_stats['ret_diff_max']:.2e}, "
          f"path equal {oracle_stats['path_equal']}/{args.n}, start mismatch "
          f"{oracle.start_mismatch}")

    res = {
        "date": time.strftime("%Y-%m-%d"),
        "config": {
            "n": args.n, "seed": args.seed, "num_envs": args.num_envs, "limit": env.limit,
            "img_res": c.img_res, "max_steps": c.max_steps, "step_cost": c.step_cost,
            "max_bounces": cap, "step_ladder": list(c.step_ladder), "step_unit": c.step_unit,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "blind_share": results["oracle"]["blind"]["share"],
            "peak_mem_gb": peak_total,
        },
        "policies": results,
        "oracle": oracle_stats,
        "homing": homing_stats,
        "scripted_params": scripted.params,
    }
    if variants:
        res["scripted_variants"] = variants
    res["interpretation"] = interpretation(res)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baselines.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_report(res, os.path.join(args.out, "baselines.md"))
    print(f"wrote {os.path.join(args.out, 'baselines.md')} and baselines.json")


if __name__ == "__main__":
    main()
