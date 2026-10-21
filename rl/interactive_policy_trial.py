"""
interactive_policy_trial.py - Interactive simulator and policy viewer (stage 2 agent)

Run from rl/ (WSLg: TkAgg, the only interactive backend installed here; no Qt):
    uv run python interactive_policy_trial.py                          # newest runs/ppo/*/*best*.pt
    uv run python interactive_policy_trial.py --ckpt runs/ppo/<run>/best.pt
    uv run python interactive_policy_trial.py --no-ckpt                # simulator only
    uv run python interactive_policy_trial.py --selftest               # headless (Agg)
--selftest drives the callbacks programmatically, plays episodes, checks them against
evaluation.evaluate (same trajectory, same oracle ref_return) and the decision log, and
writes figures/viewer.png; with --backend TkAgg it does the same in a real window. The UI
is laid out for 18.8 x 10.6 in at 100 dpi (1880 x 1060 px); --dpi scales all of it.

Panels
    landscape  the M2 +-2 deg box on the atlas grid (0.01 deg, 401^2 cells, cap 128),
               colored by bounce count or P / P_max; one Triton scan per scenario change.
               Overlay: the episode's tilt trajectory, start, current tilt, best tilt
               (P_max), the tilt-head estimate (mean +- 1 std) and, once the episode
               ended, the oracle's target. Below: the scenario and the last episode vs its
               oracle
    camera     the 64 px uint8 image the agent receives, in the render style (sim:
               env.SpotRenderer; camera: render_camera.CameraRenderer with the camera
               look) and the agent's input channels (RGB for 3, gray for 1), step, bounce
               count, path, stop reason, input (style, channels), episode state
    look       camera look panel (right of the camera image), see Camera look
    beam       x-z and y-z side views (z / separation) of the ray between the mirrors
               (dashed: the leaving ray), both mirror faces with the hit points (local x / y,
               color = bounce index)
    policy     per-axis ladder probabilities (zero step highlighted, next / taken step
               outlined), P(not move) = p_pitch(0) p_yaw(0), value (de-normalized), path,
               tilt and scenario heads (mean +- std) against the true values
Scenario space: (separation 20-400 mm, laser pitch and yaw -8..+8 deg, laser dx / dy: the
offset from the M1 hole centre, anywhere in the hole disk of radius 1.5 mm; measured cell
0.75 / 1.04 mm). Validity from the box scan: CLIPPED BY HOLE (0 bounces and BLOCKED at
zero M2 tilt: the laser hits the back of M1 at the hole edge, independent of the tilt),
MISSES M2 (0 bounces at every tilt), else VALID; good = valid and P_max >= 20 x
separation. P_max, best tilt and good come from the atlas for its grid points and named
setups (measured offset), otherwise from the scan. Training range of the loaded
checkpoint: its scenario-head range (agent config scen_lo / scen_hi; the offsets count as
trained when the head predicts laser_dx / laser_dy, else only the measured offset);
4407578: separation 50-200 mm, laser 0-8 deg, measured offset. Scenarios outside it are
marked EXTRAPOLATION. The scenario head's parameters and normalization come from the
checkpoint config (a scen* list of names, else 3 or 5 values in the order separation,
laser pitch, laser yaw, laser dx, laser dy); a head this agent.py cannot load (another
number of values) is reported as not loaded.
Controls and keys
    mouse      landscape: click or drag sets the start tilt (new episode, mount resolution
               0.01 deg); laser-offset disk: click or drag sets laser dx / dy (kept inside the
               hole); sliders: separation (0.5 mm steps), laser pitch / yaw, M2 pitch / yaw
               (start tilt; work without a policy); value box right of each slider: click,
               type a number, Enter applies it (clipped to the slider range, snapped to its
               step), Escape or a click elsewhere cancels; checkpoint path box (Enter loads)
    buttons    list / prev / next (right of the checkpoint box) over runs/ppo/*/*best*.pt
               (sorted by run and file name, rescanned on every click so newly copied files
               appear): list opens a menu of them (a section per run directory, the loaded
               one marked; a native Tk menu, TkAgg only), prev / next take the previous /
               next file (from the loaded checkpoint, else the first / last); the file is
               loaded as from the box, which then shows its path; render: sim / camera
               (the env is rebuilt and a new episode starts at the start tilt): sim =
               env.SpotRenderer (1 channel, runs r2-r6), camera =
               render_camera.CameraRenderer under the camera look (3 channels). A loaded
               checkpoint starts in camera if it trained with --camera, else in sim (the
               camera look stays). The image reaches the agent with its
               encoder's in_ch channels: 1 -> 3 copies the gray image (as the 3-channel
               encoders were trained to take it), 3 -> 1 takes the channel mean (rounded);
               the camera panel shows that image
               named setups (measured, sep116, sep144, sep169: their separation, laser tilt
               and measured offset; the start moves to the setup's nominal tilt), ROC 200 /
               150 mm (mirror radius of curvature, both mirrors; 200: the measured cell, the
               atlas and every run; 150: the simulator of March 2026): a switch rescans the
               scenario and starts a new episode at the start tilt, at 150 P_max comes from
               the scan and every scenario is EXTRAPOLATION, random
               good (separation 20-400 mm), random short (20-49 mm), random long (201-400
               mm): laser angles uniform in -8..+8 deg, offset uniform in the hole disk, only
               valid scenarios, the first good one of up to 64 candidates (else the valid one
               closest to good); Step, Run, Pause, Reset (restart at the start tilt), Random
               start (uniform in the run's init box), policy argmax / sampled, map color
    keys       right or s: step; space: run / pause; r: reset; n: random start; d: argmax /
               sampled; m: map color; g: random good scenario; [ / ]: separation -/+0.5 mm;
               { / }: -/+5 mm
Camera look: the parameters of every camera render are fixed (AlignEnv draws none per
episode: Session._install_look), so tilt changes and new episodes keep the look; they are
a base look with the panel settings on top. Any change starts a new episode at the start
tilt; in sim it is kept for the camera style.
    new look   base = a draw from the run's training ranges (ppo.camera_ranges(training=
               True); a run without --camera: CameraRanges() without rotation; never clean)
    clean      base = every effect off (the camera runs' evals; the default)
               (both set the panel from the base, except the scatter amplitude)
    camera     mono / red / green / blue laser (a color base keeps its crosstalk, tint and
               white balance; from mono: 0.1 crosstalk, neutral color)
    noise      frozen (default): the camera generator restarts before every render, so
               the per-frame draws (sensor noise, speckle, edge-scatter shape) repeat and
               the same tilt gives the same frame; live: redrawn every frame (training);
               off: no read / shot / row noise and speckle (draws repeat as frozen)
    scatter    edge scatter on / off (amplitude: slider; drawn per frame in training)
    pose       off: no rotation / shift / scale / perspective (on: the base's)
    marks      off: no rings, grain, engraving, scratches, dust, rim glow (on: the base's)
    sliders    lit face (face brightness after the sensor, 0 = dark), laser (first-spot
               peak), spot sigma (mm), scatter (amplitude), exposure, gamma, read noise
               (std), speckle (per-spot log-normal std); a value box as the other sliders
Decision log (one pair of files per session in runs/viewer_logs/, gitignored; --no-log):
    <session>_steps.csv     a row per agent step: the scenario (incl. laser_dx_mm /
                            laser_dy_mm, roc_mm), render style, camera look (look:
                            Session.look_text, "" in sim), tilt before / after, the
                            action (ladder step and applied delta, deg; not_move), P(not
                            move), the
                            probability of the taken step and the top 3 steps per axis,
                            the full per-axis distributions (p_pitch_<units>, p_yaw_...),
                            value, path head vs the true path (m, P / P_ref), tilt head
                            (mean, std) vs the tilt before, scenario head (mean, std per
                            parameter, scen_head_<sep_mm|laser_pitch|laser_yaw|laser_dx_mm|
                            laser_dy_mm>, nan where the head has no value) vs the scenario,
                            reward, return so far, stop reason, done flags
    <session>_episodes.csv  a row per played episode (stopped, truncated, or abandoned by
                            a new start / scenario / checkpoint / ROC / render style / look):
                            scenario (incl. laser_dx_mm / laser_dy_mm, roc_mm), render
                            style, camera look, P_max, valid / invalid_reason,
                            in_training_range (0: extrapolation) / range_note, start,
                            steps, moves, final tilt and path, P / P_max, return, and the
                            oracle of that scenario and start (baselines.plan_oracle):
                            return, regret, % of the oracle, target tilt, path, moves
    episode numbers count every episode start of the session; each logged episode is also
    printed as one line, and the last one is shown under the scenario text
Agent state: as ppo.Trainer.rollout at batch 1. AlignEnv in scenario mode with one scenario
(P_ref = max(P_max, P_floor), the atlas rule), no auto-reset, history_len 0, the run's
max_steps / step_cost / image size (ppo.env_config), the camera of the render style.
agent.act() on each new frame (Session.agent_image: its encoder's channels) with the
applied action that led to it (zero at the episode start) and the episode-start flag; the
Ring holds the last K = 16 (embedding, action) tokens. Action = per-axis argmax
(agent.greedy) or Gumbel-max sample (agent.sample); the episode ends on (0, 0) or after
max_steps. The act() output on the current frame is shown before the step; Step commits
its ring and applies its action (the next action shown is the one taken).
Rendering: one figure, artists updated in place, no axes cleared or re-created. Every
changing artist is animated and blitted in two layers over the background of the last full
draw (startup, resize): "scene" (landscape, scenario text and sliders, settings) is redrawn
and captured only when the scenario or a setting changes, "frame" (the rest) on every
update. Panel texts are FastText (FreeType line bitmaps cached per line; matplotlib 3.11
rasterizes Text glyph by glyph in Python, the main cost before), labels are static. Slider
and button redraw hooks are off, and the checkpoint box does not redraw the whole figure
on every click (matplotlib's TextBox does); a slider value box redraws only its slider; a
checkpoint picked with list / prev / next costs one full draw (the box is in the
background).
Slider and map-drag recomputes are throttled (one per throttle_ms, the last value always
applied, flushed on mouse release); a coalesced slider event only redraws its slider. Run
steps on a canvas timer.
GPU: simulation, rendering and the agent stay on the device; agent.act() replays a CUDA
graph (GraphAct: bit-identical, ~1.4 ms instead of 5-10 ms of kernel launches). One
device-to-host copy per frame; a scenario change adds the copy of the scan (and AlignEnv's
own P_ref read).
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib.artist import Artist
from matplotlib.transforms import Bbox

from agent import Agent, AgentConfig, Ring, greedy, sample
from baselines import MoveTable, ladder_units, plan_oracle
from env import AlignEnv, EnvConfig, Scenarios
from find_setups import box_grid, candidate_states
from geometry import SETUPS, CellGeometry, setup_by_name
from herriott_sim import (DEVICE, S_LASER_DX, S_LASER_DY, STOP_BLOCKED, STOP_HOLE, STOP_MISSED,
                          STOP_NAMES, HerriottSim)
from ppo import PPOConfig, ValueNorm, agent_config, camera_ranges, env_config
from scenarios import DEFAULT_ATLAS, AtlasConfig, load_atlas

RL_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.path.join(RL_DIR, "runs", "ppo")


def best_checkpoints(root: str = CKPT_ROOT) -> List[str]:
    """Every *best*.pt (r<N>-<tag>_best*.pt) under runs/ppo/<run>/, absolute paths sorted by
    run and file name (the checkpoint picker)."""
    return sorted(os.path.abspath(p) for p in glob.glob(os.path.join(root, "*", "*best*.pt")))


def newest_best(root: str = CKPT_ROOT) -> str:
    """The most recently written *best*.pt (r<N>-<tag>_best*.pt) under runs/ppo/<run>/ (copies from the cluster keep
    the cluster's write time with rsync -a), "" if there is none."""
    paths = best_checkpoints(root)
    return max(paths, key=os.path.getmtime) if paths else ""


DEFAULT_CKPT = newest_best()
FIG_DIR = os.path.join(RL_DIR, "figures")

# ── Colors (reference palette; bounce ramp as find_setups.py) ──
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BTN = "#ecebe7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#6250d6"
RED = "#e34948"
BOUNCE_RAMP = ("#f7f9fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
               "#0d366b")
HIT_RAMP = ("#fbd3c0", "#f29a6e", "#d95926", "#9a3a12", "#5c2209")

# ── Layout: design pixels of the default 18.8 x 10.6 in figure at 100 dpi (left, bottom,
# width, height), stored as figure fractions (another figsize / dpi scales everything) ──
_W, _H = 1880.0, 1060.0


def _px(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (x / _W, y / _H, w / _W, h / _H)


MAP_RECT = _px(80, 525, 490, 490)
CBAR_RECT = _px(582, 525, 12, 490)
SCEN_TXT_RECT = _px(30, 380, 590, 38)
LAST_TXT_RECT = _px(30, 352, 590, 22)
CAM_RECT = _px(655, 730, 300, 300)
CAM_TXT_RECT = _px(655, 570, 520, 148)
FACE_RECTS = (_px(670, 265, 235, 235), _px(925, 265, 235, 235))
HIT_CBAR_RECT = _px(1170, 280, 9, 200)
SIDE_RECTS = (_px(1265, 925, 595, 105), _px(1265, 768, 595, 105))
BAR_RECTS = (_px(1265, 600, 265, 95), _px(1595, 600, 265, 95))
POL_RECT = _px(1240, 290, 620, 230)
HOLE_RECT = _px(535, 160, 95, 95)
HOLE_TXT_RECT = _px(522, 120, 125, 36)
CKPT_RECT = _px(655, 40, 1205, 22)
MSG_RECT = _px(655, 14, 1205, 22)
FRAME_BLIT = (_px(0, 225, 1880, 835), _px(0, 0, 640, 62))  # Frame-layer blit areas

# ── Font sizes (pt) ──
FS_TICK = 9.0
FS_LABEL = 10.0
FS_TITLE = 11.0
FS_TEXT = 10.0  # Panel values and labels (monospace)
FS_SMALL = 9.0  # Legend and info lines
FS_HEAD = 11.5  # Control group headers

# ── Scenario space: (separation, laser pitch, laser yaw, laser dx, laser dy), the offsets
# from the M1 hole centre anywhere in the hole disk (the training space of the runs after
# 4407578, which trained on 50-200 mm, 0-8 deg and the measured offsets) ──
SEP_RANGE = (20.0, 400.0)  # mm (slider step 0.5 mm; keys [ ] and { })
ROCS = (200.0, 150.0)  # mm, mirror ROC button (200: measured cell; 150: March 2026 sim)
RENDER_STYLES = ("sim", "camera")  # Render button (env images)
# Camera look panel: camera button, noise button, sliders (CameraParams key, label, range
# (the CameraRanges defaults; 0 = off where it has an off), step, format)
LOOK_CAMERAS = ("mono", "red", "green", "blue")
LOOK_NOISE = ("frozen", "live", "off")
LOOK_SLIDERS = (("face", "lit face", (0.0, 0.7), 0.01, "%.2f"),
                ("laser", "laser", (0.4, 6.0), 0.05, "%.2f"),
                ("sigma", "spot sigma", (0.2, 0.6), 0.01, "%.2f mm"),
                ("scatter_amp", "scatter", (1.0, 20.0), 0.5, "%.1f"),
                ("exposure", "exposure", (0.4, 2.5), 0.05, "%.2f"),
                ("gamma", "gamma", (0.7, 1.4), 0.01, "%.2f"),
                ("read_noise", "read noise", (0.0, 0.03), 0.001, "%.3f"),
                ("speckle", "speckle", (0.0, 0.3), 0.01, "%.2f"))
MARK_KEYS = ("ring_a", "grain", "engrave_amp", "scratch_amp", "dust_amp", "rim")  # Face marks
ANGLE_RANGE = (-8.0, 8.0)  # deg, laser pitch and yaw
SHORT_SEPS = (20.0, 49.0)  # "random short"
LONG_SEPS = (201.0, 400.0)  # "random long"
RANDOM_TRIES = 64  # Random scenario buttons: candidates scanned at most
RANDOM_BATCH = 8  # Candidates per box scan
PARAM_NAMES = ("separation", "laser_pitch", "laser_yaw", "laser_dx", "laser_dy")
PARAM_COLS = ("sep_mm", "laser_pitch", "laser_yaw", "laser_dx_mm", "laser_dy_mm")  # Log names


# ── Config ──────────────────────────────────────────────────────────────────


def ratio(a: float, b: float) -> float:
    """a / b, nan when b <= 0 (P_max is 0 where the beam misses M2 at every tilt)."""
    return a / b if b > 0 else math.nan


def run_tag(path: str, name: Optional[str]) -> str:
    """Short checkpoint tag: the run directory when it is a job id (runs/ppo/<id>/), else
    the trailing job id of the run name, else the file name."""
    d = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if d.isdigit():
        return d
    tail = (name or "").rsplit("_", 1)[-1]
    return tail if tail.isdigit() else os.path.splitext(os.path.basename(path))[0]


def canonical_param(name: str) -> str:
    """Scenario-head parameter name in PARAM_NAMES terms (sep..., ...pitch, ...yaw, ...dx,
    ...dy), else unchanged."""
    k = name.lower()
    for key, canon in (("sep", "separation"), ("pitch", "laser_pitch"), ("yaw", "laser_yaw"),
                       ("dx", "laser_dx"), ("dy", "laser_dy")):
        if key in k:
            return canon
    return name


def head_spec(agent_cfg: dict) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Scenario-head parameter names and normalization range (lo, hi) of a checkpoint's
    agent config: lo / hi from scen_lo / scen_hi (AgentConfig defaults if absent), names
    from a scen* field holding as many strings (canonicalized), else PARAM_NAMES[:n] for
    n = 3 or 5. The head predicts (x - mid) / half per parameter, mid / half of lo / hi."""
    d = AgentConfig()
    lo = tuple(float(v) for v in agent_cfg.get("scen_lo", d.scen_lo))
    hi = tuple(float(v) for v in agent_cfg.get("scen_hi", d.scen_hi))
    n = len(lo)
    names = None
    for k, v in agent_cfg.items():
        if ("scen" in k and isinstance(v, (list, tuple)) and len(v) == n
                and all(isinstance(x, str) for x in v)):
            names = tuple(canonical_param(x) for x in v)
            break
    if names is None:
        names = PARAM_NAMES[:n] if n in (3, 5) else tuple(f"param{i}" for i in range(n))
    return names, lo, hi


@dataclass
class ViewerConfig:
    """Viewer settings (CLI arguments with the same names)."""

    ckpt: str = DEFAULT_CKPT  # "" = simulator only
    atlas: str = DEFAULT_ATLAS  # Missing file = no atlas (box scans only)
    device: str = ""  # "" = herriott_sim.DEVICE
    seed: int = 0  # Random starts, random good scenarios, sampled actions
    cuda_graph: bool = True  # agent.act() in a CUDA graph (GraphAct); False: eager
    throttle_ms: float = 50.0  # Slider / map-drag recompute interval
    run_interval_ms: int = 40  # Run timer: pause between steps (step time comes on top)
    figsize: Tuple[float, float] = (18.8, 10.6)
    log_dir: str = os.path.join(RL_DIR, "runs", "viewer_logs")  # Decision log; "" = none
    log_tag: str = ""  # Appended to the session name of the log files
    dpi: float = 100.0


# ── Host snapshots ──────────────────────────────────────────────────────────


@dataclass
class ScenarioInfo:
    """A scenario and its M2 landscape (host copies)."""

    name: str
    params: Tuple[float, ...]  # Separation (mm), laser pitch, yaw (deg), laser dx, dy (mm)
    atlas_index: int  # -1: not in the atlas (values from the scan)
    pmax: float  # mm, longest path in the box
    best_tilt: Tuple[float, float]  # M2 pitch, yaw (deg) reaching P_max
    best_bounces: int
    p_floor: float  # floor_bounces x separation
    path_ref: float  # P_ref = max(P_max, P_floor), the env's reward scale
    good: bool  # Valid and P_max >= P_floor
    valid: bool  # The laser enters the cavity and reaches M2 at some tilt
    invalid_reason: str  # "", "clipped by hole" (BLOCKED at bounce 0), "misses M2" (0 bounces)
    in_training_range: bool  # Inside the loaded checkpoint's training range
    range_note: str  # The violated training ranges ("" inside)
    bounces: np.ndarray  # (W, W) int16, rows = pitch, cols = yaw
    path: np.ndarray  # (W, W) float32 mm
    scan_ms: float


@dataclass
class PolicyView:
    """Agent outputs on one frame (host copies)."""

    probs: np.ndarray  # (2, n_bins) per-axis ladder probabilities (env ladder order)
    p_stop: float  # P(not move) = p_pitch(0) p_yaw(0)
    value: float  # De-normalized value (return-to-go, units of P_ref)
    path_norm: float  # Path head: P_t / P_ref
    tilt_mu: np.ndarray  # (2,) deg
    tilt_std: np.ndarray  # (2,) deg
    scen_mu: np.ndarray  # (n,) scenario head means in Session.head_names order (0 if none)
    scen_std: np.ndarray  # (n,)
    act_greedy: np.ndarray  # (2,) ladder indices (argmax per axis)
    act_sample: np.ndarray  # (2,) ladder indices (one Gumbel-max sample)


@dataclass
class Frame:
    """The current env frame (host copies, one device-to-host copy)."""

    step: int
    tilt: np.ndarray  # (2,) M2 pitch, yaw (deg)
    path_mm: float
    bounces: int
    stop: int  # herriott_sim STOP_*
    image: np.ndarray  # (H, W) gray or (H, W, 3) RGB uint8, the agent's input
    applied: np.ndarray  # (2,) deg, applied delta that led to the frame (0 at the start)
    hits: np.ndarray  # (n, 3) world hit points
    hit_mirror: np.ndarray  # (n,) 1 = M1, 2 = M2
    hit_local: np.ndarray  # (n, 2) mirror-local x, y (mm)
    ep_return: float = 0.0  # Episode return so far (env)
    moves: int = 0  # Moves so far
    reward: float = 0.0  # Reward of the step that led to the frame
    terminated: bool = False  # Not move
    truncated: bool = False  # max_steps reached
    action: Optional[np.ndarray] = None  # Ladder indices of the action that led to the frame
    policy: Optional[PolicyView] = None  # On a terminated frame: the output that stopped

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def to_host(items: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """One device-to-host copy of several tensors: flattened into one float32 buffer
    (integers below 2^24 are exact), split back into arrays of the original shapes."""
    flat = [v.reshape(-1).float() for v in items.values()]
    buf = torch.cat(flat).cpu().numpy()
    out, i = {}, 0
    for (k, v), f in zip(items.items(), flat):
        n = f.numel()
        out[k] = buf[i:i + n].reshape(tuple(v.shape))
        i += n
    return out


def short_path(path: str) -> str:
    """Path relative to rl/ when inside it, else absolute."""
    if not path:
        return ""
    p = os.path.abspath(path)
    return os.path.relpath(p, RL_DIR) if p.startswith(RL_DIR + os.sep) else p


def rotation(pitch: float, yaw: float) -> np.ndarray:
    """(3, 3) rotation of herriott_sim (Ry(yaw) Rx(pitch), degrees) on the host."""
    p, y = (torch.tensor([float(v)], dtype=torch.float64) for v in (pitch, yaw))
    return HerriottSim.rotation_matrix(p, y)[0].numpy()


# ── Fast monospace text ─────────────────────────────────────────────────────


class FastText(Artist):
    """Left-aligned multi-line monospace text anchored at its top left (x, y in axes
    units), rasterized per line by FreeType in C (FT2Font.draw_glyphs_to_bitmap) and cached
    per line string; drawn with renderer.draw_image. matplotlib 3.11's Text rasterizes glyph
    by glyph in Python (~0.2 ms layout per line plus ~20 us per glyph), the main per-update
    cost of the panels. No rotation, math text or kerning (monospace)."""

    _fonts: Dict[Tuple[float, float], object] = {}
    _CACHE = 2048  # Lines per artist

    def __init__(self, ax, x: float, y: float, text: str = "", size: float = FS_TEXT,
                 color: str = INK, linespacing: float = 1.25):
        super().__init__()
        self._xy = (x, y)
        self._text = text
        self._size = size
        self._rgb = (np.asarray(matplotlib.colors.to_rgb(color)) * 255).astype(np.uint8)
        self._linespacing = linespacing
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self.set_transform(ax.transAxes)
        self.set_in_layout(False)
        ax.add_artist(self)

    def set_text(self, text: str):
        if text != self._text:
            self._text = text
            self.stale = True

    def get_text(self) -> str:
        return self._text

    def _font(self, dpi: float):
        key = (self._size, dpi)
        f = self._fonts.get(key)
        if f is None:
            from matplotlib import font_manager, ft2font

            path = font_manager.findfont(font_manager.FontProperties(family="monospace"))
            f = ft2font.FT2Font(path, matplotlib.rcParams["text.hinting_factor"])
            f.set_size(self._size, dpi)
            self._fonts[key] = f
        return f

    def _line(self, line: str, dpi: float) -> tuple:
        """(RGBA image, x offset, descent) of one line, cached."""
        hit = self._cache.get(line)
        if hit is not None:
            self._cache.move_to_end(line)
            return hit
        from matplotlib import ft2font

        f = self._font(dpi)
        f.clear()
        f.set_size(self._size, dpi)
        f.set_text(line, 0.0, flags=ft2font.LoadFlags.FORCE_AUTOHINT)
        f.draw_glyphs_to_bitmap(antialiased=True)
        gray = np.asarray(f.get_image())
        rgba = np.empty(gray.shape + (4,), np.uint8)  # draw_image: row 0 at the bottom
        rgba[..., :3] = self._rgb
        rgba[..., 3] = gray[::-1]
        out = (rgba, f.get_bitmap_offset()[0] / 64.0, f.get_descent() / 64.0)
        self._cache[line] = out
        if len(self._cache) > self._CACHE:
            self._cache.popitem(last=False)
        return out

    def draw(self, renderer):
        if not self.get_visible() or not self._text:
            return
        dpi = renderer.dpi
        x0, top = self.get_transform().transform(self._xy)
        asc = 0.76 * self._size * dpi / 72  # Height of "l" (DejaVu Sans Mono): top alignment
        lh = self._size * self._linespacing * dpi / 72
        gc = renderer.new_gc()
        for i, line in enumerate(self._text.split("\n")):
            if not line.strip():
                continue
            rgba, xo, d = self._line(line, dpi)
            base = top - asc - i * lh
            renderer.draw_image(gc, round(x0 + xo), round(base - d), rgba)
        gc.restore()
        self.stale = False


# ── Agent step in a CUDA graph ──────────────────────────────────────────────


class GraphAct:
    """agent.act() at batch 1 captured in a CUDA graph: eager it is ~200 kernel launches,
    launch bound (5-10 ms under WSL); a replay takes ~1.4 ms with bit-identical outputs
    (checked by --selftest against evaluation.evaluate). Inputs are copied into static
    buffers; the head outputs stay in graph memory until the next call, the new ring is
    returned as a copy."""

    def __init__(self, agent: Agent, device: torch.device):
        self.agent = agent
        e = agent.cfg.encoder
        self.image = torch.zeros(1, e.in_ch, e.img_res, e.img_res, dtype=torch.uint8,
                                 device=device)
        self.prev = torch.zeros(1, 2, device=device)
        self.first = torch.ones(1, dtype=torch.bool, device=device)
        self.ring = agent.init_ring(1, device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # Warm-up (allocator, cuDNN) off the capture
            for _ in range(3):
                agent.act(self.image, self.prev, self.first, self.ring)
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out, self.ring_out = agent.act(self.image, self.prev, self.first, self.ring)

    def __call__(self, image: torch.Tensor, prev_action: torch.Tensor, first: torch.Tensor,
                 ring: Ring) -> Tuple[Dict[str, torch.Tensor], Ring]:
        self.image.copy_(image)
        self.prev.copy_(prev_action)
        self.first.copy_(first)
        for dst, src in zip(self.ring, ring):
            dst.copy_(src)
        self.graph.replay()
        return self.out, Ring(*(t.clone() for t in self.ring_out))


# ── Decision log ────────────────────────────────────────────────────────────


def _cell(v):
    """CSV value: numpy scalars as Python numbers, floats rounded to 6 decimals."""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return "" if math.isnan(v) else round(v, 6)
    if isinstance(v, (np.integer, np.bool_)):
        return int(v)
    return v


class DecisionLog:
    """Decision log of one viewer session in a directory (default runs/viewer_logs/):
    <session>_steps.csv, a row per agent step (Session.step_row), and
    <session>_episodes.csv, a row per played episode with its oracle comparison
    (Session.episode_row). Files are created with the first row, flushed per row."""

    def __init__(self, directory: str, tag: str = ""):
        self.session = time.strftime("%Y%m%d-%H%M%S") + (f"_{tag}" if tag else "")
        self.paths = {k: os.path.join(directory, f"{self.session}_{k}.csv")
                      for k in ("steps", "episodes")}
        self._files: Dict[str, object] = {}
        self._writers: Dict[str, csv.DictWriter] = {}

    def write(self, kind: str, row: Dict[str, object]):
        w = self._writers.get(kind)
        if w is None:
            os.makedirs(os.path.dirname(self.paths[kind]), exist_ok=True)
            f = self._files[kind] = open(self.paths[kind], "w", newline="")
            w = self._writers[kind] = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
        w.writerow({k: _cell(v) for k, v in row.items()})
        self._files[kind].flush()

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()


# ── Session (simulator, scenario, env, agent; no GUI) ───────────────────────


class Session:
    """State behind the viewer: scan grid, atlas, agent, the scenario's env and episode."""

    def __init__(self, cfg: ViewerConfig, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.log = log
        self.device = dev = torch.device(cfg.device) if cfg.device else DEVICE
        self.geom = CellGeometry()
        self.acfg = AtlasConfig()
        self.ecfg = EnvConfig()
        self.limit = self.geom.m2_tilt_limit
        self.cap = self.acfg.cap
        self.sim = self.geom.make_sim(dev, max_bounces=self.cap)
        self.grid, self.W = box_grid(self.acfg.limit, self.acfg.step, dev)
        self.ladder_units = np.asarray(self.ecfg.step_ladder, dtype=np.int64)
        self.ladder = self.ladder_units * self.ecfg.step_unit  # deg
        self.hole_r = self.geom.m1_hole_radius
        self.roc0 = self.geom.roc  # Measured cell: the atlas and every run
        self._center = (self.W // 2) * self.W + self.W // 2  # Grid cell of zero M2 tilt
        # Training range of the loaded checkpoint (without one: the new scenario space)
        self.train_ranges: Dict[str, object] = {
            "separation": SEP_RANGE, "laser_pitch": ANGLE_RANGE, "laser_yaw": ANGLE_RANGE,
            "offset": "disk"}

        # ── Atlas (host copies: lookup of grid / named scenarios) ──
        self.atlas = None
        self.atlas_msg = "no atlas"
        if cfg.atlas and os.path.exists(cfg.atlas):
            try:
                at = load_atlas(cfg.atlas, torch.device("cpu"))
                at.check(self.cap, self.limit, self.ecfg.step_unit)
                self.atlas = at
                self._atlas_params = at.params.double().numpy()
                self.atlas_msg = (f"atlas {short_path(cfg.atlas)}: {at.n} scenarios, "
                                  f"{int(at.good.sum())} good")
            except (AssertionError, KeyError) as e:
                self.atlas_msg = f"atlas not used: {e}"
        self.log(self.atlas_msg)

        # ── Agent ──
        self.pcfg = PPOConfig()  # Env settings of the loaded run (defaults without one)
        self.render_style = "sim"  # RENDER_STYLES; a loaded checkpoint sets its own
        # Camera look: fixed parameters of every camera render (no per-episode draw): the
        # base (clean or a draw) with the panel's settings on top (look_params())
        self._look_base: Optional[Dict[str, torch.Tensor]] = None  # (1, ...) CameraParams
        self.look_name = "clean"  # Of the base: clean / draw <k>
        self.look_edited = False  # A setting changed since the base
        self.look_ctl: Dict[str, object] = {"camera": "mono", "scatter": False, "pose": True,
                                            "marks": True, "noise": "frozen"}
        self.look_ctl.update(face=0.0, laser=1.0, sigma=0.5, scatter_amp=4.5, exposure=1.0,
                             gamma=1.0, read_noise=0.0, speckle=0.0)
        self.look_draws = 0
        self.look_gen = torch.Generator(device=dev)
        self.look_gen.manual_seed(cfg.seed)
        self._look_cams: Dict[int, object] = {}  # img_res -> CameraRenderer (draws only)
        self.agent: Optional[Agent] = None
        self._act_fn = None  # agent.act or its GraphAct
        self.vnorm: Optional[ValueNorm] = None
        self.scen_head = False  # Scenario head loaded and matching its parameter list
        self.head_names: Tuple[str, ...] = ()
        self.head_mid = self.head_half = None  # (n,) device tensors
        self.ckpt_msg = self.ckpt_line = "none (simulator only)"
        self.gen = torch.Generator(device=dev)  # Sampled actions
        self.gen.manual_seed(cfg.seed * 1_000_003 + 17)
        self.rng = np.random.default_rng(cfg.seed)  # Random starts and scenarios
        self.sampled = False

        # ── Episode ──
        self.env: Optional[AlignEnv] = None
        self.scenario: Optional[ScenarioInfo] = None
        self.frame: Optional[Frame] = None
        self.start_tilt = (self.geom.m2_pitch, self.geom.m2_yaw)
        self.traj: List[np.ndarray] = []
        self.ring = None
        self._ring_next = None
        self._act = None  # (greedy, sampled) actions on the current frame
        self._first = torch.ones(1, dtype=torch.bool, device=dev)
        self.frame0: Optional[Frame] = None  # First frame of the episode

        # ── Decision log ──
        self.log_writer = DecisionLog(cfg.log_dir, cfg.log_tag) if cfg.log_dir else None
        if self.log_writer is not None:
            self.log(f"decision log {self.log_writer.paths['steps']}, "
                     f"{os.path.basename(self.log_writer.paths['episodes'])}")
        self.episode = 0  # Episode counter of the session
        self.last_result: Optional[Dict[str, object]] = None  # Last logged episode
        self._ep_open = False  # Current episode not logged yet
        self._ep_steps = 0
        self._ep_modes: set = set()
        self._table: Optional[MoveTable] = None
        self.ckpt_path = ""
        if cfg.ckpt:
            self.load_checkpoint(cfg.ckpt)

    # ── Agent ──

    def load_checkpoint(self, path: str) -> str:
        """Agent of a ppo.py checkpoint (best.pt / latest.pt): its AgentConfig, every tensor
        whose name and shape match (Agent.init_from; an old setup head is skipped, a
        missing scenario head stays untrained and is hidden), the value norm and the run's
        PPOConfig (env settings; render style camera for a --camera run, else sim; the
        camera look stays).
        Restarts the episode at the start tilt."""
        path = os.path.abspath(path)
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        with torch.inference_mode():
            self._finish_open()  # Logged with the agent that played it
        known = {f.name for f in fields(PPOConfig)}
        pcfg = PPOConfig(**{k: tuple(v) if isinstance(v, list) else v
                            for k, v in (ck.get("cfg") or {}).items() if k in known})
        n_bins = self.ladder_units.size
        acfg = (AgentConfig.from_dict(ck["agent_cfg"]) if ck.get("agent_cfg")
                else agent_config(pcfg, n_bins))
        assert acfg.n_bins == n_bins, f"agent n_bins {acfg.n_bins} != env ladder {n_bins}"
        vn = ValueNorm(pcfg.value_norm_beta, self.device)
        if ck.get("vnorm") is not None:
            vn.load_state_dict(ck["vnorm"])
        names, lo, hi = head_spec(ck.get("agent_cfg") or {})
        del ck
        agent = Agent(acfg)
        info = agent.init_from(path)
        agent.to(self.device).eval().requires_grad_(False)
        self.agent, self.vnorm, self.pcfg = agent, vn, pcfg
        self.render_style = "camera" if pcfg.camera else "sim"
        self.ckpt_path = path
        try:  # Outputs of the scenario head of this agent.py: mean and log-variance per value
            n_out = agent.scenario[-1].out_features // 2
        except (AttributeError, IndexError, TypeError):
            n_out = -1
        loaded = not any(k.startswith("scenario.") for k in info["missing"])
        self.scen_head = loaded and n_out == len(names)
        self.head_names = names
        lo_t, hi_t = (torch.tensor(v, dtype=torch.float32, device=self.device) for v in (lo, hi))
        self.head_mid, self.head_half = (hi_t + lo_t) / 2, (hi_t - lo_t) / 2
        rng = dict(zip(names, zip(lo, hi)))
        self.train_ranges = {k: rng.get(k, d) for k, d in (
            ("separation", SEP_RANGE), ("laser_pitch", ANGLE_RANGE), ("laser_yaw", ANGLE_RANGE))}
        self.train_ranges["offset"] = "disk" if "laser_dx" in names else "fixed"
        head = (f"scenario head: {', '.join(names)}" if self.scen_head else
                f"scenario head not loaded (checkpoint: {len(names)} values, this agent.py: "
                f"{n_out})")
        with torch.inference_mode():
            self.ring = agent.init_ring(1, self.device)
            self._act_fn = agent.act
            mode = "eager"
            if self.cfg.cuda_graph and self.device.type == "cuda":
                try:
                    self._act_fn = GraphAct(agent, self.device)
                    mode = "CUDA graph"
                except RuntimeError as e:  # Capture failed: stay eager
                    self.log(f"CUDA graph capture failed ({e}); agent.act runs eager")
        steps, it = info.get("env_steps") or 0, info.get("iteration")
        self.ckpt_line = (f"{run_tag(path, info.get('name'))} ({os.path.basename(path)}): "
                          f"iteration {it}, {steps / 1e6:.0f} M env steps")
        self.ckpt_msg = self.ckpt_line
        details = [f"checkpoint {path}",
                   f"  run {info.get('name')}, iteration {it}, {steps / 1e6:.1f} M env steps; "
                   f"act {mode}; input {acfg.encoder.in_ch} ch, render {self.render_style}",
                   f"  {head}", f"  training range: {self.range_text()}"]
        if info["skipped"]:
            details.append(f"  skipped tensors: {sorted(info['skipped'])}")
        self.log("\n".join(details))
        if self.scenario is not None:  # Env settings of the new run
            sc = self.scenario
            self.scenario.in_training_range, self.scenario.range_note = (
                self.training_range(sc.params))
            self._build_env(sc.params, torch.tensor([sc.path_ref]))
            self.start(self.start_tilt)
        return self.ckpt_msg

    # ── Scenario ──

    def full_params(self, params) -> Tuple[float, float, float, float, float]:
        """(separation, laser pitch, laser yaw, laser dx, laser dy); 3 values: the measured
        laser offset."""
        v = [float(x) for x in params]
        if len(v) == 3:
            v += [self.geom.laser_dx, self.geom.laser_dy]
        assert len(v) == 5, f"scenario needs 3 or 5 parameters, got {len(v)}"
        return tuple(v)

    def atlas_index(self, params) -> int:
        """Atlas row of the scenario (grid point or named setup, measured offset and ROC
        only), -1 if none."""
        sep, lp, ly, dx, dy = self.full_params(params)
        if self.atlas is None or abs(dx - self.geom.laser_dx) > 1e-6 or \
                abs(dy - self.geom.laser_dy) > 1e-6 or self.geom.roc != self.roc0:
            return -1
        d = np.abs(self._atlas_params - np.array([sep, lp, ly])[None]).max(1)
        i = int(d.argmin())
        return i if d[i] < 1e-4 else -1

    def _states(self, prm: torch.Tensor) -> torch.Tensor:
        """(n * G, 13) states of scenarios prm (n, 5) over the M2 box grid."""
        n, G = prm.shape[0], self.grid.shape[0]
        s = candidate_states(self.geom, prm[:, :3], self.grid).view(n, G, -1)
        s[..., S_LASER_DX] = prm[:, 3, None]
        s[..., S_LASER_DY] = prm[:, 4, None]
        return s.view(n * G, -1)

    def _scan(self, prm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Box scan of scenarios prm (n, 5) (one Triton call): bounces / path / stop (n, G),
        P_max (n,) and validity: clipped = 0 bounces and BLOCKED at zero M2 tilt (the
        laser hits the back of M1 at the hole edge, independent of the tilt), misses = no
        bounce at any tilt."""
        n = prm.shape[0]
        out = self.sim.simulate(self._states(prm), store_world=False, store_hits=False)
        b = out["hit_counts"].view(n, -1)
        p = out["total_path_length"].view(n, -1)
        st = out["stop_reason"].view(n, -1)
        c = self._center
        clipped = (b[:, c] == 0) & (st[:, c] == STOP_BLOCKED)
        misses = (b.amax(1) == 0) & ~clipped
        return {"b": b, "p": p, "clipped": clipped, "misses": misses}

    def invalid_reason(self, clipped: bool, misses: bool) -> str:
        return "clipped by hole" if clipped else ("misses M2" if misses else "")

    def range_text(self) -> str:
        """The loaded checkpoint's training range in words."""
        tr = self.train_ranges
        (a, b), (c, d), (e, f) = tr["separation"], tr["laser_pitch"], tr["laser_yaw"]
        laser = (f"{c:g} to {d:g} deg" if (c, d) == (e, f) else
                 f"pitch {c:g} to {d:g}, yaw {e:g} to {f:g} deg")
        offset = "in the hole" if tr["offset"] == "disk" else "fixed (measured)"
        return f"separation {a:g} to {b:g} mm, laser {laser}, offset {offset}"

    def training_range(self, params) -> Tuple[bool, str]:
        """Inside the loaded checkpoint's training range (self.train_ranges), and the
        violated ranges."""
        sep, lp, ly, dx, dy = self.full_params(params)
        tr, e = self.train_ranges, 1e-6
        viol = []
        lo, hi = tr["separation"]
        if not lo - e <= sep <= hi + e:
            viol.append(f"sep {lo:g} to {hi:g} mm")
        (plo, phi), (ylo, yhi) = tr["laser_pitch"], tr["laser_yaw"]
        if not (plo - e <= lp <= phi + e and ylo - e <= ly <= yhi + e):
            viol.append(f"laser {plo:g} to {phi:g} deg" if (plo, phi) == (ylo, yhi) else
                        f"laser pitch {plo:g} to {phi:g}, yaw {ylo:g} to {yhi:g} deg")
        if tr["offset"] == "fixed":
            if abs(dx - self.geom.laser_dx) > e or abs(dy - self.geom.laser_dy) > e:
                viol.append("fixed offset")
        elif math.hypot(dx, dy) > self.hole_r + e:
            viol.append("offset in the hole")
        if self.geom.roc != self.roc0:
            viol.append(f"ROC {self.roc0:g} mm")
        return not viol, ", ".join(viol)

    @torch.inference_mode()
    def random_scenario(self, seps: Tuple[float, float]) -> Tuple[Tuple[float, ...], int, str]:
        """A valid scenario with separation in seps (0.5 mm steps), laser pitch / yaw
        uniform in ANGLE_RANGE (0.01 deg) and the laser offset uniform in the hole disk
        (0.01 mm), preferring a good one: up to RANDOM_TRIES candidates, box-scanned in
        batches of RANDOM_BATCH, stopping at the first valid good one; else the valid one
        closest to good, else the last draw. Returns (params, draws, "good" / "valid" /
        "invalid")."""
        lo, hi = seps
        r_max = self.hole_r - 0.01  # Rounding to 0.01 mm stays inside the disk
        best, best_score, drawn, last = None, -1.0, 0, None
        while drawn < RANDOM_TRIES:
            n = min(RANDOM_BATCH, RANDOM_TRIES - drawn)
            sep = lo + 0.5 * self.rng.integers(0, int(round((hi - lo) / 0.5)) + 1, n)
            ang = np.round(self.rng.uniform(ANGLE_RANGE[0], ANGLE_RANGE[1], (n, 2)), 2)
            r = r_max * np.sqrt(self.rng.uniform(0.0, 1.0, n))
            th = self.rng.uniform(0.0, 2 * math.pi, n)
            off = np.round(np.stack((r * np.cos(th), r * np.sin(th)), 1), 2)
            prm = np.concatenate((sep[:, None], ang, off), 1)
            sc = self._scan(torch.tensor(prm, dtype=torch.float32, device=self.device))
            h = to_host({"pmax": sc["p"].amax(1), "ok": ~(sc["clipped"] | sc["misses"])})
            score = h["pmax"].astype(np.float64) / (self.acfg.floor_bounces * sep)  # >= 1: good
            ok = h["ok"] > 0.5
            good = np.nonzero(ok & (score >= 1.0))[0]
            if good.size:
                return tuple(float(v) for v in prm[good[0]]), drawn + int(good[0]) + 1, "good"
            if ok.any():
                i = int(np.where(ok, score, -1.0).argmax())
                if score[i] > best_score:
                    best, best_score = tuple(float(v) for v in prm[i]), float(score[i])
            last = tuple(float(v) for v in prm[-1])
            drawn += n
        return (best, drawn, "valid") if best is not None else (last, drawn, "invalid")

    @torch.inference_mode()
    def set_scenario(self, params, name: str = "") -> ScenarioInfo:
        """Scan the M2 box of the scenario (one Triton call, one host copy): P_max, best
        tilt, validity; P_max / best tilt / good from the atlas when it has the scenario
        (measured offset); build its env (while the scan runs: P_ref stays on the
        device)."""
        t0 = time.perf_counter()
        self._finish_open()
        prm5 = self.full_params(params)
        sep = prm5[0]
        pt = torch.tensor([prm5], dtype=torch.float32, device=self.device)
        scan = self._scan(pt)
        b, p = scan["b"][0], scan["p"][0]
        pmax, imax = p.max(0)
        idx = self.atlas_index(prm5)
        pf = self.acfg.floor_bounces * sep
        self._build_env(prm5, pmax.clamp(min=pf).view(1) if idx < 0 else
                        self.atlas.path_ref[idx].view(1))
        h = to_host({"pmax": pmax.view(1), "tilt": self.grid[imax], "bb": b[imax].view(1),
                     "clipped": scan["clipped"], "misses": scan["misses"], "b": b, "p": p})
        W = self.W
        reason = self.invalid_reason(h["clipped"][0] > 0.5, h["misses"][0] > 0.5)
        if idx >= 0:
            at = self.atlas
            pm, bt, bb = float(at.pmax[idx]), at.pmax_tilt[idx].tolist(), int(at.pmax_bounces[idx])
            pf, good, pref = float(at.p_floor[idx]), bool(at.good[idx]), float(at.path_ref[idx])
        else:
            pm, bt, bb = float(h["pmax"][0]), h["tilt"].tolist(), int(h["bb"][0])
            good, pref = pm >= pf, max(pm, pf)
        if not name and idx >= 0:
            name = next((k for k, v in self.atlas.named.items() if v == idx), "grid")
        inside, note = self.training_range(prm5)
        self.scenario = ScenarioInfo(
            name=name or "custom", params=prm5, atlas_index=idx, pmax=pm,
            best_tilt=(bt[0], bt[1]), best_bounces=bb, p_floor=pf, path_ref=pref,
            good=good and not reason, valid=not reason, invalid_reason=reason,
            in_training_range=inside, range_note=note,
            bounces=h["b"].reshape(W, W).astype(np.int16), path=h["p"].reshape(W, W),
            scan_ms=1000 * (time.perf_counter() - t0))
        return self.scenario

    def set_roc(self, roc: float):
        """Mirror ROC of both mirrors (scan, env, beam view); an open episode is logged
        first, the caller re-applies the scenario."""
        self._finish_open()
        self.geom = replace(self.geom, roc=float(roc))
        self.sim = self.geom.make_sim(self.device, max_bounces=self.cap)

    # ── Camera look ──

    def look_ranges(self):
        """render_camera.CameraRanges the new looks are drawn from: the run's training
        ranges (a --camera run), else the defaults without rotation; never clean."""
        from render_camera import CameraRanges

        r = (camera_ranges(self.pcfg, training=True) if self.pcfg.camera
             else CameraRanges(rotation_p=0.0))
        return replace(r, clean_p=0.0)

    def _look_cam(self, env: Optional[AlignEnv] = None):
        """A CameraRenderer of the run's image size (on the SpotRenderer of env, default
        the session's) for neutral / sampled parameters (not the env's camera: its
        sample_params returns the look)."""
        res = self.pcfg.img_res
        if res not in self._look_cams:
            from render_camera import CameraRenderer

            self._look_cams[res] = CameraRenderer((env or self.env).renderer,
                                                  self.look_ranges())
        return self._look_cams[res]

    def _set_look_base(self, p: Dict[str, torch.Tensor], name: str):
        """New base look; the panel settings are read back from it (the scatter amplitude,
        drawn per frame in training, stays)."""
        self._look_base, self.look_name, self.look_edited = p, name, False
        f = lambda k: float(p[k][0])
        mono = f("mono") > 0.5
        rgb = p["laser_rgb"][0]
        self.look_ctl.update(
            camera="mono" if mono else LOOK_CAMERAS[1 + int(rgb.argmax())],
            scatter=f("scatter_on") > 0.5, pose=True, marks=True,
            face=f("face"), laser=f("laser"), sigma=f("sigma"), exposure=f("exposure"),
            gamma=f("gamma"), read_noise=f("read_noise"), speckle=f("speckle"))

    def look_params(self, env: Optional[AlignEnv] = None) -> Dict[str, torch.Tensor]:
        """(1, ...) CameraParams of every camera render: the base look (clean until set)
        with the panel settings: camera (mono / laser color; a color look keeps its
        crosstalk, tint and white balance, from a mono base 0.1 crosstalk, neutral color),
        edge scatter on / off, pose off = identity, marks off = no rings, grain, engraving,
        scratches, dust, rim glow; noise off = no read / shot / row noise and speckle."""
        if self._look_base is None or self._look_base["prnu_map"].shape[-1] != self.pcfg.img_res:
            self._set_look_base(self._look_cam(env).neutral_params(1), "clean")
        c = self.look_ctl
        p = {k: v.clone() for k, v in self._look_base.items()}
        base_mono = float(p["mono"][0]) > 0.5
        if c["camera"] == "mono":
            qe = p["laser_rgb"].max(1, keepdim=True).values if base_mono else 1.0
            p["mono"].fill_(1.0)
            p["laser_rgb"] = torch.ones_like(p["laser_rgb"]) * qe
            for k in ("ambient_rgb", "wb"):
                p[k] = torch.ones_like(p[k])
        else:
            k = LOOK_CAMERAS.index(c["camera"]) - 1
            rgb = self._look_base["laser_rgb"]
            x = 0.1 if base_mono else float(rgb[0].sort().values[:2].mean())
            p["mono"].fill_(0.0)
            p["laser_rgb"] = torch.full_like(rgb, x)
            p["laser_rgb"][:, k] = 1.0
        p["scatter_on"].fill_(float(c["scatter"]))
        if not c["pose"]:
            eye = torch.eye(3, device=p["warp"].device)[None]
            p["warp"], p["warp_inv"] = eye.clone(), eye.clone()
            p["scale"].fill_(1.0)
        if not c["marks"]:
            for k in MARK_KEYS:
                p[k] = torch.zeros_like(p[k])
        for k in ("face", "laser", "sigma", "exposure", "gamma", "read_noise", "speckle"):
            p[k].fill_(float(c[k]))
        if c["noise"] == "off":
            for k in ("read_noise", "shot_noise", "row_noise", "speckle"):
                p[k].fill_(0.0)
        return p

    def _install_look(self, env: AlignEnv):
        """The look on an env's camera: its parameters for every reset (no per-episode
        draw) and now, the scatter amplitude fixed (drawn per frame in training)."""
        if env.cam is None:
            return
        p = self.look_params(env)
        env.cam.sample_params = lambda n, gen, ranges=None: {
            k: v.expand(n, *v.shape[1:]).clone() for k, v in p.items()}
        env.cam_params = env.cam.sample_params(env.num_envs, None)
        a = float(self.look_ctl["scatter_amp"])
        env.cam.ranges = replace(env.cam.ranges, scatter_amp=(a, a))

    def _reseed_camera(self):
        """Noise frozen / off: the camera generator restarts before every render, so the
        per-frame draws (sensor noise, speckle, edge-scatter shape) repeat and the same
        tilt gives the same frame; live: they go on (as in training)."""
        if self.env.cam is not None and self.look_ctl["noise"] != "live":
            self.env.reseed_camera(self.cfg.seed + self.look_draws)

    @torch.inference_mode()
    def set_look(self, **ctl) -> Optional[Frame]:
        """Change panel settings (look_ctl keys); in the camera style a new episode starts
        at the start tilt (an open one is logged), None in sim (kept for the camera)."""
        assert set(ctl) <= set(self.look_ctl), f"look settings {sorted(ctl)}"

        def update():
            self.look_ctl.update(ctl)
            self.look_edited = True

        return self._change_look(update)

    @torch.inference_mode()
    def new_look(self, clean: bool = False) -> Optional[Frame]:
        """Base look: clean (every effect off) or a new draw from look_ranges(); the panel
        settings follow it. A new episode as set_look."""
        cam = self._look_cam()

        def update():
            if clean:
                self._set_look_base(cam.neutral_params(1), "clean")
            else:
                self.look_draws += 1
                self._set_look_base(cam.sample_params(1, self.look_gen, self.look_ranges()),
                                    f"draw {self.look_draws}")

        return self._change_look(update)

    def _change_look(self, update: Callable[[], None]) -> Optional[Frame]:
        """Log an open episode under the old look, update, install, restart."""
        if self.env.cam is None:
            update()
            return None
        self._finish_open()
        update()
        self._install_look(self.env)
        return self.start(self.start_tilt)

    def look_text(self) -> str:
        """The look of the camera renders in one line ("" in sim): base, settings."""
        if self.render_style == "sim":
            return ""
        c = self.look_ctl
        on = lambda k: "on" if c[k] else "off"
        return (f"{self.look_name}{' edited' if self.look_edited else ''}: {c['camera']}, "
                f"face {c['face']:.2f}, laser {c['laser']:.2f}, sigma {c['sigma']:.2f}, "
                f"scatter {on('scatter')} {c['scatter_amp']:.1f}, pose {on('pose')}, "
                f"marks {on('marks')}, exposure {c['exposure']:.2f}, gamma {c['gamma']:.2f}, "
                f"read noise {c['read_noise']:.3f}, speckle {c['speckle']:.2f}, "
                f"noise {c['noise']}")

    @torch.inference_mode()
    def set_render_style(self, style: str) -> Frame:
        """Render style of the env images (RENDER_STYLES): an open episode is logged, the
        env rebuilt (camera: with the camera look), a new episode starts at the start
        tilt."""
        assert style in RENDER_STYLES, f"render style {style!r}"
        self._finish_open()
        self.render_style = style
        sc = self.scenario
        self._build_env(sc.params, torch.tensor([sc.path_ref]))
        return self.start(self.start_tilt)

    def agent_image(self, image: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) uint8 env image with the agent encoder's in_ch channels: 1 -> n
        copies the gray image (what the 3-channel encoders take for a mono image), n -> 1
        the channel mean (rounded); as is without an agent."""
        c = image.shape[1]
        n = c if self.agent is None else self.agent.cfg.encoder.in_ch
        if n == c:
            return image
        if c == 1:
            return image.expand(-1, n, -1, -1)
        assert n == 1, f"no channel adapter for {c} -> {n}"
        return image.float().mean(1, keepdim=True).round().to(torch.uint8)

    def input_text(self) -> str:
        """Render style and channels: the env's, -> the agent's when adapted (the camera
        look: look_text())."""
        c = self.env.img_shape[0]
        n = c if self.agent is None else self.agent.cfg.encoder.in_ch
        return f"{self.render_style}, {c} ch" + (f" -> {n}" if n != c else "")

    def make_env(self, params, path_ref: torch.Tensor) -> AlignEnv:
        """AlignEnv of the run's settings with one scenario (5 parameters): separation and
        laser tilt as its one-row table, the laser offset in its geometry; P_ref (1,);
        images of the render style (camera: the camera look on every reset)."""
        sep, lp, ly, dx, dy = self.full_params(params)
        ec = env_config(self.pcfg, 1, False, self.cfg.seed, setups="full")
        ec = replace(ec, camera=self.look_ranges() if self.render_style == "camera" else None,
                     geometry=replace(ec.geometry, laser_dx=dx, laser_dy=dy,
                                      roc=self.geom.roc))
        tab = Scenarios(params=torch.tensor([[sep, lp, ly]], dtype=torch.float32),
                        path_ref=path_ref.float().reshape(1))
        env = AlignEnv(ec, scenarios=tab)
        assert env.n_bins == self.ladder_units.size
        self._install_look(env)
        return env

    def _build_env(self, params, path_ref: torch.Tensor):
        old = self.env
        self.env = self.make_env(params, path_ref)
        if self.env.cam is not None and old is not None and old.cam is not None:
            # The camera generator goes on across rebuilds (scenario, checkpoint, style):
            # live noise does not repeat after a scenario change
            self.env.cam_gen.set_state(old.cam_gen.get_state())

    # ── Episode ──

    def random_start(self) -> Tuple[float, float]:
        """Initial tilt uniform in the run's init box (as AlignEnv draws it), clamped."""
        c, r = self.pcfg.init_center, self.pcfg.init_radius
        u = self.rng.uniform(-1.0, 1.0, 2)
        return tuple(float(np.clip(c[k] + r * u[k], -self.limit, self.limit)) for k in range(2))

    @torch.inference_mode()
    def start(self, tilt) -> Frame:
        """New episode at the given M2 tilt (clamped to the box): env reset, zero action,
        episode-start flag for the agent. An open episode is logged as abandoned."""
        self._finish_open()
        t = [float(np.clip(v, -self.limit, self.limit)) for v in tilt]
        self.start_tilt = (t[0], t[1])
        self._reseed_camera()
        self.env.reset(tilts=torch.tensor([t], dtype=torch.float32, device=self.device))
        self._first = torch.ones(1, dtype=torch.bool, device=self.device)
        self.frame = self.frame0 = self._observe()
        self.traj = [self.frame.tilt.copy()]
        self.episode += 1
        self._ep_open, self._ep_steps, self._ep_modes = True, 0, set()
        return self.frame

    @torch.inference_mode()
    def step(self) -> Optional[Frame]:
        """One env step with the action of the current frame's output (argmax or the
        sample, per self.sampled); None without an agent or after the episode ended. Logs
        the step, and the episode with its oracle comparison when it ends."""
        if self.agent is None or self.frame is None or self.frame.done:
            return None
        g, s = self._act
        a = s if self.sampled else g
        self.ring = self._ring_next  # Commit the window that produced the action (rollout)
        self._reseed_camera()
        _, rew, term, trunc, _ = self.env.step(a)
        self._first = torch.zeros(1, dtype=torch.bool, device=self.device)
        prev = self.frame
        self.frame = f = self._observe(a, term, trunc, rew)
        if f.terminated:  # Show the output that chose not move
            f.policy = prev.policy
        self.traj.append(f.tilt.copy())
        mode = "sampled" if self.sampled else "argmax"
        self._ep_steps += 1
        self._ep_modes.add(mode)
        if self.log_writer is not None:
            self.log_writer.write("steps", self.step_row(prev, f, mode))
        if f.done:
            self._finish_episode("stopped" if f.terminated else "truncated")
        return f

    def _observe(self, action=None, term=None, trunc=None, reward=None) -> Frame:
        """Frame of the env state: world hit points (one sim with store_world), agent
        output on the frame (act(); its ring is committed by step()), one host copy."""
        env = self.env
        sim = env.sim.simulate(env.states(env.tilt, env.setup), store_world=True)
        img = self.agent_image(env.image)
        items = {"tilt": env.tilt[0], "path": env.path[:1], "bounces": env.bounces[:1],
                 "stop": env.stop[:1], "step": env.step_count[:1], "image": img[0],
                 "applied": env.prev_action[0] * env.max_step, "count": sim["hit_counts"][:1],
                 "hits": sim["hit_sequence"][0], "mirror": sim["hit_mirror"][0],
                 "local": sim["hit_local_xy"][0], "ep_return": env.ep_return[:1],
                 "moves": env.ep_moves[:1]}
        if action is not None:
            items.update(action=action[0], term=term[:1], trunc=trunc[:1], reward=reward[:1])
        ag = self.agent
        if ag is not None:
            out, self._ring_next = self._act_fn(img, env.prev_action, self._first, self.ring)
            self._act = (greedy(out["logits"]), sample(out["logits"], self.gen))
            ts = ag.cfg.tilt_scale
            items.update(
                probs=torch.softmax(out["logits"][0].float(), -1),
                value=self.vnorm.denormalize(out["value"][:1]),
                path_head=out["path"][:1],
                tilt_mu=out["tilt_mu"][0] * ts,
                tilt_std=(0.5 * out["tilt_logvar"][0]).exp() * ts,
                act_g=self._act[0][0], act_s=self._act[1][0])
            if self.scen_head:  # De-normalized with the checkpoint's range
                items.update(scen_mu=out["scen_mu"][0] * self.head_half + self.head_mid,
                             scen_std=(0.5 * out["scen_logvar"][0]).exp() * self.head_half)
        h = to_host(items)
        n = int(h["count"][0])
        im = h["image"].astype(np.uint8)  # (C, H, W) -> gray (H, W) or RGB (H, W, 3)
        im = im[0] if im.shape[0] == 1 else im.transpose(1, 2, 0)
        f = Frame(step=int(h["step"][0]), tilt=h["tilt"].astype(np.float64),
                  path_mm=float(h["path"][0]), bounces=int(h["bounces"][0]),
                  stop=int(h["stop"][0]), image=im,
                  applied=h["applied"].astype(np.float64), hits=h["hits"][:n],
                  hit_mirror=h["mirror"][:n].astype(np.int64), hit_local=h["local"][:n],
                  ep_return=float(h["ep_return"][0]), moves=int(h["moves"][0]))
        if action is not None:
            f.action = h["action"].astype(np.int64)
            f.terminated, f.truncated = bool(h["term"][0]), bool(h["trunc"][0])
            f.reward = float(h["reward"][0])
        if ag is not None:
            pr = h["probs"].astype(np.float64)
            f.policy = PolicyView(
                probs=pr, p_stop=float(pr[0, 0] * pr[1, 0]), value=float(h["value"][0]),
                path_norm=float(h["path_head"][0]), tilt_mu=h["tilt_mu"], tilt_std=h["tilt_std"],
                scen_mu=h.get("scen_mu", np.zeros(0)), scen_std=h.get("scen_std", np.zeros(0)),
                act_greedy=h["act_g"].astype(np.int64), act_sample=h["act_s"].astype(np.int64))
        return f

    # ── Decision log and oracle ──

    def step_row(self, prev: Frame, f: Frame, mode: str) -> Dict[str, object]:
        """Log row of one step: the decision on frame prev (heads vs the true values of
        prev) and its result f."""
        p, sc = prev.policy, self.scenario
        sep, lp, ly, dx, dy = sc.params
        pref, pmax = sc.path_ref, sc.pmax
        a = f.action
        d = self.ladder[a]
        row: Dict[str, object] = {
            "session": self.log_writer.session if self.log_writer else "",
            "episode": self.episode, "step": f.step, "mode": mode, "scenario": sc.name,
            "sep_mm": sep, "laser_pitch_deg": lp, "laser_yaw_deg": ly,
            "laser_dx_mm": dx, "laser_dy_mm": dy, "roc_mm": self.geom.roc,
            "render": self.render_style, "look": self.look_text(),
            "pitch_before": prev.tilt[0], "yaw_before": prev.tilt[1],
            "pitch_after": f.tilt[0], "yaw_after": f.tilt[1],
            "act_pitch_deg": d[0], "act_yaw_deg": d[1],
            "applied_pitch_deg": f.applied[0], "applied_yaw_deg": f.applied[1],
            "not_move": int(not a.any()), "p_not_move": p.p_stop,
            "p_act_pitch": p.probs[0, a[0]], "p_act_yaw": p.probs[1, a[1]],
        }
        for k, axis in enumerate(("pitch", "yaw")):
            for r, i in enumerate(np.argsort(-p.probs[k], kind="stable")[:3], 1):
                row[f"{axis}_top{r}_deg"] = self.ladder[i]
                row[f"{axis}_top{r}_p"] = p.probs[k, i]
        row.update({
            "value": p.value,
            "path_head_m": p.path_norm * pref / 1000, "path_head_pref": p.path_norm,
            "path_m": prev.path_mm / 1000, "path_pref": prev.path_mm / pref,
            "path_pmax": ratio(prev.path_mm, pmax), "bounces": prev.bounces,
            "path_after_m": f.path_mm / 1000, "path_after_pref": f.path_mm / pref,
            "bounces_after": f.bounces, "stop_after": STOP_NAMES[f.stop],
            "reward": f.reward, "ep_return": f.ep_return,
            "tilt_head_pitch": p.tilt_mu[0], "tilt_head_pitch_std": p.tilt_std[0],
            "tilt_head_yaw": p.tilt_mu[1], "tilt_head_yaw_std": p.tilt_std[1],
        })
        for name, col in zip(PARAM_NAMES, PARAM_COLS):  # nan where the head has no value
            mu, sd = self.head_value(p, name)
            std = col[:-3] if col.endswith("_mm") else col
            row[f"scen_head_{col}"], row[f"scen_head_{std}_std"] = mu, sd
        row.update({"terminated": int(f.terminated), "truncated": int(f.truncated)})
        for k, axis in enumerate(("pitch", "yaw")):
            for i in np.argsort(self.ladder_units, kind="stable"):
                row[f"p_{axis}_{self.ladder_units[i]:+d}"] = p.probs[k, i]
        return row

    def head_value(self, p: PolicyView, name: str) -> Tuple[float, float]:
        """(mean, std) of the scenario head for a parameter, (nan, nan) if not predicted."""
        if not self.scen_head or name not in self.head_names:
            return math.nan, math.nan
        i = self.head_names.index(name)
        return float(p.scen_mu[i]), float(p.scen_std[i])

    def _move_table(self) -> MoveTable:
        if self._table is None:
            span = int(round(2 * self.env.limit / self.env.cfg.step_unit))
            self._table = MoveTable(ladder_units(self.env), span, self.device)
        return self._table

    def episode_row(self, end: str) -> Dict[str, object]:
        """Summary of the current episode (end: stopped / truncated / abandoned) with the
        oracle of its scenario and start tilt (baselines.plan_oracle on the episode's env:
        exact best clamp-free plan, value P / P_ref - step_cost x moves)."""
        sc, f, f0 = self.scenario, self.frame, self.frame0
        start = torch.tensor([self.start_tilt], dtype=torch.float32, device=self.device)
        plan = plan_oracle(self.env, start, chunk=1, table=self._move_table())
        h = to_host({"ret": plan.ret.float(), "tilt": plan.target_tilt[0], "moves": plan.moves,
                     "tp": plan.target_path_mm, "tb": plan.target_bounces,
                     "maxp": plan.max_path_mm})
        ret, oret = f.ep_return, float(h["ret"][0])
        modes = sorted(self._ep_modes)
        return {
            "session": self.log_writer.session if self.log_writer else "",
            "episode": self.episode, "end": end, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": short_path(self.ckpt_path) if self.agent is not None else "",
            "mode": modes[0] if len(modes) == 1 else "mixed",
            "scenario": sc.name, "atlas_index": sc.atlas_index, "sep_mm": sc.params[0],
            "laser_pitch_deg": sc.params[1], "laser_yaw_deg": sc.params[2],
            "laser_dx_mm": sc.params[3], "laser_dy_mm": sc.params[4], "roc_mm": self.geom.roc,
            "render": self.render_style, "look": self.look_text(),
            "pmax_m": sc.pmax / 1000, "pref_m": sc.path_ref / 1000, "good": int(sc.good),
            "valid": int(sc.valid), "invalid_reason": sc.invalid_reason,
            "in_training_range": int(sc.in_training_range), "range_note": sc.range_note,
            "best_pitch": sc.best_tilt[0], "best_yaw": sc.best_tilt[1],
            "start_pitch": f0.tilt[0], "start_yaw": f0.tilt[1],
            "start_path_m": f0.path_mm / 1000, "start_bounces": f0.bounces,
            "steps": f.step, "moves": f.moves, "final_pitch": f.tilt[0], "final_yaw": f.tilt[1],
            "final_path_m": f.path_mm / 1000, "final_bounces": f.bounces,
            "final_pmax": ratio(f.path_mm, sc.pmax), "final_pref": f.path_mm / sc.path_ref,
            "return": ret, "oracle_return": oret, "regret": oret - ret,
            "pct_oracle": 100.0 * ret / oret if abs(oret) > 1e-3 else math.nan,
            "oracle_target_pitch": h["tilt"][0], "oracle_target_yaw": h["tilt"][1],
            "oracle_target_path_m": h["tp"][0] / 1000, "oracle_target_bounces": int(h["tb"][0]),
            "oracle_moves": int(h["moves"][0]), "oracle_max_path_m": h["maxp"][0] / 1000,
        }

    def _finish_episode(self, end: str):
        """Log the episode summary, print it, keep it as last_result."""
        r = self.last_result = self.episode_row(end)
        self._ep_open = False
        if self.log_writer is not None:
            self.log_writer.write("episodes", r)
        pct = r["pct_oracle"]
        ext = ("" if r["in_training_range"] else ", extrapolation") + (
            f", invalid: {r['invalid_reason']}" if r["invalid_reason"] else "")
        self.log(f"episode {r['episode']} [{r['scenario']}, {r['mode']}{ext}]: start "
                 f"{r['start_pitch']:+.2f} / {r['start_yaw']:+.2f} -> {r['end']} after "
                 f"{r['steps']} steps ({r['moves']} moves) at {r['final_pitch']:+.2f} / "
                 f"{r['final_yaw']:+.2f}, path {r['final_path_m']:.2f} m = "
                 f"{r['final_pmax']:.3f} P_max; return {r['return']:+.4f}, oracle "
                 f"{r['oracle_return']:+.4f} ({r['oracle_moves']} moves to "
                 f"{r['oracle_target_pitch']:+.2f} / {r['oracle_target_yaw']:+.2f}): "
                 f"{pct:.1f} % of the oracle, regret {r['regret']:.4f}")

    def _finish_open(self):
        """An episode with steps that has not ended is logged as abandoned (new start,
        scenario, checkpoint or render style, window closed)."""
        if self._ep_open and self._ep_steps > 0:
            self._finish_episode("abandoned")

    @torch.inference_mode()
    def close(self):
        """Log an open episode and close the log files (window closed)."""
        self._finish_open()
        if self.log_writer is not None:
            self.log_writer.close()

    def next_action(self, f: Frame) -> Optional[np.ndarray]:
        """Ladder indices the agent takes (or took, on a terminated frame) at f."""
        if f.terminated:
            return f.action
        if f.policy is None or f.done:
            return None
        return f.policy.act_sample if self.sampled else f.policy.act_greedy


# ── Throttle ────────────────────────────────────────────────────────────────


class Throttle:
    """Rate limit for a callback fed by an event stream (slider or map drag): a call runs
    at once if the previous run started >= min_ms ago; otherwise its arguments replace the
    pending ones and a single-shot canvas timer runs them when the interval is over.
    flush() runs a pending call now (mouse release). __call__ returns True if it ran."""

    def __init__(self, canvas, fn: Callable, min_ms: float):
        self.fn = fn
        self.min_s = min_ms / 1000.0
        self.last = -math.inf
        self.pending: Optional[tuple] = None
        self.armed = False
        self.timer = canvas.new_timer(interval=max(1, int(min_ms)))
        self.timer.single_shot = True
        self.timer.add_callback(self.flush)

    def __call__(self, *args) -> bool:
        now = time.perf_counter()
        wait = self.min_s - (now - self.last)
        if wait <= 0:
            self.pending = None
            self._disarm()
            self.last = now
            self.fn(*args)
            return True
        self.pending = args
        if not self.armed:
            self.timer.interval = max(1, int(1000 * wait) + 1)
            self.timer.start()
            self.armed = True
        return False

    def _disarm(self):
        if self.armed:
            self.timer.stop()
            self.armed = False

    def flush(self):
        self._disarm()
        if self.pending is not None:
            args, self.pending = self.pending, None
            self.last = time.perf_counter()
            self.fn(*args)


# ── Viewer (GUI) ────────────────────────────────────────────────────────────


def exit_stub(pts: np.ndarray, d0: np.ndarray, last_mirror: int, stop: int,
              centers: Dict[int, np.ndarray], sep: float) -> Optional[np.ndarray]:
    """(2, 3) leaving ray after the last hit (hole exit or miss), reflected at the last
    hit's sphere (laser direction d0 without hits), up to 6 % of the separation beyond the
    mirror plane it heads for; None for the other stop reasons."""
    if stop not in (STOP_HOLE, STOP_MISSED):
        return None
    h = pts[-1]
    if len(pts) >= 2:
        d = pts[-1] - pts[-2]
        c = centers[last_mirror]
        nrm = (c - h) / np.linalg.norm(c - h)
        d = d - 2.0 * np.dot(d, nrm) * nrm
    else:
        d = d0
    if abs(d[2]) < 1e-9:
        return None
    z_end = -0.06 * sep if d[2] < 0 else 1.06 * sep
    t = (z_end - h[2]) / d[2]
    return np.stack((h, h + t * d)) if t > 0 else None


def _fast_textbox_class():
    """TextBox without full-figure redraws (built on first use: needs matplotlib.widgets).
    matplotlib's stop_typing() draws the whole figure on every click anywhere in it (also
    when not typing) and each keystroke does the same (_rendercursor); here a click outside
    redraws only when typing ends, typing(True / False) is called when typing starts / ends
    and the cursor update calls redraw() (a blit)."""
    from matplotlib.widgets import TextBox

    class FastTextBox(TextBox):
        def __init__(self, *args, redraw: Callable[[], None], typing: Callable[[bool], None],
                     **kw):
            self._redraw = redraw
            self._typing = typing
            super().__init__(*args, **kw)

        def begin_typing(self, *args):
            self._typing(True)
            super().begin_typing(*args)

        def stop_typing(self):
            if self.capturekeystrokes:
                self._typing(False)
                super().stop_typing()  # One full draw

        def _rendercursor(self):
            # TextBox._rendercursor (matplotlib 3.11) with redraw() instead of canvas.draw()
            r = self.ax.get_figure(root=True).canvas.get_renderer()
            text = self.text_disp.get_text()
            width_text = text[:self.cursor_index]
            bb = self.text_disp.get_window_extent(r)
            self.text_disp.set_text(width_text or ",")
            bw = self.text_disp.get_window_extent(r)
            if bb.y0 == bb.y1:
                bb.y0 -= bw.height / 2
                bb.y1 += bw.height / 2
            elif not width_text:
                bb.x1 = bb.x0
            else:
                bb.x1 = bb.x0 + bw.width
            self.cursor.set(segments=[[(bb.x1, bb.y0), (bb.x1, bb.y1)]], visible=True)
            self.text_disp.set_text(text)
            self._redraw()

    return FastTextBox


def _slider_box_class():
    """Value box of a slider (FastTextBox, built on first use): shows the slider's value
    text, the bare number while typing; Enter submits the typed text and stops typing,
    Escape or a click elsewhere stops typing without submitting. Text and cursor are
    animated (drawn with the slider's parts)."""
    FastTextBox = _fast_textbox_class()

    class SliderBox(FastTextBox):
        def __init__(self, ax, slider, **kw):
            self.slider = slider
            super().__init__(ax, "", initial=slider.valtext.get_text(),
                             typing=lambda _a: None, **kw)
            self.text_disp.set_animated(True)
            self.cursor.set_animated(True)
            slider.on_changed(lambda _v: self.show_val())  # Before the viewer's callback

        def show_val(self):
            if not self.capturekeystrokes:
                self.text_disp.set_text(self.slider.valtext.get_text())

        def begin_typing(self, *args):
            super().begin_typing(*args)
            self.text_disp.set_text(self.slider.valfmt.split()[0] % self.slider.val)

        def stop_typing(self):  # Click elsewhere
            self._end(submit=False)

        def _keypress(self, event):
            if self.capturekeystrokes and event.key in ("enter", "return", "escape"):
                self._end(submit=event.key != "escape")
            else:
                super()._keypress(event)

        def _end(self, submit: bool):
            if not self.capturekeystrokes:
                return
            text = self.text
            self._on_stop_typing()  # Shortcut keys back on
            self._on_stop_typing = None
            self.capturekeystrokes = False
            self.cursor.set_visible(False)
            self.show_val()
            self._redraw()
            if submit and self.eventson:
                self._observers.process("submit", text)

    return SliderBox


class Viewer:
    """Matplotlib GUI over a Session, see the module docstring.

    Blitting in two layers over the background of the last full draw (no animated
    artists): "scene" (landscape image, best tilt, colorbar labels, scenario text and
    sliders, toggle labels, message line, fixed beam parts; the checkpoint box while typing)
    is drawn and captured only when the scenario or a setting changes; "frame" (everything
    that follows the tilt) is drawn over that capture on every update and blitted in the
    FRAME_BLIT areas. Labels are static, values are FastText.
    """

    def __init__(self, session: Session, cfg: ViewerConfig):
        import matplotlib.pyplot as plt

        self.s = session
        self.cfg = cfg
        self.plt = plt
        for k, v in (("keymap.save", ["ctrl+s"]), ("keymap.home", ["home"]),
                     ("keymap.forward", ["v"]), ("keymap.back", ["c", "backspace"]),
                     ("keymap.grid", []), ("keymap.grid_minor", []), ("keymap.yscale", []),
                     ("keymap.xscale", []), ("keymap.fullscreen", ["ctrl+f"])):
            matplotlib.rcParams[k] = v  # Free s, r, right, g; no axis-scale keys
        # Remaining matplotlib Text (ticks, slider values, buttons): FreeType's autohinter
        # rasterizes 1.5x faster than the default native hinting (as FastText uses)
        matplotlib.rcParams["text.hinting"] = "force_autohint"
        self.fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi, facecolor=SURFACE)
        self.canvas = self.fig.canvas
        mgr = getattr(self.canvas, "manager", None)
        if mgr is not None:
            mgr.set_window_title("Herriott cell: simulator and policy viewer")
        self._layers: Dict[str, List] = {"scene": [], "frame": []}
        self._bg = None  # Background without animated artists
        self._bg_scene = None  # Background plus the scene layer
        self._regions: Dict = {}  # Slider -> (bbox, background of that region)
        self.map_mode = "bounces"
        self.running = False
        self.message = ""
        self._msg_dirty = False
        self._dragging_map = False
        self._dragging_hole = False
        self._build()
        c = self.canvas
        c.mpl_connect("draw_event", self._on_draw)
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("close_event", lambda _e: self.s.close())
        self.th_tilt = Throttle(c, self._apply_tilt, cfg.throttle_ms)
        self.th_scen = Throttle(c, self._apply_scenario, cfg.throttle_ms)
        self.th_look = Throttle(c, self._apply_look, cfg.throttle_ms)
        self._show_look()
        self.timer = c.new_timer(interval=cfg.run_interval_ms)
        self.timer.add_callback(self._on_tick)
        g = SETUPS[0]
        self._apply_scenario((g.separation, g.laser_pitch, g.laser_yaw), g.name,
                             (g.m2_pitch, g.m2_yaw))

    # ── Construction ──

    def _anim(self, artist, layer: str = "frame"):
        """Animated artist of a layer: left out of the background, drawn in order."""
        artist.set_animated(True)
        self._layers[layer].append(artist)
        return artist

    def _text(self, rect, text: str = "", layer: Optional[str] = "frame", x: float = 0.0,
              size: float = FS_TEXT, ax=None):
        """FastText at the top left (x in axes units) of an axis-less axes; layer None =
        static (background)."""
        if ax is None:
            ax = self.fig.add_axes(rect)
            ax.axis("off")
        t = FastText(ax, x, 1.0, text, size=size)
        if layer is not None:
            self._anim(t, layer)
        return t, ax

    def _build(self):
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.patches import Circle
        from matplotlib.widgets import Button, Slider

        fig, s, A = self.fig, self.s, self._anim
        cap, lim, W = s.cap, s.acfg.limit, s.W
        hw = s.acfg.step / 2
        R = s.geom.diameter / 2
        ticks = dict(labelsize=FS_TICK, colors=INK2)
        lab = dict(fontsize=FS_LABEL, color=INK2)
        tit = dict(fontsize=FS_TITLE, color=INK)
        self.cmap_b = LinearSegmentedColormap.from_list("bounces", BOUNCE_RAMP)
        cmap_hit = LinearSegmentedColormap.from_list("hits", HIT_RAMP)
        self._lut_b = np.round(self.cmap_b(np.linspace(0, 1, cap + 1)) * 255).astype(np.uint8)
        self._lut_p = np.round(self.cmap_b(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

        # ── Landscape ──
        ax = self.ax_map = fig.add_axes(MAP_RECT)
        ext = (-lim - hw, lim + hw, -lim - hw, lim + hw)
        self.im_map = ax.imshow(np.zeros((W, W, 4), np.uint8), origin="lower", extent=ext,
                                interpolation="nearest", aspect="equal")
        ax.set_xlim(ext[:2])
        ax.set_ylim(ext[2:])
        ax.set_xlabel("M2 yaw (deg)", **lab)
        ax.set_ylabel("M2 pitch (deg)", **lab)
        ax.tick_params(**ticks)
        self.ln_traj, = ax.plot([], [], "-", color=ORANGE, lw=1.5, marker=".", ms=5,
                                label="trajectory")
        self.ln_start, = ax.plot([], [], "o", ms=9, mfc=AQUA, mec="white", mew=1.2,
                                 label="start")
        self.ln_cur, = ax.plot([], [], "o", ms=10, mfc=ORANGE, mec=INK, mew=1.0, label="current")
        self.ln_best, = ax.plot([], [], "*", ms=16, mfc=YELLOW, mec=INK, mew=0.8,
                                label="best (P_max)")
        self.ln_head, = ax.plot([], [], "-", color=VIOLET, lw=1.8, label="tilt head +-1 std")
        self.ln_head_c, = ax.plot([], [], "x", color=VIOLET, ms=8, mew=1.8)
        self.ln_oracle, = ax.plot([], [], "D", ms=8, mfc="white", mec=INK, mew=1.4,
                                  label="oracle target (ended)")
        ax.legend(handles=[self.ln_traj, self.ln_start, self.ln_cur, self.ln_best, self.ln_head,
                           self.ln_oracle],
                  loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=FS_SMALL,
                  frameon=False, handlelength=1.6, columnspacing=1.2, labelcolor=INK2)
        ax.set_title("M2 tilt landscape", **tit)
        for a in (self.im_map, self.ln_best):
            A(a, "scene")
        for a in (self.ln_traj, self.ln_head, self.ln_head_c, self.ln_start, self.ln_cur,
                  self.ln_oracle):
            A(a)
        # Colorbar: static gradient and ticks at quarters; labels per map mode (scene)
        cb = fig.colorbar(ScalarMappable(Normalize(0, 1), self.cmap_b),
                          cax=fig.add_axes(CBAR_RECT))
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.tick_params(labelright=False, colors=INK2)
        cb.outline.set_edgecolor(GRID)
        self.cb_ticks = [A(cb.ax.text(2.2, q, "", transform=cb.ax.transAxes, va="center",
                                      fontsize=FS_TICK, color=INK2), "scene")
                         for q in (0, 0.25, 0.5, 0.75, 1.0)]
        self.cb_label = A(cb.ax.text(4.9, 0.5, "", transform=cb.ax.transAxes, va="center",
                                     ha="center", rotation=90, **lab), "scene")
        self.tx_scen, _ = self._text(SCEN_TXT_RECT, layer="scene")
        self.tx_last, _ = self._text(LAST_TXT_RECT)

        # ── Camera ──
        ax = self.ax_cam = fig.add_axes(CAM_RECT)
        res = s.pcfg.img_res
        self.im_cam = A(ax.imshow(np.zeros((res, res), np.uint8), cmap="gray", vmin=0, vmax=255,
                                  origin="upper", extent=(-R, R, -R, R),
                                  interpolation="nearest"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("camera: agent input (64 px)", **tit)
        _, tax = self._text(CAM_TXT_RECT, "step\nbounces\npath\nstop\ntilt (deg)\nlast move\n"
                            "input\nepisode", layer=None)
        self.tx_cam, _ = self._text(None, ax=tax, x=0.24)

        # ── Beam: side views and mirror faces ──
        self.side = []
        for k, (rect, axis, title) in enumerate(zip(
                SIDE_RECTS, ("x", "y"), ("beam, top view: x vs z / separation",
                                         "beam, side view: y vs z / separation"))):
            ax = fig.add_axes(rect)
            ax.set_xlim(-0.09, 1.09)
            ax.set_ylim(-1.12 * R, 1.12 * R)
            ax.set_ylabel(f"{axis} (mm)", **lab)
            ax.tick_params(**ticks)
            ax.set_title(title, **tit)
            if k == 1:
                ax.set_xlabel("z / separation (M1 at 0, M2 at 1; dashed: leaving ray)", **lab)
            else:
                ax.set_xticklabels([])
            ax.axhline(0, color=GRID, lw=0.6, zorder=0)
            art = {
                "m1": A(ax.plot([], [], "-", color=INK2, lw=3, solid_capstyle="butt")[0],
                        "scene"),
                "laser": A(ax.plot([], [], "o", ms=5, mfc=AQUA, mec=INK, mew=0.6)[0], "scene"),
            }
            if k == 1:  # M1 hole (y offset)
                art["hole"] = A(ax.plot([], [], "-", color=YELLOW, lw=3.5,
                                        solid_capstyle="butt")[0], "scene")
            art["beam"] = A(ax.plot([], [], "-", color=RED, lw=0.7, alpha=0.75)[0])
            art["stub"] = A(ax.plot([], [], "--", color=RED, lw=1.0)[0])
            art["m2"] = A(ax.plot([], [], "-", color=INK2, lw=3, solid_capstyle="butt")[0])
            self.side.append(art)
        self.faces = []
        g = s.geom
        for k, (rect, title) in enumerate(zip(FACE_RECTS, ("M1 face (laser side)",
                                                           "M2 face (as the camera)"))):
            ax = fig.add_axes(rect)
            ax.set_aspect("equal")
            ax.set_xlim(-1.1 * R, 1.1 * R)
            ax.set_ylim(-1.1 * R, 1.1 * R)
            ax.add_patch(Circle((0, 0), R, fill=False, ec=INK2, lw=1))
            if k == 0:
                ax.add_patch(Circle((0, g.m1_hole_offset_y), g.m1_hole_radius, fc=YELLOW,
                                    ec="none"))
            ax.tick_params(**ticks)
            ax.set_xlabel("local x (mm)", labelpad=1, **lab)
            if k == 0:
                ax.set_ylabel("local y (mm)", **lab)
            else:
                ax.tick_params(labelleft=False)  # Same scale as M1
            ax.set_title(title, **tit)
            sc = ax.scatter(np.zeros(0), np.zeros(0), c=np.zeros(0), s=18, cmap=cmap_hit,
                            vmin=0, vmax=cap, edgecolors="none")
            self.faces.append(A(sc))
        cb = fig.colorbar(ScalarMappable(Normalize(0, cap), cmap_hit),
                          cax=fig.add_axes(HIT_CBAR_RECT))
        cb.set_label("bounce index", **lab)
        cb.ax.tick_params(**ticks)
        cb.outline.set_edgecolor(GRID)

        # ── Policy: step probabilities and heads ──
        self.order = np.argsort(s.ladder_units, kind="stable")  # Display slot -> ladder index
        nb = self.order.size
        self._bar_x = np.arange(nb, dtype=np.float64)
        self.bars = []
        for rect, name in zip(BAR_RECTS, ("pitch", "yaw")):
            ax = fig.add_axes(rect)
            cols = [ORANGE if s.ladder_units[i] == 0 else BLUE for i in self.order]
            pc = PolyCollection(self._bar_verts(np.zeros(nb)), facecolors=cols, edgecolors=INK,
                                linewidths=0.0)
            ax.add_collection(pc, autolim=False)
            ax.set_xlim(-0.6, nb - 0.4)
            ax.set_ylim(0, 1)
            ax.set_xticks(self._bar_x)
            ax.set_xticklabels([f"{u:+d}" if u else "0" for u in s.ladder_units[self.order]],
                               rotation=90)
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(**ticks)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(f"P({name} step)", **tit)
            ax.set_xlabel("step (0.01 deg; orange = 0)", labelpad=1, **lab)
            self.bars.append(A(pc))
        _, pax = self._text(POL_RECT, "\nnext action\nP(not move)\nvalue (P_ref units)\n"
                            "path (m), P / P_ref\ntilt pitch (deg)\ntilt yaw (deg)\n"
                            "separation (mm)\nlaser pitch (deg)\nlaser yaw (deg)\n"
                            "laser dx (mm)\nlaser dy (mm)", layer=None)
        self._text(None, "agent (heads: mean +- std)", layer=None, x=0.27, ax=pax)
        self._text(None, "true", layer=None, x=0.7, ax=pax)
        self.tx_pol, _ = self._text(None, ax=pax, x=0.27)
        self.tx_true, _ = self._text(None, ax=pax, x=0.7)

        # ── Controls ──
        for x, y, t in ((30, 268, "scenario"), (30, 66, "M2 tilt (episode start)"),
                        (655, 154, "episode and policy")):
            fig.text(x / _W, y / _H, t, fontsize=FS_HEAD, color=INK, weight="bold")

        def button(x, y, w, label, cb, h=32):
            b = Button(fig.add_axes(_px(x, y, w, h)), label, color=BTN, hovercolor=BTN)
            b.drawon = False
            b.label.set_fontsize(FS_LABEL)
            b.label.set_color(INK)
            b.on_clicked(lambda _e: cb())
            return b

        SliderBox = _slider_box_class()
        self.boxes = {}  # Slider -> its value box

        def slider(y, label, lo, hi, v0, step, fmt, cb, layer, x=150, w=270, bw=88):
            ax = fig.add_axes(_px(x, y, w, 20))
            sl = Slider(ax, label, lo, hi, valinit=v0, valstep=step, valfmt=fmt,
                        initcolor="none", track_color=GRID, color=BLUE)
            sl.drawon = False
            sl.label.set_fontsize(FS_LABEL)
            sl.label.set_color(INK)
            sl.valtext.set_visible(False)  # Shown by the value box
            box = self.boxes[sl] = SliderBox(
                fig.add_axes(_px(x + w + 8, y, bw, 20)), sl, color="white", hovercolor="white",
                redraw=lambda sl=sl: self._blit_slider(sl))
            box.text_disp.set_fontsize(FS_LABEL)
            box.text_disp.set_color(INK)
            box.on_submit(lambda text, sl=sl: self._on_slider_text(sl, text))
            sl.on_changed(lambda _v, sl=sl: cb(sl))
            for a in self._slider_parts(sl):
                A(a, layer)
            return sl

        self.buttons = {}
        x = 30
        for g in SETUPS:
            w = 100 if g.name == "measured" else 85
            self.buttons[g.name] = button(x, 225, w, g.name, lambda n=g.name: self._on_named(n))
            x += w + 8
        self.buttons["roc"] = button(x, 225, 100, "", self._on_roc)
        x = 30
        for kind in ("good", "short", "long"):
            self.buttons[f"random {kind}"] = button(x, 186, 125, f"random {kind}",
                                                    lambda k=kind: self._on_random(k))
            x += 133
        g0 = SETUPS[0]
        rows = (("separation", SEP_RANGE, g0.separation, 0.5, "%.1f mm", 152),
                ("laser pitch", ANGLE_RANGE, g0.laser_pitch, 0.01, "%+.2f deg", 124),
                ("laser yaw", ANGLE_RANGE, g0.laser_yaw, 0.01, "%+.2f deg", 96))
        self.sl_sep, self.sl_lp, self.sl_ly = (
            slider(y, label, rng[0], rng[1], v0, st, fmt, self._on_scen_slider, "scene")
            for label, rng, v0, st, fmt, y in rows)
        L = s.limit
        self.sl_pitch = slider(36, "M2 pitch", -L, L, s.start_tilt[0], 0.01, "%+.2f deg",
                               self._on_tilt_slider, "frame")
        self.sl_yaw = slider(8, "M2 yaw", -L, L, s.start_tilt[1], 0.01, "%+.2f deg",
                             self._on_tilt_slider, "frame")

        # Laser offset in the M1 hole (click / drag inside the disk)
        ax = self.ax_hole = fig.add_axes(HOLE_RECT)
        hr = s.hole_r
        ax.set_aspect("equal")
        ax.set_xlim(-1.2 * hr, 1.2 * hr)
        ax.set_ylim(-1.2 * hr, 1.2 * hr)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(Circle((0, 0), hr, fc="#f2f1ed", ec=INK2, lw=1.2))
        ax.plot([g0.laser_dx], [g0.laser_dy], "+", color=INK2, ms=7, mew=1.2)  # Measured
        ax.set_title("laser offset", fontsize=FS_LABEL, color=INK, pad=3)
        self.ln_hole = A(ax.plot([], [], "o", ms=7, mfc=ORANGE, mec=INK, mew=0.8)[0], "scene")
        self.tx_hole, _ = self._text(HOLE_TXT_RECT, layer="scene", size=FS_SMALL)
        self.offset = (g0.laser_dx, g0.laser_dy)

        x = 655
        for name, w, cb in (("Step", 75, self._on_step), ("Run", 75, self._on_run),
                            ("Pause", 75, self._on_pause), ("Reset", 75, self._on_reset),
                            ("Random start", 130, self._on_random_start),
                            ("policy", 165, self._on_mode), ("map", 165, self._on_map_mode),
                            ("render", 200, self._on_render)):
            self.buttons[name] = button(x, 112, w, name, cb)
            x += w + 8

        # Camera look (right of the camera panel): base look buttons, toggles, sliders
        fig.text(968 / _W, 1038 / _H, "camera look", fontsize=FS_HEAD, color=INK,
                 weight="bold")
        self.tx_look, _ = self._text(_px(1090, 1030, 98, 16), layer="scene", size=FS_SMALL)
        for name, x, y, w, cb in (
                ("new look", 968, 1000, 110, lambda: self._on_new_look(False)),
                ("clean", 1084, 1000, 102, lambda: self._on_new_look(True)),
                ("camera", 968, 966, 110, lambda: self._on_look_cycle("camera", LOOK_CAMERAS)),
                ("noise", 1084, 966, 102, lambda: self._on_look_cycle("noise", LOOK_NOISE)),
                ("scatter", 968, 932, 110, lambda: self._on_look_toggle("scatter")),
                ("pose", 1084, 932, 102, lambda: self._on_look_toggle("pose")),
                ("marks", 968, 898, 110, lambda: self._on_look_toggle("marks"))):
            self.buttons[name] = button(x, y, w, name, cb, h=28)
        self.look_sliders = {}  # CameraParams key -> slider
        for i, (key, label, (lo, hi), st, fmt) in enumerate(LOOK_SLIDERS):
            self.look_sliders[key] = slider(866 - 26 * i, label, lo, hi,
                                            float(np.clip(s.look_ctl[key], lo, hi)), st, fmt,
                                            self._on_look_slider, "scene", x=1040, w=76, bw=62)
        for name in ("policy", "map", "roc", "render", "camera", "noise", "scatter", "pose",
                     "marks"):
            A(self.buttons[name].label, "scene")
        self._set_toggle_labels()
        # Checkpoint picker, right of the box: menu of best_checkpoints(), previous / next file
        self.buttons["list"] = button(1702, 69, 50, "list", self._on_list)
        self.buttons["prev"] = button(1756, 69, 50, "prev", lambda: self._on_pick(-1))
        self.buttons["next"] = button(1810, 69, 50, "next", lambda: self._on_pick(1))
        self.picked = self.s.ckpt_path  # Picker position: the last picked or loaded file
        self._menu = None  # The list menu (Tk), kept for the self-test
        ax = fig.add_axes(_px(770, 70, 926, 30))
        self.textbox = _fast_textbox_class()(
            ax, "checkpoint ", initial=short_path(self.s.cfg.ckpt), color="white",
            hovercolor="white", textalignment="left", redraw=lambda: self._blit(scene=True),
            typing=self._on_typing)
        self.textbox.label.set_fontsize(FS_LABEL)
        self.textbox.text_disp.set_fontsize(FS_LABEL)
        self.textbox.on_submit(self._on_load)
        self.tx_msg, _ = self._text(MSG_RECT, layer="scene")
        self.tx_ckpt, _ = self._text(CKPT_RECT, layer="scene")
        self._show_info()

    def _slider_parts(self, sl) -> list:
        """The artists of a slider that move: value bar, handle, value box text and cursor
        (the label, track and box stay in the background)."""
        box = self.boxes[sl]
        handle = getattr(sl, "_handle", None)
        return [a for a in (sl.poly, handle, box.text_disp, box.cursor) if a is not None]

    def _bar_verts(self, h: np.ndarray) -> List[np.ndarray]:
        x0, x1 = self._bar_x - 0.375, self._bar_x + 0.375
        z = np.zeros_like(h)
        return list(np.stack((np.stack((x0, z), 1), np.stack((x0, h), 1),
                              np.stack((x1, h), 1), np.stack((x1, z), 1)), 1))

    def _set_toggle_labels(self):
        self.buttons["policy"].label.set_text(
            "policy: sampled" if self.s.sampled else "policy: argmax")
        self.buttons["map"].label.set_text(
            "map: bounces" if self.map_mode == "bounces" else "map: P / P_max")
        self.buttons["roc"].label.set_text(f"ROC {self.s.geom.roc:g} mm")
        self.buttons["render"].label.set_text(f"render: {self.s.render_style}")
        c = self.s.look_ctl
        self.buttons["camera"].label.set_text(f"camera: {c['camera']}")
        self.buttons["noise"].label.set_text(f"noise: {c['noise']}")
        for k in ("scatter", "pose", "marks"):
            self.buttons[k].label.set_text(f"{k}: {'on' if c[k] else 'off'}")

    # ── Drawing ──

    def _on_draw(self, _event):
        """Full draw (startup, resize, toolbar): new backgrounds, then both layers."""
        c = self.canvas
        if c.is_saving():
            return
        self._bg = c.copy_from_bbox(self.fig.bbox)
        self._regions = {}
        for sl, box in self.boxes.items():
            a, b = sl.ax.bbox, box.ax.bbox
            bb = Bbox.from_extents(a.x0 - 8, min(a.y0, b.y0) - 6, b.x1 + 6,
                                   max(a.y1, b.y1) + 6)
            self._regions[sl] = (bb, c.copy_from_bbox(bb))
        self._draw_layer("scene")
        self._bg_scene = c.copy_from_bbox(self.fig.bbox)
        self._draw_layer("frame")

    def _draw_layer(self, layer: str):
        for a in self._layers[layer]:
            if a.get_visible():
                self.fig.draw_artist(a)

    def _blit(self, scene: bool = False):
        """Frame layer over the scene capture (scene: redraw and capture the scene layer
        first), blit the figure."""
        c = self.canvas
        if self._bg is None:
            c.draw()  # Calls _on_draw
            return
        if scene:
            c.restore_region(self._bg)
            self._draw_layer("scene")
            self._bg_scene = c.copy_from_bbox(self.fig.bbox)
        else:
            c.restore_region(self._bg_scene)
        self._draw_layer("frame")
        if scene:
            c.blit(self.fig.bbox)
        else:  # The frame layer lives in the panels and the M2 tilt sliders
            W, H = c.get_width_height(physical=True)
            for x0, y0, w, h in FRAME_BLIT:
                c.blit(Bbox.from_extents(x0 * W, y0 * H, (x0 + w) * W, (y0 + h) * H))

    def _blit_slider(self, sl):
        """Redraw one slider only (coalesced drag events)."""
        reg = self._regions.get(sl)
        if reg is None:
            return
        bb, bg = reg
        self.canvas.restore_region(bg)
        for a in self._slider_parts(sl):
            self.fig.draw_artist(a)
        self.canvas.blit(bb)

    # ── Artist updates ──

    def _show_scenario(self):
        """Scene-layer artists of the scenario (drawn by the next _show(scene=True))."""
        sc = self.s.scenario
        sep, lp, ly, dx, dy = sc.params
        for sl, v in ((self.sl_sep, sep), (self.sl_lp, lp), (self.sl_ly, ly)):
            self._set_slider(sl, v)
        self.offset = (dx, dy)
        self.ln_hole.set_data([dx], [dy])
        self.tx_hole.set_text(f"{dx:+.2f} / {dy:+.2f} mm")
        state = (("valid, good" if sc.good else "valid, impossible") if sc.valid else
                 sc.invalid_reason.upper())
        best = (f": P_max {sc.pmax / 1000:.2f} m ({sc.best_bounces} b) at "
                f"{sc.best_tilt[0]:+.2f}/{sc.best_tilt[1]:+.2f}" if sc.valid else "")
        self.tx_scen.set_text(
            f"{sc.name}: {sep:.1f} mm, laser {lp:+.2f}/{ly:+.2f} deg, offset "
            f"{dx:+.2f}/{dy:+.2f} mm\n"
            f"{state}{best}{'' if sc.in_training_range else '  EXTRAPOLATION'}")
        self.ln_best.set_data([sc.best_tilt[1]], [sc.best_tilt[0]])
        g, R = self.s.geom, self.s.geom.diameter / 2
        for k, art in enumerate(self.side):  # M1, laser origin, hole: fixed per scenario
            R1 = rotation(g.m1_pitch, g.m1_yaw)
            p1 = np.array([g.m1_tx, g.m1_ty, 0.0])
            u = np.zeros(3)
            u[k] = R
            e = np.stack((p1 - R1 @ u, p1 + R1 @ u))
            art["m1"].set_data(e[:, 2] / sep, e[:, k])
            o = R1 @ np.array([dx, g.m1_hole_offset_y + dy, -5.0]) + p1
            art["laser"].set_data([o[2] / sep], [o[k]])
            if "hole" in art:
                c = R1 @ np.array([0.0, g.m1_hole_offset_y, 0.0]) + p1
                art["hole"].set_data([c[2] / sep] * 2, [c[1] - g.m1_hole_radius,
                                                       c[1] + g.m1_hole_radius])
        self._show_map()

    def _show_map(self):
        sc = self.s.scenario
        if self.map_mode == "bounces":
            rgba = self._lut_b[np.clip(sc.bounces, 0, self.s.cap)]
            title = "bounces (cap 128)"
        else:
            q = np.rint(sc.path * (255.0 / max(sc.pmax, 1e-6))).clip(0, 255).astype(np.intp)
            rgba = self._lut_p[q]
            title = "path / P_max"
        self.im_map.set_data(rgba)
        labels = (("0", "32", "64", "96", "128") if self.map_mode == "bounces"
                  else ("0", "0.25", "0.5", "0.75", "1"))
        for t, lab in zip(self.cb_ticks, labels):
            t.set_text(lab)
        self.cb_label.set_text(title)

    def _show_info(self):
        """Checkpoint line (scene layer: drawn with the next scene blit)."""
        self.tx_ckpt.set_text(f"checkpoint {self.s.ckpt_line}")

    def _set_slider(self, sl, v: float):
        """Move a slider without its callback (and without a redraw)."""
        sl.eventson = False
        sl.set_val(v)
        sl.eventson = True
        self.boxes[sl].show_val()

    def _show(self, f: Frame, scene: bool = False):
        """Update the frame-layer artists in place and blit (scene: also redraw the
        scene layer)."""
        s = self.s
        tr = np.asarray(s.traj)
        self.ln_traj.set_data(tr[:, 1], tr[:, 0])
        self.ln_start.set_data([s.start_tilt[1]], [s.start_tilt[0]])
        self.ln_cur.set_data([f.tilt[1]], [f.tilt[0]])
        p = f.policy
        if p is not None:
            mu, sd = p.tilt_mu, p.tilt_std
            nan = np.nan
            self.ln_head.set_data([mu[1] - sd[1], mu[1] + sd[1], nan, mu[1], mu[1]],
                                  [mu[0], mu[0], nan, mu[0] - sd[0], mu[0] + sd[0]])
            self.ln_head_c.set_data([mu[1]], [mu[0]])
        else:
            self.ln_head.set_data([], [])
            self.ln_head_c.set_data([], [])
        r = s.last_result
        if r is not None and r["episode"] == s.episode:  # This episode ended: oracle target
            self.ln_oracle.set_data([r["oracle_target_yaw"]], [r["oracle_target_pitch"]])
        else:
            self.ln_oracle.set_data([], [])
        self.tx_last.set_text(self._last_text())
        self.im_cam.set_data(f.image)
        self.tx_cam.set_text(self._camera_text(f))
        self._show_beam(f)
        self._show_policy(f)
        self._set_slider(self.sl_pitch, f.tilt[0])
        self._set_slider(self.sl_yaw, f.tilt[1])
        if self._msg_dirty:
            self.tx_msg.set_text(self.message)
            self._msg_dirty, scene = False, True
        self._blit(scene)

    def _last_text(self) -> str:
        """Oracle comparison of the last logged episode."""
        r = self.s.last_result
        if r is None:
            return "last episode: none yet"
        pct = r["pct_oracle"]
        vs = (f"{pct:.1f} % of oracle" if not math.isnan(pct) else
              f"regret {r['regret']:+.3f} vs oracle")
        return (f"last: {r['end']}, {r['steps']} steps, {r['final_pmax']:.3f} P_max, {vs}")

    def _episode_state(self, f: Frame) -> str:
        if f.terminated:
            return f"stopped (not move) at step {f.step}"
        if f.truncated:
            return f"truncated at max_steps {f.step}"
        if self.s.agent is None:
            return "simulator only"
        return "running" if self.running else ("ready" if f.step == 0 else "paused")

    def _camera_text(self, f: Frame) -> str:
        s = self.s
        cap = " (cap)" if f.bounces >= s.cap else ""
        return (f"{f.step} / {s.env.cfg.max_steps}\n{f.bounces}{cap}\n"
                f"{f.path_mm / 1000:.3f} m = {ratio(f.path_mm, s.scenario.pmax):.3f} P_max\n"
                f"{STOP_NAMES[f.stop]}\n{f.tilt[0]:+.3f} / {f.tilt[1]:+.3f}\n"
                f"{f.applied[0]:+.2f} / {f.applied[1]:+.2f} deg\n{s.input_text()}\n"
                f"{self._episode_state(f)}")

    def _show_beam(self, f: Frame):
        s, g = self.s, self.s.geom
        sep, lp, ly, dx, dy = s.scenario.params
        R1 = rotation(g.m1_pitch, g.m1_yaw)
        R2 = rotation(f.tilt[0], f.tilt[1])
        p1 = np.array([g.m1_tx, g.m1_ty, 0.0])
        p2 = np.array([g.m2_tx, g.m2_ty, sep])
        origin = R1 @ np.array([dx, g.m1_hole_offset_y + dy, -5.0]) + p1
        d0 = R1 @ rotation(lp, ly) @ np.array([0.0, 0.0, 1.0])
        pts = np.vstack((origin[None], f.hits.astype(np.float64)))
        centers = {1: p1 + R1 @ np.array([0, 0, 1.0]) * g.roc,
                   2: p2 + R2 @ np.array([0, 0, -1.0]) * g.roc}
        last = int(f.hit_mirror[-1]) if f.hit_mirror.size else 0
        stub = exit_stub(pts, d0, last, f.stop, centers, sep)
        r = g.diameter / 2
        for k, art in enumerate(self.side):
            u = np.zeros(3)
            u[k] = r
            art["beam"].set_data(pts[:, 2] / sep, pts[:, k])
            if stub is None:
                art["stub"].set_data([], [])
            else:
                art["stub"].set_data(stub[:, 2] / sep, stub[:, k])
            e = np.stack((p2 - R2 @ u, p2 + R2 @ u))
            art["m2"].set_data(e[:, 2] / sep, e[:, k])
        for m, sc in zip((1, 2), self.faces):
            sel = f.hit_mirror == m
            sc.set_offsets(f.hit_local[sel] if sel.any() else np.zeros((0, 2)))
            sc.set_array(np.nonzero(sel)[0].astype(np.float64))

    def _show_policy(self, f: Frame):
        s = self.s
        p = f.policy
        act = s.next_action(f)
        for k, pc in enumerate(self.bars):
            h = np.zeros(self.order.size) if p is None else p.probs[k, self.order]
            pc.set_verts(self._bar_verts(h))
            pc.set_linewidths([1.6 if act is not None and act[k] == i else 0.0
                               for i in self.order])
        if p is None:
            self.tx_pol.set_text("\nno policy loaded (simulator only)")
            self.tx_true.set_text("")
            return
        mode = "sampled" if s.sampled else "argmax"
        if f.terminated:
            nxt = f"not move: stopped ({mode})"
        elif f.truncated:
            nxt = "none: max_steps reached"
        else:
            d = s.ladder[act]
            nxt = f"{d[0]:+.2f} / {d[1]:+.2f} deg ({mode}{', stop' if not act.any() else ''})"
        sc = s.scenario
        pref = sc.path_ref
        mu, sd = p.tilt_mu, p.tilt_std
        rows = ["", nxt, f"{p.p_stop:.3f}", f"{p.value:+.3f}",
                f"{p.path_norm * pref / 1000:.3f} = {p.path_norm:.3f} P_ref",
                f"{mu[0]:+.3f} +- {sd[0]:.3f}", f"{mu[1]:+.3f} +- {sd[1]:.3f}"]
        fmts = ("{:.1f} +- {:.1f}", "{:+.2f} +- {:.2f}", "{:+.2f} +- {:.2f}",
                "{:+.2f} +- {:.2f}", "{:+.2f} +- {:.2f}")
        for k, (name, fmt) in enumerate(zip(PARAM_NAMES, fmts)):
            m, e = s.head_value(p, name)
            rows.append(fmt.format(m, e) if not math.isnan(m) else
                        ("not predicted" if s.scen_head else
                         "scenario head not loaded" if k == 0 else ""))
        sep, lp, ly, dx, dy = sc.params
        self.tx_pol.set_text("\n".join(rows))
        self.tx_true.set_text(
            f"\n\n\n\n{f.path_mm / 1000:.3f} = {f.path_mm / pref:.3f}\n{f.tilt[0]:+.3f}\n"
            f"{f.tilt[1]:+.3f}\n{sep:.1f}\n{lp:+.2f}\n{ly:+.2f}\n{dx:+.2f}\n{dy:+.2f}")

    # ── Actions ──

    def _say(self, msg: str):
        """Message line (scene layer: set on events, not per frame)."""
        if msg != self.message:
            self.message = msg
            self._msg_dirty = True

    def _apply_scenario(self, params, name: str = "", nominal=None, note: str = ""):
        self._pause()
        self.s.set_scenario(params, name)
        f = self.s.start(nominal if nominal is not None else self.s.start_tilt)
        sc = self.s.scenario
        self._say(note or f"scenario {sc.name}; new episode at {f.tilt[0]:+.2f} / "
                  f"{f.tilt[1]:+.2f} deg")
        self._show_scenario()
        self._show(f, scene=True)

    def _apply_tilt(self, tilt):
        self._pause()
        self._show(self.s.start(tilt))

    def _on_named(self, name: str):
        g = setup_by_name(name)
        self._apply_scenario((g.separation, g.laser_pitch, g.laser_yaw), g.name,
                             (g.m2_pitch, g.m2_yaw))

    def _on_roc(self):
        """Next mirror ROC of ROCS: rescan the scenario, new episode at the start tilt."""
        s = self.s
        roc = ROCS[(ROCS.index(s.geom.roc) + 1) % len(ROCS)]
        self._pause()
        s.set_roc(roc)
        self._set_toggle_labels()
        self._apply_scenario(s.scenario.params, s.scenario.name, note=f"mirror ROC {roc:g} mm")

    def _on_render(self):
        """Next render style of RENDER_STYLES: env rebuilt, new episode at the start tilt
        (the first camera render of the process compiles the camera renderer: seconds)."""
        s = self.s
        style = RENDER_STYLES[(RENDER_STYLES.index(s.render_style) + 1) % len(RENDER_STYLES)]
        self._pause()
        t0 = time.perf_counter()
        f = s.set_render_style(style)
        self._set_toggle_labels()
        self._say(f"render {s.input_text()} ({time.perf_counter() - t0:.1f} s); new episode "
                  f"at {f.tilt[0]:+.2f} / {f.tilt[1]:+.2f} deg")
        self._show(f, scene=True)

    # ── Camera look ──

    def _show_look(self):
        """Look panel from the session: sliders (without callbacks), toggle labels, the
        base line (scene layer: drawn with the next scene blit)."""
        s = self.s
        for k, sl in self.look_sliders.items():
            self._set_slider(sl, float(np.clip(s.look_ctl[k], sl.valmin, sl.valmax)))
        self._set_toggle_labels()
        self.tx_look.set_text(s.look_name + (" edited" if s.look_edited else ""))

    def _look_done(self, f: Optional[Frame], what: str):
        s = self.s
        self._show_look()
        if f is None:
            self._say(f"camera look: {what} (render sim: shown in render camera)")
            f = s.frame
        else:
            self._say(f"camera look: {what}; new episode at {f.tilt[0]:+.2f} / "
                      f"{f.tilt[1]:+.2f} deg")
        self._show(f, scene=True)

    def _on_new_look(self, clean: bool):
        """Base look: clean, or a new draw from the run's training ranges."""
        self._pause()
        f = self.s.new_look(clean)
        self._look_done(f, "clean" if clean else f"new look, {self.s.look_name}")

    def _on_look_cycle(self, key: str, values: Tuple[str, ...]):
        c = self.s.look_ctl
        v = values[(values.index(c[key]) + 1) % len(values)]
        self._pause()
        self._look_done(self.s.set_look(**{key: v}), f"{key} {v}")

    def _on_look_toggle(self, key: str):
        v = not self.s.look_ctl[key]
        self._pause()
        self._look_done(self.s.set_look(**{key: v}), f"{key} {'on' if v else 'off'}")

    def _on_look_slider(self, sl):
        key = next(k for k, v in self.look_sliders.items() if v is sl)
        if not self.th_look(key):
            self._blit_slider(sl)

    def _apply_look(self, key: str):
        sl = self.look_sliders[key]
        self._pause()
        self._look_done(self.s.set_look(**{key: float(sl.val)}),
                        f"{sl.label.get_text()} {sl.valtext.get_text()}")

    def _on_random(self, kind: str):
        """Random valid scenario over the scenario space (good: separation 20-400 mm;
        short / long), preferring a good one (Session.random_scenario)."""
        seps = {"good": SEP_RANGE, "short": SHORT_SEPS, "long": LONG_SEPS}[kind]
        prm, k, status = self.s.random_scenario(seps)
        found = {"good": f"good (draw {k})",
                 "valid": f"valid, but no good one in {k} draws",
                 "invalid": f"no valid scenario in {k} draws"}[status]
        self._apply_scenario(prm, f"random {kind}", note=f"random {kind}: {found}")

    def _controls(self, offset=None) -> Tuple[float, ...]:
        """Scenario of the controls: sliders and the laser offset (or offset)."""
        dx, dy = self.offset if offset is None else offset
        return (self.sl_sep.val, self.sl_lp.val, self.sl_ly.val, dx, dy)

    def _nudge_sep(self, d: float):
        """Separation -/+ d mm from the keyboard (throttled like the slider)."""
        lo, hi = SEP_RANGE
        v = float(np.clip(round((self.sl_sep.val + d) * 2) / 2, lo, hi))
        self._set_slider(self.sl_sep, v)
        if not self.th_scen(self._controls()):
            self._blit_slider(self.sl_sep)

    def _on_scen_slider(self, sl):
        if not self.th_scen(self._controls()):
            self._blit_slider(sl)

    def _on_tilt_slider(self, sl):
        if not self.th_tilt((self.sl_pitch.val, self.sl_yaw.val)):
            self._blit_slider(sl)

    def _on_slider_text(self, sl, text: str):
        """A value typed into a slider's box (Enter): snapped to the slider step, clipped to
        its range and applied at once (as a click)."""
        try:
            v = round(float(text.split()[0]) / sl.valstep) * sl.valstep
        except (IndexError, ValueError, OverflowError):  # Empty, not a number, nan, inf
            self._say(f"not a number: {text!r}")
            self._show(self.s.frame, scene=True)
            return
        th = (self.th_tilt if sl in (self.sl_pitch, self.sl_yaw) else
              self.th_look if sl in self.look_sliders.values() else self.th_scen)
        th.last = -math.inf
        sl.set_val(round(float(np.clip(v, sl.valmin, sl.valmax)), 6))  # Callback applies

    def _on_step(self):
        self._pause()
        if self._can_step():
            self._show(self.s.step())

    def _can_step(self) -> bool:
        if self.s.agent is None:
            self._say("no policy loaded")
        elif self.s.frame.done:
            self._say("episode over")
        else:
            return True
        self._show(self.s.frame)
        return False

    def _on_run(self):
        if self.running or not self._can_step():
            return
        self.running = True
        self._show(self.s.frame)
        self.timer.start()

    def _on_pause(self):
        was = self.running
        self._pause()
        if was:
            self._show(self.s.frame)

    def _pause(self):
        if self.running:
            self.running = False
            self.timer.stop()

    def _on_tick(self):
        """Run timer: one policy step and redraw; stops at the end of the episode."""
        if not self.running:
            return
        f = self.s.step()
        if f is None or f.done:
            self._pause()
        if f is not None:
            self._show(f)

    def _on_reset(self):
        self._apply_tilt(self.s.start_tilt)

    def _on_random_start(self):
        self._apply_tilt(self.s.random_start())

    def _on_mode(self):
        self.s.sampled = not self.s.sampled
        self._set_toggle_labels()
        self._say(f"policy: {'sampled (Gumbel-max per axis)' if self.s.sampled else 'argmax'}")
        self._show(self.s.frame, scene=True)

    def _on_map_mode(self):
        self.map_mode = "path" if self.map_mode == "bounces" else "bounces"
        self._set_toggle_labels()
        self._show_map()
        self._show(self.s.frame, scene=True)

    def _on_load(self, text: str, note: str = ""):
        path = text.strip()
        if not path:
            return
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(RL_DIR, path)
        t0 = time.perf_counter()
        self._pause()
        try:
            self.s.load_checkpoint(path)
            if self.s.frame.step != 0:
                self.s.start(self.s.start_tilt)
            self.picked = self.s.ckpt_path
            self._say(f"loaded {short_path(path)}{note} in {time.perf_counter() - t0:.1f} s; "
                      f"render {self.s.render_style}")
        except Exception as e:  # Bad path or format: keep the current agent
            self._say(f"load failed: {type(e).__name__}: {e}")
        # Everything derived from the checkpoint: its line, the training-range marking
        # (scenario text), the render style, the heads (policy panel, redrawn with the frame)
        self._show_info()
        self._set_toggle_labels()
        self._show_scenario()
        self._show(self.s.frame, scene=True)

    def _on_pick(self, d: int):
        """Previous (d = -1) / next (+1) checkpoint of best_checkpoints(), rescanned on every
        click (newly copied files appear), from the last picked or loaded file (else the
        first / last); loaded as from the checkpoint box, which then shows its path."""
        paths = best_checkpoints()
        if not paths:
            self._say(f"no *best*.pt in {short_path(CKPT_ROOT)}/*/")
            self._show(self.s.frame, scene=True)
            return
        i = (paths.index(self.picked) + d if self.picked in paths else
             (0 if d > 0 else -1)) % len(paths)
        self._load_picked(paths, i)

    def _on_list(self):
        """Menu of best_checkpoints() (rescanned), a section per run directory, the loaded
        file marked; an entry loads as prev / next do. A native Tk menu: TkAgg only."""
        paths = best_checkpoints()
        widget = getattr(self.canvas, "get_tk_widget", None)
        if not paths or widget is None:
            self._say(f"no *best*.pt in {short_path(CKPT_ROOT)}/*/" if not paths else
                      "checkpoint list: TkAgg only (use prev / next)")
            self._show(self.s.frame, scene=True)
            return
        import tkinter as tk

        w = widget()
        if self._menu is not None:
            self._menu.destroy()
        self._menu = menu = tk.Menu(w, tearoff=0, font="TkFixedFont")
        self._menu_var = tk.StringVar(w, self.s.ckpt_path)  # Radio mark of the loaded file
        run = None
        for i, p in enumerate(paths):
            r, name = os.path.basename(os.path.dirname(p)), os.path.basename(p)
            if run is not None and r != run:
                menu.add_separator()
            menu.add_radiobutton(label=f"{r if r != run else '':8s} {name}",
                                 variable=self._menu_var, value=p,
                                 command=lambda i=i: self._load_picked(paths, i))
            run = r
        # Above the button (it is near the window bottom); Tk keeps the menu on screen
        b = self.buttons["list"].ax.bbox
        menu.update_idletasks()
        menu.tk_popup(w.winfo_rootx() + int(b.x0),
                      w.winfo_rooty() + int(w.winfo_height() - b.y1) - menu.winfo_reqheight())

    def _load_picked(self, paths: List[str], i: int):
        """Load paths[i] (picker) as from the checkpoint box, which then shows its path."""
        self.picked = path = paths[i]  # Also when it fails to load: the next click moves on
        tb = self.textbox
        tb.text_disp.set_text(short_path(path))  # No submit; drawn by the full draw below
        tb.cursor_index = len(tb.text)
        self._on_load(path, note=f" ({i + 1} / {len(paths)})")
        self.canvas.draw_idle()  # The box is in the background

    # ── Events ──

    def _map_event(self, event) -> bool:
        mode = getattr(getattr(self.canvas, "toolbar", None), "mode", "")
        busy = getattr(mode, "value", mode) != ""  # Pan / zoom active
        return event.inaxes is self.ax_map and event.xdata is not None and not busy

    @staticmethod
    def _map_tilt(event) -> Tuple[float, float]:
        return round(float(event.ydata), 2), round(float(event.xdata), 2)  # Mount resolution

    def _hole_offset(self, event) -> Optional[Tuple[float, float]]:
        """Laser offset (mm, 0.01 steps) at a click in the hole panel, kept inside the disk;
        None outside the panel."""
        if event.inaxes is not self.ax_hole or event.xdata is None:
            return None
        x, y = float(event.xdata), float(event.ydata)
        r, r_max = math.hypot(x, y), self.s.hole_r - 0.01  # Rounding stays inside
        if r > r_max:
            x, y = x * r_max / r, y * r_max / r
        return round(x, 2), round(y, 2)

    def _on_press(self, event):
        if event.button != 1:
            return
        if self._map_event(event):
            self._dragging_map = True
            self.th_tilt.last = -math.inf  # A click always applies at once
            self.th_tilt(self._map_tilt(event))
            return
        off = self._hole_offset(event)
        if off is not None:
            self._dragging_hole = True
            self.th_scen.last = -math.inf
            self.th_scen(self._controls(off))

    def _on_motion(self, event):
        if self._dragging_map and self._map_event(event):
            self.th_tilt(self._map_tilt(event))
        elif self._dragging_hole:
            off = self._hole_offset(event)
            if off is not None:
                self.th_scen(self._controls(off))

    def _on_release(self, _event):
        self._dragging_map = self._dragging_hole = False
        self.th_tilt.flush()
        self.th_scen.flush()
        self.th_look.flush()

    def _on_key(self, event):
        if (self.textbox.capturekeystrokes
                or any(b.capturekeystrokes for b in self.boxes.values())):
            return
        acts = {"right": self._on_step, "s": self._on_step, " ": self._toggle_run,
                "r": self._on_reset, "n": self._on_random_start, "d": self._on_mode,
                "m": self._on_map_mode, "g": lambda: self._on_random("good"),
                "[": lambda: self._nudge_sep(-0.5), "]": lambda: self._nudge_sep(0.5),
                "{": lambda: self._nudge_sep(-5.0), "}": lambda: self._nudge_sep(5.0)}
        fn = acts.get(event.key)
        if fn is not None:
            fn()

    def _on_typing(self, active: bool):
        """The checkpoint box joins the scene layer while typing (static otherwise)."""
        ax, layer = self.textbox.ax, self._layers["scene"]
        ax.set_animated(active)
        if active and ax not in layer:
            layer.append(ax)
        elif not active and ax in layer:
            layer.remove(ax)

    def _toggle_run(self):
        if self.running:
            self._on_pause()
        else:
            self._on_run()

    # ── Screenshot ──

    def screenshot(self, path: str):
        """The canvas as shown (background plus both layers) to a PNG."""
        import matplotlib.image as mimage

        self._blit()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mimage.imsave(path, np.asarray(self.canvas.buffer_rgba()))


# ── Self-test ───────────────────────────────────────────────────────────────


def _reference_episode(s: Session, tilt) -> Dict[str, object]:
    """The same episode through evaluation.evaluate with ppo.EvalPolicy (argmax) on a
    separate env of the session's settings and scenario, with the oracle reference as
    ppo.Trainer._oracle builds it (baselines.plan_oracle -> ref_return)."""
    from evaluation import evaluate
    from ppo import EvalPolicy

    sc = s.scenario
    env = s.make_env(sc.params, torch.tensor([sc.path_ref]))
    tilts = torch.tensor([tilt], dtype=torch.float32)
    plan = plan_oracle(env, tilts, chunk=8)
    ep, _ = evaluate(env, EvalPolicy(s.agent), tilts, ref_path_mm=plan.target_path_mm,
                     ref_return=plan.ret.float(), init_setups=torch.zeros(1, dtype=torch.long))
    ret, ref = float(ep["ep_return"][0]), float(ep["ref_return"][0])
    return {"final_path_mm": float(ep["final_path_mm"][0]), "length": int(ep["ep_length"][0]),
            "terminated": bool(ep["terminated"][0]),
            "final_tilt": ep["final_tilt"][0].cpu().numpy().astype(np.float64),
            "return": ret, "ref_return": ref, "regret": ref - ret}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def selftest(cfg: ViewerConfig, screenshot: str) -> int:
    """Drive the callbacks, play episodes and check them against evaluation.evaluate (same
    trajectory, same oracle reference), the decision log and the checkpoint box; 0 if
    every check passes."""
    from matplotlib.backend_bases import KeyEvent, MouseEvent

    interactive = matplotlib.get_backend().lower() != "agg"
    s = Session(replace(cfg, log_tag="selftest"))
    assert s.agent is not None, "--selftest needs a checkpoint"
    v = Viewer(s, cfg)
    plt = v.plt
    if interactive:
        plt.show(block=False)
        plt.pause(0.5)
    else:
        v.canvas.draw()
    flush = v.canvas.flush_events if interactive else (lambda: None)
    checks: Dict[str, bool] = {}

    def run_episode(label: str) -> Frame:
        v._on_run()
        while v.running:  # Interactive: the canvas timer steps; Agg: no timer, tick here
            if interactive:
                flush()
                time.sleep(0.001)
            else:
                v._on_tick()
        f = s.frame
        sc = s.scenario
        st = "stopped (not move)" if f.terminated else ("truncated" if f.truncated else "open")
        print(f"  {label}: start {s.start_tilt[0]:+.2f} / {s.start_tilt[1]:+.2f} -> final "
              f"{f.tilt[0]:+.2f} / {f.tilt[1]:+.2f} deg, {f.step} steps, {st}; path "
              f"{f.path_mm / 1000:.2f} m = {ratio(f.path_mm, sc.pmax):.3f} P_max "
              f"(P_max {sc.pmax / 1000:.2f} m, {f.bounces} bounces)")
        return f

    def mouse(name: str, x: float, y: float):
        e = MouseEvent(name, v.canvas, x, y, button=1)
        v.canvas.callbacks.process(e.name, e)
        flush()

    def click(ax=None, xy=None):
        if xy is None:
            xy = (ax.bbox.x0 + 0.5 * ax.bbox.width, ax.bbox.y0 + 0.5 * ax.bbox.height)
        mouse("button_press_event", *xy)
        mouse("button_release_event", *xy)

    def key(k: str):
        e = KeyEvent("key_press_event", v.canvas, k)
        v.canvas.callbacks.process(e.name, e)
        flush()

    def check_logged(f: Frame, tilt) -> bool:
        """Same episode as evaluation.evaluate, and the decision log holds it with the
        same return and oracle reference (ref_return, regret)."""
        ref = _reference_episode(s, tilt)
        same = (ref["length"] == f.step and ref["terminated"] == f.terminated
                and np.allclose(ref["final_tilt"], f.tilt, atol=1e-5)
                and abs(ref["final_path_mm"] - f.path_mm) < 1e-2)
        ep = _read_csv(s.log_writer.paths["episodes"])[-1]
        steps = [r for r in _read_csv(s.log_writer.paths["steps"])
                 if r["episode"] == ep["episode"]]
        oracle = all(abs(float(ep[k]) - ref[r]) < 1e-5 for k, r in (
            ("return", "return"), ("oracle_return", "ref_return"), ("regret", "regret")))
        logged = (int(ep["episode"]) == s.episode and int(ep["steps"]) == f.step
                  and len(steps) == f.step and [int(r["step"]) for r in steps]
                  == list(range(1, f.step + 1))
                  and abs(float(steps[0]["pitch_before"]) - tilt[0]) < 1e-5
                  and abs(float(steps[0]["yaw_before"]) - tilt[1]) < 1e-5
                  and int(steps[-1]["not_move"]) == int(f.terminated))
        print(f"    evaluation.evaluate, same start: {ref['length']} steps, path "
              f"{ref['final_path_mm'] / 1000:.2f} m, return {ref['return']:+.4f}, oracle "
              f"ref_return {ref['ref_return']:+.4f}, regret {ref['regret']:+.4f}: trajectory "
              f"{'identical' if same else 'DIFFERENT'}; log {len(steps)} step rows, oracle "
              f"numbers {'match' if oracle else 'DIFFER'}")
        return same and oracle and logged

    # Inputs: scenario buttons and sliders, tilt sliders, map click (throttled, flushed)
    for name in ("sep116", "sep144", "sep169"):
        v._on_named(name)
    for _ in range(3):
        v._on_random("good")
    v._on_named("measured")
    for i in range(10):  # Back-to-back drag events: coalesced, the last one applied
        v.sl_sep.set_val(80.0 + 0.5 * i)
    v._on_release(None)
    checks["scenario slider"] = abs(s.scenario.params[0] - 84.5) < 1e-6
    v._on_named("measured")
    for i in range(10):
        v.sl_pitch.set_val(-1.0 + 0.05 * i)
    v._on_release(None)
    checks["tilt slider"] = abs(s.start_tilt[0] + 0.55) < 1e-6
    click(xy=v.ax_map.transData.transform((0.5, 0.55)))  # (yaw, pitch)
    checks["map click"] = s.start_tilt == (0.55, 0.5)
    v._on_map_mode()
    v._on_map_mode()

    # Slider value boxes: typed value + Enter (snapped, clipped), not a number, Escape;
    # keys while typing must not reach the shortcuts
    def type_in(sl, text: str, end: str = "enter"):
        click(v.boxes[sl].ax)
        for k in ("end",) + ("backspace",) * 12 + tuple(text) + (end,):
            key(k)

    v._on_named("measured")
    type_in(v.sl_sep, "123.3")
    sep = s.scenario.params[0]
    type_in(v.sl_sep, "999")
    clip = s.scenario.params[0]
    type_in(v.sl_sep, "abc")
    bad = v.message.startswith("not a number") and s.scenario.params[0] == clip
    type_in(v.sl_pitch, "-0.37")
    pitch = s.start_tilt[0]
    type_in(v.sl_yaw, "srn", end="escape")
    checks["slider value boxes"] = (
        abs(sep - 123.5) < 1e-6 and clip == SEP_RANGE[1] and bad and abs(pitch + 0.37) < 1e-6
        and s.frame.step == 0 and s.start_tilt[0] == pitch
        and v.boxes[v.sl_yaw].text == v.sl_yaw.valtext.get_text())

    # Episodes: measured cell from the nominal tilt and from a blind start (argmax, checked
    # against evaluation.evaluate and the log), sampled, the other named setups
    print("episodes:")
    g = SETUPS[0]
    v._on_named("measured")
    f = run_episode("measured, nominal start, argmax")
    # Policy quality, not a viewer check (4407578 reaches the trapped 11.27 m and stops)
    print(f"    policy on the nominal start: {f.path_mm / s.scenario.pmax:.3f} P_max, "
          f"{'stopped' if f.terminated else 'truncated'} ({s.ckpt_line})")
    checks["nominal: evaluate and log"] = check_logged(f, (g.m2_pitch, g.m2_yaw))
    v.screenshot(screenshot)
    print(f"    screenshot {screenshot}")
    sc = s.scenario
    blind = np.argwhere(sc.bounces <= 2)
    i, j = blind[np.random.default_rng(cfg.seed).integers(len(blind))]
    lim, h = s.acfg.limit, s.acfg.step
    start = (-lim + h * i, -lim + h * j)
    v._apply_tilt(start)
    f = run_episode(f"measured, blind start ({sc.bounces[i, j]} bounces), argmax")
    checks["blind: evaluate and log"] = check_logged(f, start)
    v._on_mode()
    v._on_reset()
    run_episode("measured, blind start, sampled")
    v._on_named("measured")
    run_episode("measured, nominal start, sampled")
    checks["sampled episode logged"] = (s.last_result["mode"] == "sampled"
                                        and s.last_result["episode"] == s.episode)
    v._on_mode()
    for name in ("sep116", "sep144", "sep169"):
        v._on_named(name)
        run_episode(f"{name}, nominal start, argmax")
    v._on_named("measured")
    v._on_step()
    v._on_step()
    n = s.episode
    v._on_random_start()  # The open episode is logged as abandoned
    r = s.last_result
    checks["abandoned episode logged"] = (r["episode"] == n and r["end"] == "abandoned"
                                          and r["steps"] == 2)

    # Scenario space: random short / long (valid by construction), a random valid one with
    # negative laser angles and an offset, validity (clipped by the hole, misses M2)
    def label(sc: ScenarioInfo) -> str:
        sep, lp, ly, dx, dy = sc.params
        state = "good" if sc.good else ("valid" if sc.valid else sc.invalid_reason)
        return (f"{sep:.1f} mm, laser {lp:+.2f} / {ly:+.2f} deg, offset {dx:+.2f} / {dy:+.2f} "
                f"mm, {state}{'' if sc.in_training_range else ', extrapolation'}")

    def logged(sc: ScenarioInfo, f: Frame) -> bool:
        ep = _read_csv(s.log_writer.paths["episodes"])[-1]
        steps = [r for r in _read_csv(s.log_writer.paths["steps"])
                 if r["episode"] == ep["episode"]]
        offsets = (("laser_dx_mm", sc.params[3]), ("laser_dy_mm", sc.params[4]))
        return (f.done and int(ep["episode"]) == s.episode and len(steps) == f.step
                and ep["valid"] == str(int(sc.valid))
                and ep["invalid_reason"] == sc.invalid_reason
                and ep["in_training_range"] == str(int(sc.in_training_range))
                and all(abs(float(ep[k]) - val) < 1e-6 and abs(float(steps[-1][k]) - val) < 1e-6
                        for k, val in offsets)
                and ("EXTRAPOLATION" in v.tx_scen.get_text()) != sc.in_training_range)

    for kind, (lo, hi) in (("short", SHORT_SEPS), ("long", LONG_SEPS)):
        v._on_random(kind)
        sc = s.scenario
        f = run_episode(f"random {kind} ({label(sc)}), argmax")
        checks[f"random {kind}: valid, played and logged"] = (
            lo <= sc.params[0] <= hi and sc.valid and logged(sc, f))
    for _ in range(50):
        prm, _, status = s.random_scenario(SEP_RANGE)
        if status != "invalid" and prm[1] < 0 and prm[2] < 0 and math.hypot(*prm[3:]) > 0.3:
            break
    v._apply_scenario(prm, "random negative")
    sc = s.scenario
    f = run_episode(f"random ({label(sc)}), argmax")
    checks["negative angles and offset: played and logged"] = (
        sc.valid and sc.params[1] < 0 and sc.params[2] < 0 and logged(sc, f))
    v._apply_scenario((88.0, 0.0, 8.0, 1.45, 0.0))  # Pointing out of the hole edge
    clipped = s.scenario.invalid_reason == "clipped by hole"
    clipped &= "CLIPPED BY HOLE" in v.tx_scen.get_text()
    f = run_episode(f"clipped ({label(s.scenario)}), argmax")
    clipped &= logged(s.scenario, f)
    v._apply_scenario((400.0, 0.0, 8.0, 0.0, 0.0))  # 56 mm off-axis at M2: misses it
    f = run_episode(f"misses M2 ({label(s.scenario)}), argmax")
    misses = (s.scenario.invalid_reason == "misses M2" and s.scenario.pmax == 0.0
              and "MISSES M2" in v.tx_scen.get_text() and logged(s.scenario, f))
    checks["validity: clipped by hole / misses M2, played and logged"] = clipped and misses

    # Mirror ROC: 150 mm rescans (no atlas, extrapolation), plays and logs; 200 restores
    v._on_named("measured")
    sc200 = s.scenario
    v._on_roc()
    sc = s.scenario
    f = run_episode(f"ROC 150 mm, measured ({label(sc)}), argmax")
    ep = _read_csv(s.log_writer.paths["episodes"])[-1]
    roc150 = (s.geom.roc == 150.0 and s.env.cfg.geometry.roc == 150.0 and sc.atlas_index < 0
              and sc.pmax != sc200.pmax and not sc.in_training_range and logged(sc, f)
              and float(ep["roc_mm"]) == 150.0 and v.buttons["roc"].label.get_text() == "ROC 150 mm")
    v._on_roc()
    checks["mirror ROC 150 / 200"] = (roc150 and s.geom.roc == 200.0
                                      and s.scenario.atlas_index == sc200.atlas_index >= 0
                                      and s.scenario.pmax == sc200.pmax
                                      and s.scenario.in_training_range)

    # A five-parameter scenario head (synthesized checkpoint: this agent's weights, head
    # with 2 x 5 outputs, range over the scenario space): loads, plays and logs; its head
    # shows if this agent.py builds five-value heads, else it is reported as not loaded
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ck = torch.load(s.ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        sd = {k: v.clone() for k, v in ck["agent"].items()}
        for k in ("scenario.2.weight", "scenario.2.bias"):
            w = sd[k]
            sd[k] = torch.zeros((10,) + tuple(w.shape[1:]), dtype=w.dtype)
        acfg = dict(ck["agent_cfg"])
        r = s.hole_r
        acfg.update(scen_lo=(SEP_RANGE[0], ANGLE_RANGE[0], ANGLE_RANGE[0], -r, -r),
                    scen_hi=(SEP_RANGE[1], ANGLE_RANGE[1], ANGLE_RANGE[1], r, r),
                    scen_names=PARAM_NAMES)
        path5 = os.path.join(tmp, "best_5param.pt")
        torch.save({"agent": sd, "agent_cfg": acfg, "cfg": ck["cfg"], "vnorm": ck["vnorm"],
                    "name": "synthesized_5param", "iteration": 0, "env_steps": 0}, path5)
        del ck
        a_path = s.ckpt_path
        v._on_load(path5)
        v._on_random("good")
        sc = s.scenario
        f = run_episode(f"five-value head checkpoint ({label(sc)}), argmax")
        step = _read_csv(s.log_writer.paths["steps"])[-1]
        head_ok = (step["scen_head_laser_dx_mm"] != "") == s.scen_head
        checks["five-value head checkpoint: loads, plays, logs"] = (
            s.head_names == PARAM_NAMES and s.train_ranges["offset"] == "disk"
            and sc.in_training_range and logged(sc, f) and head_ok)
        print(f"    head names {s.head_names}; scenario head "
              f"{'loaded' if s.scen_head else 'not loaded with this agent.py (3-value head)'}")

        # Checkpoint switching: A, then B (another checkpoint, preferably with another
        # scenario head): the checkpoint line, the head rows and the training-range marking
        # must follow B (checked against B's own file), then back to A
        def n_head(path: str) -> int:
            c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            return len(head_spec(c.get("agent_cfg") or {})[0])

        others = sorted(os.path.abspath(q) for q in glob.glob(
            os.path.join(RL_DIR, "runs", "ppo", "*", "*best*.pt")) if os.path.abspath(q) != a_path)
        n_a = n_head(a_path)
        b_path = next((q for q in others if n_head(q) != n_a), others[0] if others else path5)

        def shows(path: str) -> bool:
            c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            it, steps = c.get("iteration"), c.get("env_steps") or 0
            names, lo, hi = head_spec(c.get("agent_cfg") or {})
            del c
            txt = v.tx_ckpt.get_text()
            rows = v.tx_pol.get_text().split("\n")
            dx_shown = len(rows) > 10 and rows[10] not in ("", "not predicted")
            sep_rng = dict(zip(names, zip(lo, hi))).get("separation", SEP_RANGE)
            return (os.path.basename(path) in txt and f"iteration {it}," in txt
                    and f"{steps / 1e6:.0f} M env steps" in txt and s.head_names == names
                    and dx_shown == ("laser_dx" in names and s.scen_head)
                    and s.train_ranges["separation"] == sep_rng
                    and (("EXTRAPOLATION" in v.tx_scen.get_text())
                         != s.training_range(s.scenario.params)[0]))

        v._on_load(a_path)
        a_ok, text_a = shows(a_path), v.tx_ckpt.get_text()
        v._on_load(b_path)
        b_ok, text_b = shows(b_path), v.tx_ckpt.get_text()
        checks["checkpoint A -> B: displayed info follows B"] = a_ok and b_ok and text_a != text_b
        print(f"    A: {text_a}\n    B: {text_b}")
        v._on_load(a_path)

    # Checkpoint box: click in, type (x, backspace), click outside (submit: load); bad path
    step0 = s.frame.step
    click(v.textbox.ax)
    for k in ("x", "backspace", "x", "backspace"):
        key(k)
    keys_ok = s.frame.step == step0  # Keys while typing must not reach the shortcuts
    click(v.ax_cam)  # Outside: stop typing, submit, load
    loaded = v.message.startswith("loaded")
    v._on_load("runs/ppo/missing/best.pt")
    checks["checkpoint box"] = (keys_ok and loaded and v.message.startswith("load failed")
                                and s.agent is not None)

    # Render styles: each style's images reach the agent with its encoder's channels (the
    # ring's frames) and the camera panel (gray / RGB); the cycle ends at the checkpoint's
    # default style
    print("render styles and camera look:")
    in_ch, res = s.agent.cfg.encoder.in_ch, s.pcfg.img_res
    shape = (res, res) if in_ch == 1 else (res, res, in_ch)
    default = "camera" if s.pcfg.camera else "sim"
    styles_ok = s.render_style == default
    for _ in RENDER_STYLES:
        v._on_render()
        v._on_step()
        f = s.frame
        styles_ok &= (s.env.img_shape[0] == (1 if s.render_style == "sim" else 3)
                      and f.step == 1 and f.image.shape == shape
                      and v.im_cam.get_array().shape == shape
                      and s.ring.frame.shape[2] == in_ch and s.render_style in v.message
                      and _read_csv(s.log_writer.paths["steps"])[-1]["render"] == s.render_style)
        print(f"  {s.input_text()}: agent input {tuple(s.ring.frame.shape[2:])}, camera panel "
              f"{f.image.shape}")
    checks["render styles: agent input and camera panel"] = (
        styles_ok and s.render_style == default)

    # Camera look: fixed over tilt changes and episodes (noise frozen: the same tilt gives
    # the same frame), live noise differs per frame, off repeats; the panel changes the
    # frame (lit face, laser color) and the log; clean restores the clean base
    if s.render_style != "camera":
        v._on_render()
    click(v.buttons["new look"].ax)
    drawn, t0 = s.look_name, s.start_tilt
    im0 = s.frame.image.copy()
    p0 = {k: t.clone() for k, t in s.env.cam_params.items()}
    v._apply_tilt((t0[0] + 0.05, t0[1] - 0.05))
    moved = s.frame.image.copy()
    v._apply_tilt(t0)
    fixed = (all(torch.equal(p0[k], t) for k, t in s.env.cam_params.items())
             and np.array_equal(s.frame.image, im0) and not np.array_equal(moved, im0))

    def two_episodes() -> bool:
        ims = []
        for _ in range(2):
            v._on_reset()
            ims.append(s.frame.image.copy())
        return np.array_equal(*ims)

    type_in(v.look_sliders["read_noise"], "0.02")
    click(v.buttons["noise"].ax)  # frozen -> live
    live = s.look_ctl["noise"] == "live" and not two_episodes()
    click(v.buttons["noise"].ax)  # live -> off
    off = s.look_ctl["noise"] == "off" and two_episodes()
    mean = lambda: s.env.image[0].float().mean((1, 2)).cpu().numpy()
    type_in(v.look_sliders["face"], "0")
    m0 = mean()
    type_in(v.look_sliders["face"], "0.6")
    face = s.look_ctl["face"] == 0.6 and mean().mean() > m0.mean() + 5
    type_in(v.look_sliders["face"], "0")
    while s.look_ctl["camera"] != "red":
        click(v.buttons["camera"].ax)
    m = mean()
    red = bool(m[0] > m[1] and m[0] > m[2]) and v.buttons["camera"].label.get_text() == "camera: red"
    v._on_step()
    look = _read_csv(s.log_writer.paths["steps"])[-1]["look"]
    logged = look.startswith(f"{drawn} edited: red") and "noise off" in look
    click(v.buttons["clean"].ax)
    clean = (s.look_name == "clean" and not s.look_edited and s.look_ctl["camera"] == "mono"
             and s.look_ctl["face"] == 0.0 and v.tx_look.get_text() == "clean")
    while s.look_ctl["noise"] != "frozen":
        click(v.buttons["noise"].ax)
    print(f"  {drawn}: fixed over a tilt change and back {fixed}, live noise differs {live}, "
          f"off repeats {off}, lit face {face}, red laser {red}, logged {logged}, clean "
          f"{clean}")
    checks["camera look: fixed, noise modes, controls, log, clean"] = (
        fixed and live and off and face and red and logged and clean)
    while s.render_style != default:
        v._on_render()

    # Checkpoint picker: next / prev over best_checkpoints() (from the loaded file), loaded
    # through the box path (the box shows the file, the render style follows the run)
    paths, a_path = best_checkpoints(), s.ckpt_path
    nxt = paths[(paths.index(a_path) + 1) % len(paths) if a_path in paths else 0]
    v._on_pick(1)
    v._on_step()
    next_ok = (s.ckpt_path == nxt and v.textbox.text == short_path(nxt)
               and v.message.startswith("loaded") and s.frame.step == 1
               and s.render_style == ("camera" if s.pcfg.camera else "sim")
               and v.buttons["render"].label.get_text() == f"render: {s.render_style}")
    print(f"  picker next: {v.message}")
    v._on_pick(-1)
    prev_ok = (s.ckpt_path == (a_path if a_path in paths else paths[-1])
               and v.textbox.text == short_path(s.ckpt_path))
    print(f"  picker prev: {v.message}")
    checks["checkpoint picker: next / prev"] = next_ok and prev_ok

    # List menu (TkAgg: the entry of a file loads it; Agg: no menu, the message says so)
    v._on_list()
    if interactive:
        m = v._menu
        m.unpost()
        j = next(j for j in range(m.index("end") + 1) if m.type(j) == "radiobutton"
                 and m.entrycget(j, "value") == paths[0])
        m.invoke(j)
        list_ok = s.ckpt_path == paths[0] and v.textbox.text == short_path(paths[0])
        print(f"  picker list: {v.message}")
    else:
        list_ok = v.message.startswith("checkpoint list: TkAgg only")
    checks["checkpoint picker: list"] = list_ok

    # Simulator only: tilt and scenario changes work, Step and Run report the missing policy
    s0 = Session(replace(cfg, ckpt="", log_dir=""), log=lambda m: None)
    v0 = Viewer(s0, cfg)
    v0.canvas.draw()
    v0._apply_tilt((0.09, 1.03))
    v0._on_step()
    v0._on_run()
    sim_ok = (s0.frame.step == 0 and s0.frame.bounces == s0.cap and not v0.running
              and v0.message == "no policy loaded")
    v0._on_named("sep144")
    v0._on_map_mode()
    checks["simulator only"] = sim_ok and s0.scenario.name == "sep144" and s0.frame.policy is None
    plt.close(v0.fig)

    s.close()
    plt.close(v.fig)
    print(f"decision log: {s.log_writer.paths['steps']}\n              "
          f"{s.log_writer.paths['episodes']}")
    for k, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {k}")
    ok = all(checks.values())
    print("selftest " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    d = ViewerConfig()
    ap = argparse.ArgumentParser(
        description="Interactive Herriott-cell simulator and policy viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=(
            "mouse: click or drag on the landscape sets the start tilt (new episode); click or\n"
            "  drag in the laser-offset disk sets laser dx / dy; Enter in the checkpoint box\n"
            "  loads it; list (menu) / prev / next pick a runs/ppo/*/*best*.pt; the box right\n"
            "  of each slider takes a typed value (Enter applies, Escape cancels)\n"
            "render button: sim / camera images for the agent and the camera panel (adapted\n"
            "  to the agent's input channels); the camera look panel sets the camera render\n"
            "  (fixed over tilt changes and episodes)\n"
            "keys: right or s step, space run / pause, r reset (start tilt), n random start,\n"
            "  d argmax / sampled, m map color, g random good scenario, [ / ] separation\n"
            "  -/+0.5 mm, { / } -/+5 mm\n"
            "every played episode is logged (steps and oracle comparison) under --log-dir;\n"
            "the console shows the atlas and log paths at start and the checkpoint details\n"
            "(path, run name, act mode, input channels, render style, scenario head,\n"
            "training range) on every load"))
    ap.add_argument("--ckpt", default=d.ckpt,
                    help="ppo.py checkpoint (default: the newest runs/ppo/*/*best*.pt)")
    ap.add_argument("--no-ckpt", action="store_true", help="Simulator only")
    ap.add_argument("--atlas", default=d.atlas, help="Scenario atlas (scenarios.py)")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--no-cuda-graph", action="store_true", help="agent.act() eager")
    ap.add_argument("--throttle-ms", type=float, default=d.throttle_ms)
    ap.add_argument("--run-interval-ms", type=int, default=d.run_interval_ms)
    ap.add_argument("--figsize", type=float, nargs=2, default=list(d.figsize))
    ap.add_argument("--dpi", type=float, default=d.dpi, help="Scales the whole UI")
    ap.add_argument("--log-dir", default=d.log_dir, help="Decision log directory")
    ap.add_argument("--no-log", action="store_true", help="No decision log")
    ap.add_argument("--backend", default="", help="Matplotlib backend (default TkAgg; Agg "
                                                  "for --selftest)")
    ap.add_argument("--selftest", action="store_true", help="Headless functional test")
    ap.add_argument("--screenshot", default=os.path.join(FIG_DIR, "viewer.png"),
                    help="--selftest: PNG of the UI after an episode")
    args = ap.parse_args(argv)
    cfg = ViewerConfig(ckpt="" if args.no_ckpt else args.ckpt, atlas=args.atlas,
                       device=args.device, seed=args.seed, cuda_graph=not args.no_cuda_graph,
                       throttle_ms=args.throttle_ms,
                       run_interval_ms=args.run_interval_ms, figsize=tuple(args.figsize),
                       dpi=args.dpi, log_dir="" if args.no_log else args.log_dir)
    backend = args.backend or ("Agg" if args.selftest else "TkAgg")
    if backend.lower() != "agg" and not (os.environ.get("DISPLAY")
                                         or os.environ.get("WAYLAND_DISPLAY")):
        print("no display (DISPLAY / WAYLAND_DISPLAY unset); use --selftest for a headless run")
        return 1
    matplotlib.use(backend)
    if args.selftest:
        return selftest(cfg, args.screenshot)
    import matplotlib.pyplot as plt

    s = Session(cfg)
    Viewer(s, cfg)
    plt.show()
    s.close()  # Also on close_event; idempotent
    return 0


if __name__ == "__main__":
    sys.exit(main())
