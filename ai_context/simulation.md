# Optical simulation

Two unrelated simulators exist in this repo:

1. **Zemax OpticStudio** (`data_process/generate_simulated_data.py`) - produced the feasibility
   dataset for the paper. Windows + ZOS-API only. See [data_sources.md](data_sources.md).
   Model files in `simulation/zemax/multireflection-1in.*`.
2. **`simulation/herriott_sim.py`** - a batched PyTorch ray tracer written for the RL work.
   This file is what the rest of this document covers.

## `HerriottSim` - `simulation/herriott_sim.py:80`

Fully tensorized, no Python loop over the batch, differentiable, GPU by default.

**State: `(B, 13)`** with module-level index constants (`S_M1_PITCH` .. `S_LASER_YAW`,
`STATE_DIM = 13`):

| idx | name | meaning |
|---|---|---|
| 0,1 | m1_pitch, m1_yaw | mirror 1 tilt (deg) |
| 2,3 | m2_pitch, m2_yaw | mirror 2 tilt (deg) |
| 4 | separation | mirror spacing along Z (mm) |
| 5-8 | m1_tx/ty, m2_tx/ty | transverse mirror offsets (mm) |
| 9-12 | laser_dx/dy, laser_pitch/yaw | laser offset/angle error from ideal (mm/deg) |

The laser is always mounted on M1 and enters through its hole; DOFs 9-12 are *errors*, not
controls. Build states with `make_state(...)` (`:473`), get a default sim from `create_sim()` (`:512`).

**Geometry** - `MirrorConfig` (`:52`): `roc=150`, `diameter=24.4`, `hole_radius=1.8`,
`hole_offset_y=7.0`, `reflectivity=0.98`. Two values carry `FIXME: for sim2real` comments -
they deviate from the physical 25.4 mm / 1.5 mm to compensate for model error.
`SimConfig` (`:70`): `max_bounces=100` (commented alternative 50 for training),
`intensity_threshold=1e-4`.

**`simulate(state) -> dict`** (`:239`): unrolls `max_bounces` reflection steps
(ray-sphere intersect `:176`, aperture/hole check `:207`, mirror reflection, intensity
attenuation by reflectivity, deactivate when intensity drops below threshold). Returns
`hit_counts (B,)`, `final_positions (B,3)`, `total_path_length (B,)`,
`hit_sequence (B, max_bounces, 3)`, `intensity (B,)`.

`reward()` (`:457`) is a stray example reward (distance to a target bounce count + path bonus);
the RL env computes its own, see [rl.md](rl.md).

## Parameter and stability search - `simulation/stability_search.py`

Offline design study, independent of RL. Batched brute force over `BATCH_SIZE = 180_000` states,
`MIN_PATH_MM = 4000`.

- **Stage 1** (`run_stage1`, `:175`): for each mirror separation, sweep all
  (laser_pitch x laser_yaw) combos; the bounce threshold is `ceil(MIN_PATH_MM / separation)`.
  Records max bounces, all winning angle pairs, a flood-filled contiguous winning region
  (`flood_fill_nd`, `:98`) and per-axis tolerances (`contiguous_range_1d`, `:118`).
  `--symmetrical` halves the yaw sweep by mirroring around 0.
  Outputs `results/stage1.json`, `results/stage1_maps.npz`.
- **Stage 2** (`run_stage2`, `:356`): mechanical robustness. For each Stage-1 winner, build a 5D
  disturbance grid (sep_err x laser_dx x laser_dy x m2_tx x m2_ty); at every point, tile all
  m2_pitch x m2_yaw compensator combos and keep the best bounce count. Reports the contiguous
  stable region, marginal tolerances per axis, and a bounce histogram.
  Outputs `results/stage2_summary.json`, `results/stage2_grids_sep{N}.npz`.

`simulation/search_results_viz.py` renders those artifacts: laser-angle maps, scorecard,
marginals, 2D slices, bounce histograms, tolerance/path tradeoff. Figures land in
`graphs/Simulation graphs/`.

## Interactive viewers

- `simulation/interactive_sim.py` / `interactive_sim_exp.py` - matplotlib sliders over the sim
  DOFs with a live 3D layout and mirror-face spot pattern. `_exp` is the experimental fork.
- `interactive_policy_trial.py` (repo root) - same viewer plus a trained RL policy: "Run Policy",
  "Step", "Reset", "Load Ckpt". See [rl.md](rl.md) for the caveat about it being out of sync
  with the current sim API.
