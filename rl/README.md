# RL alignment of a Herriott cell

Goal: an agent that tilts one mirror of a Herriott cell until the optical path inside the
cavity is maximal, using only camera images of the spot pattern. Trained entirely in
simulation; transfer to the real cell comes later.

## Setup

- **Simulator** - `herriott_sim.py` (RL copy of `simulation/herriott_sim.py`), a batched
  PyTorch ray tracer with 13 DOF
  (tilt and transverse offset of both mirrors, separation, laser offset and angle). The
  simulator, the image renderer, the policy and the PPO update all run on the GPU; nothing
  crosses to the host inside the training loop.
- **Cell, first stage** - the measured geometry: ROC 200 mm, 25.4 mm mirrors, 1.5 mm coupling
  hole 7 mm off-axis in M1 (laser side), separation 88 mm, laser pitch/yaw 4.08/4.82 deg,
  laser offset 0.75/1.04 mm (values from `tools/zemax_herriott_build.py`). The nominal
  alignment gives 31 bounces (16 spots on M2) and a 2.73 m path, then exits through the hole.
- **Controlled mirror** - M2, the one without the laser window: pitch and yaw. Separation,
  laser tilt and M1 stay fixed in the first stage.
- **Observation** - a simulated camera image of the spot pattern on the M2 face (the mirror
  without the laser window). A CNN encodes each frame;
  the policy and the value function both see a stack of the last K encodings together with
  the actions that led to them.
- **Actions** - discrete tilt steps per axis, `{0, +-1, +-3, +-10, +-30, +-100} x 0.01 deg`.
  (0, 0) means "not move" and ends the episode: the agent decides it has reached the
  longest path.
- **Reward** - optical path length before the beam leaves the cavity, capped at 128 bounces
  (beams that never leave are scored at the cap). Per move: change in path, normalized by
  the nominal 2.73 m, minus a small step cost; the return is final minus initial path.
- **Algorithm** - PPO.
- **Path-length head** - a second head on the same encoder and stack predicts the current
  light path length from the images (labels from the sim). It is an auxiliary loss during
  training and a camera-only path estimate on the real cell.

## Reward landscape at the first-stage geometry

M2 tilt scanned +-3 deg around nominal on a 0.01 deg grid (361k configurations, ~2 s on
an RTX A2000):

| bounces | path (m) | how the beam leaves | region size (deg) |
|---|---|---|---|
| 1-3 | <0.3 | misses a mirror | 77 % of the scan |
| 31 | 2.73 | M1 hole | 3.5 x 2.5 plateau containing nominal |
| 57 | 5.02 | M1 hole | 1.1 x 1.2 island |
| 89 | 7.84 | M1 hole | 0.24 x 0.23 island |
| cap (400) | 35 at cap | never leaves | 0.20 x 0.16, next to the 89 island |

The plateaus are nested: the higher-order re-entrant patterns sit inside smaller and
smaller regions, and at their centre the beam never finds the hole and runs until the
bounce cap. Reaching the long paths needs steps of about 0.02 deg, while crossing the
plateau takes about 1 deg; the real mount resolves 0.01 deg.

## Stages

1. Single geometry, random initial M2 tilt (this is where we start).
2. Wider initial range; jitter of separation, laser tilt and offsets (domain randomization).
3. M1 as the controlled mirror.
4. Sim-to-real: camera model fitted to the real setup, encoder fine-tuned on real images.

## Future extensions

- **Model-based control for real-world deployment.** On the real cell every interaction is
  slow and expensive, which is where model-based methods pay off. Two routes:
  - a learned world model (DreamerV3 / TD-MPC2 style) pretrained in simulation and then
    fine-tuned on a small amount of real-cell data, with the policy trained or planned in
    its latent space;
  - the known physics: estimate the hidden state (M2 tilt, separation, laser tilt) from the
    image history, then search the simulator for the action that maximizes path. Depends
    on the simulator matching the real cell closely, since the target regions are ~0.2 deg wide.
- **Mirror substrate and edge scattering.** The simulator treats each mirror as an
  infinitely thin spherical cap: a ray outside the aperture just misses, and a back-side
  hit is detected only within the cap. The real mirrors have glass thickness, edges and
  bevels; a beam landing on an edge scatters light that shows up in the camera image.
  Modelling the substrate geometry and that scattering is needed for sim-to-real.

Plan and code review: [PLAN.md](PLAN.md).
