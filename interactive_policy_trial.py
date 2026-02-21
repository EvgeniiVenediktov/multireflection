"""
Interactive Herriott Cell visualization with trained policy integration.
Uses herriott_sim.py as backend and policy.py for the trained agent.

Controls:
  - Sliders: manually set M1/M2 pitch/yaw and separation
  - "Run Policy": auto-step the policy with animation
  - "Step":       single policy step
  - "Reset":      reset policy hidden state and prev_action
  - "Load Ckpt":  load a policy checkpoint file
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from simulation.herriott_sim import create_sim, DEVICE
from policy import RecurrentPolicy
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
DIAMETER = 25.4
HOLE_DIAMETER = 3.0
HOLE_OFFSET = 7.0

# Policy action scaling
ACTION_SCALE = 0.5  # max degrees per step

# Observation rendering
OBS_SIZE = 64
OBS_EXTENT = 15.0  # mm, half-width of the rendered region
SPOT_SIGMA = 0.5


class InteractiveHerriottCell:
    def __init__(self, checkpoint_path: str = None):
        self.sim = create_sim(mounted_laser=True)

        # ── Policy state ──────────────────────────────────────────────
        self.policy = None
        self.hidden = None
        self.prev_action = None
        self.step_count = 0
        self.running = False
        self.timer = None

        if checkpoint_path:
            self._load_policy(checkpoint_path)

        self.setup_plot()

    # ── Policy management ─────────────────────────────────────────────

    def _load_policy(self, path: str):
        """Load a trained policy checkpoint (matches train.py CheckpointManager format)."""
        self.policy = RecurrentPolicy(hidden_dim=128, feat_dim=128).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        # train.py CheckpointManager saves:
        #   { "iteration", "policy_state_dict", "optimizer_state_dict", "sampler_state_dict", "extra" }
        if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
            sd = ckpt["policy_state_dict"]
            print(f"[INFO] Checkpoint from iteration {ckpt.get('iteration', '?')}")
        elif isinstance(ckpt, dict) and any(k.startswith("encoder.") for k in ckpt):
            # Raw state_dict saved directly
            sd = ckpt
        else:
            raise ValueError(
                f"Unrecognized checkpoint format. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )

        self.policy.load_state_dict(sd)
        self.policy.eval()
        self._reset_policy_state()
        print(f"[INFO] Policy loaded from {path}")

    def _reset_policy_state(self):
        """Reset GRU hidden state and prev_action."""
        self.hidden = torch.zeros(1, self.policy.hidden_dim, device=DEVICE)
        self.prev_action = torch.zeros(1, 2, device=DEVICE)
        self.step_count = 0

    # ── Observation rendering ─────────────────────────────────────────

    def _render_obs(self, hits) -> torch.Tensor:
        """
        Render current spot pattern as a (1, 1, OBS_SIZE, OBS_SIZE) tensor
        for the policy's encoder. Single-channel Gaussian spots on M1.
        """
        x = np.linspace(-OBS_EXTENT, OBS_EXTENT, OBS_SIZE)
        X, Y = np.meshgrid(x, x)
        img = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)

        m1_hits = [
            (h["point"][0], h["point"][1], 0.98 ** (i + 1))
            for i, h in enumerate(hits)
            if h["mirror"] == "M1"
        ]
        for px, py, intensity in m1_hits:
            img += intensity * np.exp(
                -((X - px) ** 2 + (Y - py) ** 2) / (2 * SPOT_SIGMA**2)
            )

        # Clip and normalize to [0, 1]
        img = np.clip(img, 0, None)
        if img.max() > 0:
            img /= img.max()

        return (
            torch.tensor(img, device=DEVICE, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    # ── Policy stepping ───────────────────────────────────────────────

    def _policy_step(self):
        """Execute one policy step: observe → act → apply → redraw."""
        if self.policy is None:
            print("[WARN] No policy loaded.")
            return

        # 1. Get current sim state and hits
        state = self._current_state_tensor()
        result = self.sim.simulate(state)
        hits = self._result_to_hits(result)

        # 2. Render observation
        obs = self._render_obs(hits)

        # 3. Policy forward pass
        with torch.no_grad():
            action, log_prob, value, h_new = self.policy.act(
                obs, self.prev_action, self.hidden
            )

        # 4. Scale and apply action (adjust M2 pitch/yaw)
        delta_pitch = action[0, 0].item() * ACTION_SCALE
        delta_yaw = action[0, 1].item() * ACTION_SCALE

        new_m2_pitch = np.clip(self.s_m2_pitch.val + delta_pitch, -5, 5)
        new_m2_yaw = np.clip(self.s_m2_yaw.val + delta_yaw, -5, 5)

        # Update sliders (this triggers self.update via on_changed)
        self.s_m2_pitch.set_val(new_m2_pitch)
        self.s_m2_yaw.set_val(new_m2_yaw)

        # 5. Update policy state
        self.hidden = h_new
        self.prev_action = action
        self.step_count += 1

        # 6. Update info display
        self._update_info(action, log_prob, value)

    def _update_info(self, action, log_prob, value):
        """Update the policy info text on the figure."""
        self.info_text.set_text(
            f"Step: {self.step_count}  |  "
            f"Action: [{action[0,0]:.3f}, {action[0,1]:.3f}]  |  "
            f"V: {value[0]:.3f}  |  "
            f"log π: {log_prob[0]:.3f}"
        )
        self.fig.canvas.draw_idle()

    # ── Button callbacks ──────────────────────────────────────────────

    def _on_step(self, event):
        """Single policy step."""
        self._policy_step()

    def _on_run(self, event):
        """Toggle continuous policy stepping."""
        if self.running:
            self.running = False
            if self.timer is not None:
                self.timer.stop()
                self.timer = None
            self.btn_run.label.set_text("Run Policy")
        else:
            if self.policy is None:
                print("[WARN] No policy loaded.")
                return
            self.running = True
            self.btn_run.label.set_text("Stop")
            self.timer = self.fig.canvas.new_timer(interval=200)  # ms between steps
            self.timer.add_callback(self._timer_step)
            self.timer.start()
        self.fig.canvas.draw_idle()

    def _timer_step(self):
        """Called by the animation timer."""
        if not self.running:
            return
        self._policy_step()

    def _on_reset(self, event):
        """Reset policy hidden state (keep current slider positions)."""
        if self.running:
            self._on_run(None)  # stop first
        if self.policy is not None:
            self._reset_policy_state()
            self.info_text.set_text("Policy state reset.")
            self.fig.canvas.draw_idle()

    def _on_load(self, text):
        """Load a checkpoint from the text box path."""
        path = text.strip()
        if not path:
            return
        try:
            self._load_policy(path)
            self.info_text.set_text(f"Loaded: {path}")
        except Exception as e:
            self.info_text.set_text(f"Load failed: {e}")
        self.fig.canvas.draw_idle()

    # ── Sim helpers ───────────────────────────────────────────────────

    def _current_state_tensor(self):
        return torch.tensor(
            [
                [
                    self.s_m1_pitch.val,
                    self.s_m1_yaw.val,
                    self.s_m2_pitch.val,
                    self.s_m2_yaw.val,
                    self.s_sep.val,
                ]
            ],
            device=DEVICE,
            dtype=torch.float32,
        )

    def _result_to_hits(self, result):
        n = result["hit_counts"][0].item()
        seq = result["hit_sequence"][0, :n].cpu().numpy()
        sep = self.s_sep.val
        return [
            {"point": pt, "mirror": "M1" if pt[2] < sep / 2 else "M2"} for pt in seq
        ]

    def _rotation_matrix(self, pitch, yaw):
        px, py = np.radians(pitch), np.radians(yaw)
        cx, sx, cy, sy = np.cos(px), np.sin(px), np.cos(py), np.sin(py)
        return np.array([[cy, sx * sy, cx * sy], [0, cx, -sx], [-sy, sx * cy, cx * cy]])

    def _get_laser_pos(self, state):
        R = self._rotation_matrix(state[0, 0].item(), state[0, 1].item())
        return R @ np.array([0, HOLE_OFFSET, -5])

    # ── Plot setup ────────────────────────────────────────────────────

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 9))

        # Main axes: 3D, schematic, realistic, observation
        self.ax3d = self.fig.add_subplot(141, projection="3d")
        self.ax2d = self.fig.add_subplot(142)
        self.ax2d_real = self.fig.add_subplot(143)
        self.ax_obs = self.fig.add_subplot(144)

        plt.subplots_adjust(bottom=0.35, top=0.93, left=0.05, right=0.97)

        # ── Sliders ──────────────────────────────────────────────────
        ax_m1_pitch = plt.axes([0.10, 0.24, 0.25, 0.025])
        ax_m1_yaw = plt.axes([0.10, 0.20, 0.25, 0.025])
        ax_m2_pitch = plt.axes([0.45, 0.24, 0.25, 0.025])
        ax_m2_yaw = plt.axes([0.45, 0.20, 0.25, 0.025])
        ax_sep = plt.axes([0.10, 0.14, 0.60, 0.025])

        self.s_m1_pitch = Slider(ax_m1_pitch, "M1 Pitch", -5, 5, valinit=0)
        self.s_m1_yaw = Slider(ax_m1_yaw, "M1 Yaw", -5, 5, valinit=0)
        self.s_m2_pitch = Slider(ax_m2_pitch, "M2 Pitch", -5, 5, valinit=0)
        self.s_m2_yaw = Slider(ax_m2_yaw, "M2 Yaw", -5, 5, valinit=0)
        self.s_sep = Slider(ax_sep, "Separation", 50, 400, valinit=200)

        for s in [
            self.s_m1_pitch,
            self.s_m1_yaw,
            self.s_m2_pitch,
            self.s_m2_yaw,
            self.s_sep,
        ]:
            s.on_changed(self.update)

        # ── Policy buttons ───────────────────────────────────────────
        ax_step = plt.axes([0.10, 0.06, 0.08, 0.04])
        ax_run = plt.axes([0.20, 0.06, 0.08, 0.04])
        ax_reset = plt.axes([0.30, 0.06, 0.08, 0.04])

        self.btn_step = Button(ax_step, "Step")
        self.btn_run = Button(ax_run, "Run Policy")
        self.btn_reset = Button(ax_reset, "Reset")

        self.btn_step.on_clicked(self._on_step)
        self.btn_run.on_clicked(self._on_run)
        self.btn_reset.on_clicked(self._on_reset)

        # ── Checkpoint path text box ─────────────────────────────────
        ax_path = plt.axes([0.52, 0.06, 0.35, 0.04])
        self.textbox = TextBox(ax_path, "Ckpt: ", initial="policy.pt")
        self.textbox.on_submit(self._on_load)

        # ── Info text ────────────────────────────────────────────────
        self.info_text = self.fig.text(
            0.50,
            0.01,
            "No policy loaded. Enter checkpoint path and press Enter.",
            ha="center",
            fontsize=9,
            family="monospace",
        )

        self.update(None)
        plt.show()

    # ── Drawing ───────────────────────────────────────────────────────

    def update(self, val):
        state = self._current_state_tensor()
        result = self.sim.simulate(state)
        hits = self._result_to_hits(result)
        self.draw(hits, state)

    def _render_gaussian_spots(self, ax, hits, spot_sigma=0.5):
        res, extent = 256, 15
        x = np.linspace(-extent, extent, res)
        X, Y = np.meshgrid(x, x)

        img = np.zeros((res, res), dtype=np.float32)
        m1_hits = [
            (h["point"][0], h["point"][1], 0.98 ** (i + 1))
            for i, h in enumerate(hits)
            if h["mirror"] == "M1"
        ]
        for px, py, intensity in m1_hits:
            img += intensity * np.exp(
                -((X - px) ** 2 + (Y - py) ** 2) / (2 * spot_sigma**2)
            )

        R2 = X**2 + Y**2
        mirror_mask = R2 <= (DIAMETER / 2) ** 2
        hole_mask = X**2 + (Y - HOLE_OFFSET) ** 2 <= (HOLE_DIAMETER / 2) ** 2

        rgb = np.zeros((res, res, 3), dtype=np.float32)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = img, img * 0.2, img * 0.1

        bg = np.zeros((res, res, 3), dtype=np.float32)
        bg[mirror_mask] = [0.1, 0.10, 0.1]
        bg[hole_mask] = 0

        ax.imshow(
            np.clip(bg + rgb * 2, 0, 1),
            extent=[-extent, extent, -extent, extent],
            origin="lower",
        )
        ax.set_facecolor("black")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")

    def draw(self, hits, state):
        self.ax3d.cla()
        self.ax2d.cla()
        self.ax2d_real.cla()
        self.ax_obs.cla()

        sep = self.s_sep.val
        u = np.linspace(0, 2 * np.pi, 30)

        # ── 3D: mirrors ──────────────────────────────────────────────
        for z, pitch, yaw, c in [
            (0, self.s_m1_pitch.val, self.s_m1_yaw.val, "blue"),
            (sep, self.s_m2_pitch.val, self.s_m2_yaw.val, "red"),
        ]:
            R = self._rotation_matrix(pitch, yaw)
            pos = np.array([0, 0, z])
            for r in np.linspace(0, DIAMETER / 2, 5):
                local = np.stack(
                    [r * np.cos(u), r * np.sin(u), np.zeros_like(u)], axis=1
                )
                world = (R @ local.T).T + pos
                self.ax3d.plot(
                    world[:, 0], world[:, 1], world[:, 2], color=c, alpha=0.3, lw=0.5
                )
            edge = np.stack(
                [DIAMETER / 2 * np.cos(u), DIAMETER / 2 * np.sin(u), np.zeros_like(u)],
                axis=1,
            )
            self.ax3d.plot(*((R @ edge.T) + pos.reshape(3, 1)), color=c, lw=2)

        # Hole
        R1 = self._rotation_matrix(self.s_m1_pitch.val, self.s_m1_yaw.val)
        hole = np.stack(
            [
                HOLE_DIAMETER / 2 * np.cos(u),
                HOLE_OFFSET + HOLE_DIAMETER / 2 * np.sin(u),
                np.zeros_like(u),
            ],
            axis=1,
        )
        self.ax3d.plot(*((R1 @ hole.T)), "yellow", lw=2)

        # Laser & path
        lp = self._get_laser_pos(state)
        self.ax3d.scatter(*lp, c="green", s=80, marker="^")
        if hits:
            path = np.array([lp] + [h["point"] for h in hits])
            self.ax3d.plot(path[:, 0], path[:, 1], path[:, 2], "g-", lw=1.5)
            for h in hits:
                self.ax3d.scatter(
                    *h["point"], c="blue" if h["mirror"] == "M1" else "red", s=15
                )

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title(f"Bounces: {len(hits)}")

        # ── 2D schematic ─────────────────────────────────────────────
        self.ax2d.add_patch(plt.Circle((0, 0), DIAMETER / 2, fill=False, color="blue"))
        self.ax2d.add_patch(
            plt.Circle((0, HOLE_OFFSET), HOLE_DIAMETER / 2, color="yellow", alpha=0.5)
        )
        for i, h in enumerate(hits):
            if h["mirror"] == "M1":
                self.ax2d.scatter(h["point"][0], h["point"][1], c="blue", s=30)
                self.ax2d.annotate(str(i + 1), h["point"][:2], fontsize=7)
        self.ax2d.set_xlim(-15, 15)
        self.ax2d.set_ylim(-15, 15)
        self.ax2d.set_aspect("equal")
        self.ax2d.set_title("M1 Schematic")

        # ── 2D realistic ─────────────────────────────────────────────
        self._render_gaussian_spots(self.ax2d_real, hits)
        self.ax2d_real.set_title("M1 Realistic")

        # ── Policy observation (what the agent sees) ──────────────────
        obs = self._render_obs(hits)
        obs_np = obs[0, 0].cpu().numpy()
        self.ax_obs.imshow(
            obs_np,
            cmap="hot",
            origin="lower",
            extent=[-OBS_EXTENT, OBS_EXTENT, -OBS_EXTENT, OBS_EXTENT],
        )
        self.ax_obs.set_title(f"Policy Obs (64×64)  Step {self.step_count}")
        self.ax_obs.set_aspect("equal")

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    InteractiveHerriottCell(checkpoint_path=ckpt)
