"""
Interactive Herriott Cell visualization - 13 DOF
Uses herriott_sim.py as backend
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from herriott_sim import create_sim, make_state, STATE_DIM, DEVICE
import torch

# Constants for visualization
DIAMETER = 24.4
HOLE_DIAMETER = 3.2
HOLE_OFFSET = 7.0


class InteractiveHerriottCell:
    def __init__(self):
        self.sim = create_sim()
        self.setup_plot()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.ax3d = self.fig.add_subplot(131, projection="3d")
        self.ax2d = self.fig.add_subplot(132)
        self.ax2d_real = self.fig.add_subplot(133)
        plt.subplots_adjust(bottom=0.42, hspace=0.3)

        # --- Slider layout ---
        # Row heights (bottom to top)
        row_h = 0.025
        row_gap = 0.007
        left_x, left_w = 0.08, 0.35
        right_x, right_w = 0.55, 0.35

        def row_y(r):
            return 0.02 + r * (row_h + row_gap)

        # Row 0: Separation (full width)
        ax_sep = plt.axes([left_x, row_y(0), 0.82, row_h])

        # Row 1: M1 translations / M2 translations
        ax_m1_ty = plt.axes([left_x, row_y(1), left_w, row_h])
        ax_m2_ty = plt.axes([right_x, row_y(1), right_w, row_h])

        # Row 2: M1 tx / M2 tx
        ax_m1_tx = plt.axes([left_x, row_y(2), left_w, row_h])
        ax_m2_tx = plt.axes([right_x, row_y(2), right_w, row_h])

        # Row 3: M1 yaw / M2 yaw
        ax_m1_yaw = plt.axes([left_x, row_y(3), left_w, row_h])
        ax_m2_yaw = plt.axes([right_x, row_y(3), right_w, row_h])

        # Row 4: M1 pitch / M2 pitch
        ax_m1_pitch = plt.axes([left_x, row_y(4), left_w, row_h])
        ax_m2_pitch = plt.axes([right_x, row_y(4), right_w, row_h])

        # Row 5: Laser dx / Laser dy
        ax_ldx = plt.axes([left_x, row_y(5), left_w, row_h])
        ax_ldy = plt.axes([right_x, row_y(5), right_w, row_h])

        # Row 6: Laser pitch / Laser yaw
        ax_lpitch = plt.axes([left_x, row_y(6), left_w, row_h])
        ax_lyaw = plt.axes([right_x, row_y(6), right_w, row_h])

        # Create sliders
        # Mirror angular
        self.s_m1_pitch = Slider(ax_m1_pitch, "M1 Pitch (°)", -5, 5, valinit=0)
        self.s_m1_yaw = Slider(ax_m1_yaw, "M1 Yaw (°)", -5, 5, valinit=0)
        self.s_m2_pitch = Slider(ax_m2_pitch, "M2 Pitch (°)", -5, 5, valinit=0)
        self.s_m2_yaw = Slider(ax_m2_yaw, "M2 Yaw (°)", -5, 5, valinit=0)

        # Mirror translation
        self.s_m1_tx = Slider(ax_m1_tx, "M1 ΔX (mm)", -3, 3, valinit=0)
        self.s_m1_ty = Slider(ax_m1_ty, "M1 ΔY (mm)", -3, 3, valinit=0)
        self.s_m2_tx = Slider(ax_m2_tx, "M2 ΔX (mm)", -3, 3, valinit=0)
        self.s_m2_ty = Slider(ax_m2_ty, "M2 ΔY (mm)", -3, 3, valinit=0)

        # Separation
        self.s_sep = Slider(ax_sep, "Separation (mm)", 50, 400, valinit=200)

        # Laser error
        self.s_ldx = Slider(ax_ldx, "Laser ΔX (mm)", -1.5, 1.5, valinit=0)
        self.s_ldy = Slider(ax_ldy, "Laser ΔY (mm)", -1.5, 1.5, valinit=0)
        self.s_lpitch = Slider(ax_lpitch, "Laser Pitch (°)", -5, 5, valinit=0)
        self.s_lyaw = Slider(ax_lyaw, "Laser Yaw (°)", -5, 5, valinit=0)

        all_sliders = [
            self.s_m1_pitch,
            self.s_m1_yaw,
            self.s_m2_pitch,
            self.s_m2_yaw,
            self.s_sep,
            self.s_m1_tx,
            self.s_m1_ty,
            self.s_m2_tx,
            self.s_m2_ty,
            self.s_ldx,
            self.s_ldy,
            self.s_lpitch,
            self.s_lyaw,
        ]
        for s in all_sliders:
            s.on_changed(self.update)

        self.update(None)
        plt.show()

    def _build_state(self):
        """Build 13-DOF state tensor from slider values."""
        return make_state(
            m1_pitch=self.s_m1_pitch.val,
            m1_yaw=self.s_m1_yaw.val,
            m2_pitch=self.s_m2_pitch.val,
            m2_yaw=self.s_m2_yaw.val,
            separation=self.s_sep.val,
            m1_tx=self.s_m1_tx.val,
            m1_ty=self.s_m1_ty.val,
            m2_tx=self.s_m2_tx.val,
            m2_ty=self.s_m2_ty.val,
            laser_dx=self.s_ldx.val,
            laser_dy=self.s_ldy.val,
            laser_pitch=self.s_lpitch.val,
            laser_yaw=self.s_lyaw.val,
        )

    def update(self, val):
        state = self._build_state()
        result = self.sim.simulate(state)
        hits = self._result_to_hits(result)
        self.draw(hits, state)

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
        return np.array(
            [
                [cy, sx * sy, cx * sy],
                [0, cx, -sx],
                [-sy, sx * cy, cx * cy],
            ]
        )

    def _get_laser_pos(self, state):
        """Compute laser origin in world coords matching the sim logic."""
        s = state[0].cpu().numpy()
        R = self._rotation_matrix(s[0], s[1])  # M1 rotation
        m1_pos = np.array([s[5], s[6], 0.0])  # M1 translation
        hole_local = np.array([s[9], HOLE_OFFSET + s[10], -5.0])
        return R @ hole_local + m1_pos

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

        sep = self.s_sep.val
        u = np.linspace(0, 2 * np.pi, 30)

        # Mirror drawing parameters: (base_z, pitch, yaw, tx, ty, color)
        mirrors = [
            (
                0,
                self.s_m1_pitch.val,
                self.s_m1_yaw.val,
                self.s_m1_tx.val,
                self.s_m1_ty.val,
                "blue",
            ),
            (
                sep,
                self.s_m2_pitch.val,
                self.s_m2_yaw.val,
                self.s_m2_tx.val,
                self.s_m2_ty.val,
                "red",
            ),
        ]

        for z, pitch, yaw, tx, ty, c in mirrors:
            R = self._rotation_matrix(pitch, yaw)
            pos = np.array([tx, ty, z])
            for r in np.linspace(0, DIAMETER / 2, 5):
                local = np.stack(
                    [r * np.cos(u), r * np.sin(u), np.zeros_like(u)], axis=1
                )
                world = (R @ local.T).T + pos
                self.ax3d.plot(
                    world[:, 0],
                    world[:, 1],
                    world[:, 2],
                    color=c,
                    alpha=0.3,
                    lw=0.5,
                )
            edge = np.stack(
                [DIAMETER / 2 * np.cos(u), DIAMETER / 2 * np.sin(u), np.zeros_like(u)],
                axis=1,
            )
            self.ax3d.plot(*((R @ edge.T) + pos.reshape(3, 1)), color=c, lw=2)

        # Hole on M1
        R1 = self._rotation_matrix(self.s_m1_pitch.val, self.s_m1_yaw.val)
        m1_pos = np.array([self.s_m1_tx.val, self.s_m1_ty.val, 0.0])
        hole = np.stack(
            [
                HOLE_DIAMETER / 2 * np.cos(u),
                HOLE_OFFSET + HOLE_DIAMETER / 2 * np.sin(u),
                np.zeros_like(u),
            ],
            axis=1,
        )
        hole_world = (R1 @ hole.T) + m1_pos.reshape(3, 1)
        self.ax3d.plot(*hole_world, "yellow", lw=2)

        # Laser origin & path
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

        # 2D schematic — M1 face, shifted by M1 translation
        m1_cx, m1_cy = self.s_m1_tx.val, self.s_m1_ty.val
        self.ax2d.add_patch(
            plt.Circle((m1_cx, m1_cy), DIAMETER / 2, fill=False, color="blue")
        )
        self.ax2d.add_patch(
            plt.Circle(
                (m1_cx, m1_cy + HOLE_OFFSET),
                HOLE_DIAMETER / 2,
                color="yellow",
                alpha=0.5,
            )
        )
        for i, h in enumerate(hits):
            if h["mirror"] == "M1":
                self.ax2d.scatter(h["point"][0], h["point"][1], c="blue", s=30)
                self.ax2d.annotate(str(i + 1), h["point"][:2], fontsize=7)
        self.ax2d.set_xlim(-15, 15)
        self.ax2d.set_ylim(-15, 15)
        self.ax2d.set_aspect("equal")
        self.ax2d.set_title("M1 Schematic")

        # 2D realistic spots
        self._render_gaussian_spots(self.ax2d_real, hits)
        self.ax2d_real.set_title("M1 Realistic")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    InteractiveHerriottCell()
