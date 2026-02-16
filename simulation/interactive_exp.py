"""
Interactive Herriott Cell visualization
Uses herriott_sim.py as backend
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from herriott_sim import create_sim, DEVICE
import torch

# Constants for visualization
DIAMETER = 25.4
HOLE_DIAMETER = 3.0
HOLE_OFFSET = 7.0


class InteractiveHerriottCell:
    def __init__(self):
        self.sim = create_sim(mounted_laser=True)
        self.setup_plot()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.ax3d = self.fig.add_subplot(131, projection="3d")
        self.ax2d = self.fig.add_subplot(132)
        self.ax2d_real = self.fig.add_subplot(133)
        plt.subplots_adjust(bottom=0.30)

        ax_m1_pitch = plt.axes([0.15, 0.20, 0.30, 0.03])
        ax_m1_yaw = plt.axes([0.15, 0.15, 0.30, 0.03])
        ax_m2_pitch = plt.axes([0.55, 0.20, 0.30, 0.03])
        ax_m2_yaw = plt.axes([0.55, 0.15, 0.30, 0.03])
        ax_sep = plt.axes([0.15, 0.08, 0.70, 0.03])

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

        self.update(None)
        plt.show()

    def update(self, val):
        state = torch.tensor(
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
        return np.array([[cy, sx * sy, cx * sy], [0, cx, -sx], [-sy, sx * cy, cx * cy]])

    def _get_laser_pos(self, state):
        R = self._rotation_matrix(state[0, 0].item(), state[0, 1].item())
        return R @ np.array([0, HOLE_OFFSET, -5])

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

        # Draw mirrors
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

        # 2D schematic
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

        # 2D realistic
        self._render_gaussian_spots(self.ax2d_real, hits)
        self.ax2d_real.set_title("M1 Realistic")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    InteractiveHerriottCell()
