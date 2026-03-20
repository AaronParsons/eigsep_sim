"""lunar_spinner.py — Plotting and animation tools for BLOOM-21cm spacecraft spin.

Rigid-body dynamics (euler_rhs, simulate_torque_free) and inertia utilities
(rod_inertia_about_com, make_x_inertia) live in eigsep_sim.lunar_orbit and
are re-exported here for convenience.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_HERE = os.path.dirname(os.path.abspath(__file__))

from eigsep_sim.lunar_orbit import (
    normalize,
    rod_inertia_about_com,
    make_x_inertia,
    euler_rhs,
    simulate_torque_free,
    OrbiterMission,
)

_cfg = OrbiterMission(os.path.join(_HERE, "bloom_config.yaml"))

__all__ = [
    "normalize",
    "rod_inertia_about_com",
    "make_x_inertia",
    "euler_rhs",
    "simulate_torque_free",
    "make_x_points",
    "set_axes_equal",
    "cartesian_to_spherical_angles",
    "plot_diagnostics",
    "animate_x_spin",
    "OPENING_ANGLE_DEG",
    "ARM_LENGTHS",
    "ARM_MASSES",
    "L_HAT",
]

# ── Hardware parameters (from bloom_config.yaml) ──────────────────────────
OPENING_ANGLE_DEG = _cfg.antenna.opening_angle_deg
ARM_LENGTHS       = _cfg.antenna.arm_lengths   # m, full tip-to-tip length of each rod
ARM_MASSES        = _cfg.antenna.arm_masses    # kg
L_HAT             = _cfg.antenna.l_hat
del _cfg


# ── Visualization geometry ─────────────────────────────────────────────────

def make_x_points(arm_lengths=1.0, opening_angle_deg=90.0, arm_axes=None):
    """
    Return line-segment endpoints for the two rods in body coordinates.

    *arm_lengths* may be a scalar (both rods equal) or a 2-element sequence.
    """
    if arm_axes is None:
        a = np.deg2rad(opening_angle_deg / 2.0)
        u1 = normalize([np.cos(a),  np.sin(a), 0.0])
        u2 = normalize([np.cos(a), -np.sin(a), 0.0])
        arm_axes = [u1, u2]

    arm_lengths = np.broadcast_to(arm_lengths, (len(arm_axes),))
    pts = []
    for u, L in zip(arm_axes, arm_lengths):
        u = normalize(u)
        pts.append(np.array([
            -0.5 * L * np.asarray(u),
             0.5 * L * np.asarray(u),
        ]))
    return pts, arm_axes


# ── Plot helpers ───────────────────────────────────────────────────────────

def set_axes_equal(ax, lim=1.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))


def cartesian_to_spherical_angles(v):
    """
    Convert Cartesian vectors to spherical angles.

    Returns
    -------
    theta_deg : polar angle from +z (degrees), in [0, 180]
    phi_deg   : azimuth angle (degrees), in (−180, 180]
    """
    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v, axis=-1)
    z = np.clip(v[..., 2] / r, -1.0, 1.0)
    return np.degrees(np.arccos(z)), np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def plot_diagnostics(sim, arm_lengths=1.0, opening_angle_deg=90.0, arm_axes=None, stride=None):
    """
    Plot the inertial-frame (θ, φ) orientation of each dipole axis over time
    to visualise sky coverage.

    Parameters
    ----------
    sim : dict
        Output of ``simulate_torque_free``.
    arm_lengths, opening_angle_deg, arm_axes :
        Passed to ``make_x_points`` to define the body-frame dipole axes.
    stride : int or None
        Decimation factor for plotting.  If *None*, auto-chosen to keep ≤ 4000
        plotted samples.
    """
    t    = sim["t"]
    rots = sim["rot"]

    _, arm_axes  = make_x_points(arm_lengths, opening_angle_deg, arm_axes)
    body_axes    = np.asarray([normalize(u) for u in arm_axes])

    stride = stride or max(1, len(t) // 4000)
    t_plot    = t[::stride]
    rots_plot = rots[::stride]

    inertial_axes = np.empty((len(t_plot), len(body_axes), 3))
    for i, R in enumerate(rots_plot):
        inertial_axes[i] = np.array([R.apply(u) for u in body_axes])

    theta_deg, phi_deg = cartesian_to_spherical_angles(inertial_axes)

    plt.figure()
    ax = plt.gca()
    for j in range(len(body_axes)):
        plt.plot(phi_deg[:, j], theta_deg[:, j], '.', ms=2, label=f"dipole {j + 1}")
    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    plt.tight_layout()
    plt.show()


# ── Animation ─────────────────────────────────────────────────────────────

def animate_x_spin(
    sim,
    arm_lengths=1.0,
    opening_angle_deg=90.0,
    arm_axes=None,
    frame_stride=10,
    interval=80,
    show_tip_trails=True,
    trail_len=120,
    save=None,
):
    """
    Animate torque-free spin of the X body, showing trails of the four tips.

    *arm_lengths* may be a scalar (both rods equal) or a 2-element sequence.
    """
    t    = sim["t"][::frame_stride]
    rots = sim["rot"][::frame_stride]

    rods_body, arm_axes = make_x_points(arm_lengths, opening_angle_deg, arm_axes)
    tip_body = np.array([rod[1] for rod in rods_body] + [rod[0] for rod in rods_body])  # (4,3)

    lim = 0.9 * np.max(arm_lengths)
    fig = plt.figure(figsize=(9, 9))
    ax  = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, lim=lim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Torque-free spin of X-shaped rigid body")

    rod_lines  = [ax.plot([], [], [], lw=3)[0] for _ in rods_body]
    tip_pts    = [ax.plot([], [], [], marker="o", ms=5)[0] for _ in range(4)]
    trail_lines = [ax.plot([], [], [], lw=1.5, alpha=0.7)[0] for _ in range(4)] if show_tip_trails else []
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    tip_hist = np.empty((len(t), 4, 3))
    for i, R in enumerate(rots):
        tip_hist[i] = R.apply(tip_body)

    def update(i):
        R = rots[i]
        for line, rod in zip(rod_lines, rods_body):
            xyz = R.apply(rod)
            line.set_data(xyz[:, 0], xyz[:, 1])
            line.set_3d_properties(xyz[:, 2])
        tips = tip_hist[i]
        for j, pt in enumerate(tip_pts):
            pt.set_data([tips[j, 0]], [tips[j, 1]])
            pt.set_3d_properties([tips[j, 2]])
        artists = rod_lines + tip_pts + [txt]
        if show_tip_trails:
            i0 = max(0, i - trail_len + 1)
            for j, line in enumerate(trail_lines):
                tr = tip_hist[i0:i + 1, j, :]
                line.set_data(tr[:, 0], tr[:, 1])
                line.set_3d_properties(tr[:, 2])
            artists += trail_lines
        txt.set_text(f"t = {t[i]:.3f}")
        return artists

    ani = FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)
    if save:
        ani.save(save, dpi=150, fps=max(1, int(round(1000 / interval))))
    plt.show()
    return ani


# ── Example usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = _cfg.antenna.simulate(L_inertial=16.0 * _cfg.antenna.l_hat, t_final=100.0, dt_sim=0.0005)
    plot_diagnostics(sim, arm_lengths=_cfg.antenna.arm_lengths, opening_angle_deg=_cfg.antenna.opening_angle_deg)

    #animate_x_spin(
    #    sim,
    #    arm_lengths=_cfg.antenna.arm_lengths,
    #    opening_angle_deg=_cfg.antenna.opening_angle_deg,
    #    frame_stride=20, interval=120, show_tip_trails=True, trail_len=200,
    #)
