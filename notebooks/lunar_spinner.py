import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


# -----------------------------
# Geometry / inertia utilities
# -----------------------------
def skew(w):
    wx, wy, wz = w
    return np.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0],
    ])


def normalize(v, eps=1e-15):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def rod_inertia_about_com(m, L, axis):
    """
    Inertia tensor of a thin rod of length L, mass m, centered at origin,
    oriented along unit vector `axis`.
    """
    u = normalize(axis)
    return (m * L**2 / 12.0) * (np.eye(3) - np.outer(u, u))


def make_x_inertia(
    arm_lengths=1.0,
    arm_masses=1.0,
    opening_angle_deg=90.0,
    arm_axes=None,
):
    """
    Build inertia tensor for an X made of two thin rods crossing at center.

    Default:
      two rods in the body x-y plane, symmetric about +x, with included angle
      `opening_angle_deg` between them.

    arm_lengths and arm_masses may each be a scalar (applied to both rods) or a
    2-element sequence giving independent values for each rod.
    """
    if arm_axes is None:
        a = np.deg2rad(opening_angle_deg / 2.0)
        u1 = np.array([np.cos(a),  np.sin(a), 0.0])
        u2 = np.array([np.cos(a), -np.sin(a), 0.0])
        arm_axes = [u1, u2]

    arm_lengths = np.broadcast_to(arm_lengths, (len(arm_axes),))
    arm_masses  = np.broadcast_to(arm_masses,  (len(arm_axes),))

    I = np.zeros((3, 3))
    for u, L, m in zip(arm_axes, arm_lengths, arm_masses):
        I += rod_inertia_about_com(m, L, u)
    return I


def make_x_points(arm_lengths=1.0, opening_angle_deg=90.0, arm_axes=None):
    """
    Return line-segment endpoints for the two rods in body coordinates.

    arm_lengths may be a scalar (both rods equal) or a 2-element sequence.
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


# -----------------------------
# Rigid-body dynamics
# -----------------------------
def euler_rhs(t, y, I, Iinv):
    """
    State y = [q_w, q_x, q_y, q_z, Lx, Ly, Lz]
    Quaternion maps body -> inertial.

    Torque-free rigid body:
      dL_body/dt = -omega x L_body
      dq/dt from omega_body
    """
    q = y[:4]
    L = y[4:]

    q = q / np.linalg.norm(q)
    omega = Iinv @ L

    # qdot = 0.5 * q ⊗ [0, omega]
    w, x, yq, z = q
    ox, oy, oz = omega
    qdot = 0.5 * np.array([
        -x * ox - yq * oy - z * oz,
         w * ox + yq * oz - z * oy,
         w * oy + z * ox - x * oz,
         w * oz + x * oy - yq * ox,
    ])

    Ldot = -np.cross(omega, L)
    return np.concatenate([qdot, Ldot])


def simulate_torque_free(I, L_inertial, t_final=40.0, dt_sim=0.002):
    """
    Simulate a torque-free rigid body with a specified initial angular
    momentum vector in inertial coordinates.

    Initial body and inertial axes are aligned at t=0, so initially
    L_body = L_inertial.
    """
    I = np.asarray(I, dtype=float)
    Iinv = np.linalg.inv(I)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    L0_body = np.asarray(L_inertial, dtype=float)
    y0 = np.concatenate([q0, L0_body])

    t_eval = np.arange(0.0, t_final + 0.5 * dt_sim, dt_sim)

    sol = solve_ivp(
        euler_rhs,
        (0.0, t_final),
        y0,
        t_eval=t_eval,
        args=(I, Iinv),
        rtol=1e-9,
        atol=1e-11,
    )

    qs = sol.y[:4].T
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    L_body = sol.y[4:].T
    omega_body = (Iinv @ L_body.T).T

    # scipy Rotation uses quaternion order [x, y, z, w]
    rots = Rotation.from_quat(
        np.column_stack([qs[:, 1], qs[:, 2], qs[:, 3], qs[:, 0]])
    )

    # Inertial-frame angular momentum, should be constant up to numerical error
    L_inert = np.array([rots[i].apply(L_body[i]) for i in range(len(sol.t))])

    # Inertial-frame angular velocity
    omega_inert = np.array([rots[i].apply(omega_body[i]) for i in range(len(sol.t))])

    # Kinetic energy
    T = 0.5 * np.einsum("...i,...i->...", omega_body, L_body)

    return {
        "t": sol.t,
        "rot": rots,
        "q": qs,
        "L_body": L_body,
        "L_inert": L_inert,
        "omega_body": omega_body,
        "omega_inert": omega_inert,
        "T": T,
        "I": I,
    }


# -----------------------------
# Plot helpers
# -----------------------------
def set_axes_equal(ax, lim=1.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))


def cartesian_to_spherical_angles(v):
    """
    Convert vectors to spherical angles.

    Returns
    -------
    theta_deg : polar angle from +z in degrees, in [0, 180]
    phi_deg   : azimuth angle in degrees, in (-180, 180]
    """
    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v, axis=-1)
    z = np.clip(v[..., 2] / r, -1.0, 1.0)
    theta = np.degrees(np.arccos(z))
    phi = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return theta, phi



def plot_diagnostics(sim, arm_lengths=1.0, opening_angle_deg=90.0, arm_axes=None, stride=None):
    """
    Plot the inertial-frame (theta, phi) orientation of the axis of each dipole
    over time to visualize sky coverage.

    Parameters
    ----------
    sim : dict
        Output of simulate_torque_free.
    arm_lengths, opening_angle_deg, arm_axes :
        Passed to make_x_points() so the body-frame dipole axes are defined
        consistently with the simulated hardware geometry.
    stride : int or None
        Optional decimation factor for plotting. If None, choose a value that
        limits the number of plotted samples to roughly 4000.
    """
    t = sim["t"]
    rots = sim["rot"]

    _, arm_axes = make_x_points(arm_lengths, opening_angle_deg, arm_axes)
    body_axes = np.asarray([normalize(u) for u in arm_axes])

    n = len(t)
    if stride is None:
        stride = max(1, n // 4000)

    t_plot = t[::stride]
    rots_plot = rots[::stride]

    inertial_axes = np.empty((len(t_plot), len(body_axes), 3))
    for i, R in enumerate(rots_plot):
        inertial_axes[i] = np.array([R.apply(u) for u in body_axes])

    theta_deg, phi_deg = cartesian_to_spherical_angles(inertial_axes)

    plt.figure()
    ax = plt.gca()

    for j in range(len(body_axes)):
        plt.plot(phi_deg[:, j], theta_deg[:, j], '.', ms=2, label=f"dipole {j + 1}")

    ax.set_ylabel(r"$\theta$ [deg]")
    #ax.set_ylim(0, 180)
    ax.set_xlabel(r"$\phi$ [deg]")
    #ax.set_xlim(-180, 180)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Animation
# -----------------------------
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

    arm_lengths may be a scalar (both rods equal) or a 2-element sequence.
    """
    t = sim["t"][::frame_stride]
    rots = sim["rot"][::frame_stride]

    rods_body, arm_axes = make_x_points(arm_lengths, opening_angle_deg, arm_axes)
    tip_body = np.array([rod[1] for rod in rods_body] + [rod[0] for rod in rods_body])  # (4,3)

    lim = 0.9 * np.max(arm_lengths)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, lim=lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Torque-free spin of X-shaped rigid body")

    rod_lines = [ax.plot([], [], [], lw=3)[0] for _ in rods_body]
    tip_pts = [ax.plot([], [], [], marker="o", ms=5)[0] for _ in range(4)]
    trail_lines = [ax.plot([], [], [], lw=1.5, alpha=0.7)[0] for _ in range(4)] if show_tip_trails else []

    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # precompute all tip positions in inertial frame: (nframe, 4, 3)
    tip_hist = np.empty((len(t), 4, 3))
    for i, R in enumerate(rots):
        tip_hist[i] = R.apply(tip_body)

    def update(i):
        R = rots[i]

        # rods
        for line, rod in zip(rod_lines, rods_body):
            xyz = R.apply(rod)
            line.set_data(xyz[:, 0], xyz[:, 1])
            line.set_3d_properties(xyz[:, 2])

        # current tip markers
        tips = tip_hist[i]
        for j, pt in enumerate(tip_pts):
            pt.set_data([tips[j, 0]], [tips[j, 1]])
            pt.set_3d_properties([tips[j, 2]])

        # trails
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

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=interval,
        blit=False,
    )

    if save:
        ani.save(save, dpi=150, fps=max(1, int(round(1000 / interval))))

    plt.show()
    return ani

# -----------------------------
# Default hardware parameters
# -----------------------------
OPENING_ANGLE_DEG = 90.0
ARM_LENGTHS = [6.0, 4.0]   # m, full tip-to-tip length of each rod
ARM_MASSES  = [1.0, 2.25]  # kg

_eps = 0.2
L_HAT = np.array([1.0, 0.0, _eps])
L_HAT /= np.linalg.norm(L_HAT)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    arm_lengths = ARM_LENGTHS
    arm_masses  = ARM_MASSES
    opening_angle_deg = OPENING_ANGLE_DEG

    # Arbitrary inertial angular momentum vector
    L0 = 16
    L_inertial = L0 * L_HAT

    I = make_x_inertia(
        arm_lengths=arm_lengths,
        arm_masses=arm_masses,
        opening_angle_deg=opening_angle_deg,
    )

    sim = simulate_torque_free(
        I,
        L_inertial=L_inertial,
        t_final=100.0,
        dt_sim=0.0005,
    )
    plot_diagnostics(sim, arm_lengths=arm_lengths, opening_angle_deg=opening_angle_deg)

    #animate_x_spin(
    #    sim,
    #    arm_lengths=arm_lengths,
    #    opening_angle_deg=opening_angle_deg,
    #    frame_stride=20,
    #    interval=120,
    #    show_tip_trails=True,
    #    trail_len=200,
    #)
