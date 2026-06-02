"""Reusable lunar campaign, orbit, and torque-free rigid-body utilities."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    BarycentricMeanEcliptic,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation, Slerp

from .beam import Beam
from .lunar_surface import UniformLunarSurface
from .models import T21cmModel
from .observer import LunarOrbit
from .simulate import ForwardModel
from .sky import Sky


def normalize_vector(vector, eps=1e-15):
    """Return a normalized vector; reject near-zero inputs."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("zero-length vector")
    return vector / norm


def make_ecliptic_orbit_normals(
    inclinations_deg=(-15, 15), ascending_node_lon_deg=0, equinox="J2000"
):
    """Return Galactic-frame orbit normals with a common ecliptic node."""
    node = np.deg2rad(float(ascending_node_lon_deg))
    node_axis = np.array([np.cos(node), np.sin(node), 0.0])
    normals_ecl = np.array(
        [
            Rotation.from_rotvec(node_axis * np.deg2rad(inclination)).apply(
                [0.0, 0.0, 1.0]
            )
            for inclination in inclinations_deg
        ]
    )
    frame = BarycentricMeanEcliptic(equinox=Time(equinox))
    coords = SkyCoord(
        CartesianRepresentation(normals_ecl.T, unit=u.one), frame=frame
    )
    normals_gal = coords.galactic.cartesian.xyz.value.T
    return normals_gal / np.linalg.norm(normals_gal, axis=1, keepdims=True)


def crossed_rod_inertia(
    arm_lengths_m=(6.0, 4.0),
    arm_masses_kg=(1.0, 2.25),
    opening_angle_deg=90.0,
    arm_axes=None,
):
    """Inertia tensor for thin rods crossing at their centers."""
    if arm_axes is None:
        angle = np.deg2rad(opening_angle_deg / 2.0)
        arm_axes = [
            [np.cos(angle), np.sin(angle), 0.0],
            [np.cos(angle), -np.sin(angle), 0.0],
        ]
    lengths = np.broadcast_to(arm_lengths_m, (len(arm_axes),))
    masses = np.broadcast_to(arm_masses_kg, (len(arm_axes),))
    inertia = np.zeros((3, 3))
    for axis, length, mass in zip(arm_axes, lengths, masses):
        unit_axis = normalize_vector(axis)
        inertia += (mass * length**2 / 12.0) * (
            np.eye(3) - np.outer(unit_axis, unit_axis)
        )
    return inertia


def angular_momentum_for_spin_period(inertia, direction, spin_period_s):
    """Scale an angular-momentum direction to an initial rotation period.

    Torque-free tumbling generally has multiple attitude timescales. This
    convention sets the initial angular-speed magnitude to
    ``2 * pi / spin_period_s`` when body and inertial axes are aligned.
    """
    spin_period_s = float(spin_period_s)
    if spin_period_s <= 0.0:
        raise ValueError("spin_period_s must be positive")
    direction = normalize_vector(direction)
    inertia_inv = np.linalg.inv(np.asarray(inertia, dtype=float))
    unit_omega = inertia_inv @ direction
    scale = (2.0 * np.pi / spin_period_s) / np.linalg.norm(unit_omega)
    return direction * scale


def torque_free_rhs(_time_s, state, inertia, inertia_inv):
    """Quaternion and body-angular-momentum RHS for torque-free rotation."""
    quat = state[:4] / np.linalg.norm(state[:4])
    momentum = state[4:]
    omega = inertia_inv @ momentum
    w, x, y, z = quat
    ox, oy, oz = omega
    quat_dot = 0.5 * np.array(
        [
            -x * ox - y * oy - z * oz,
            w * ox + y * oz - z * oy,
            w * oy + z * ox - x * oz,
            w * oz + x * oy - y * ox,
        ]
    )
    return np.concatenate([quat_dot, -np.cross(omega, momentum)])


def integrate_torque_free(inertia, angular_momentum_inertial, times_s):
    """Integrate torque-free body rotations at requested simulation times."""
    times_s = np.asarray(times_s, dtype=float)
    if times_s.ndim != 1 or len(times_s) == 0:
        raise ValueError("times_s must be a non-empty one-dimensional array")
    inertia = np.asarray(inertia, dtype=float)
    inertia_inv = np.linalg.inv(inertia)
    state0 = np.concatenate(
        [
            [1.0, 0.0, 0.0, 0.0],
            np.asarray(angular_momentum_inertial, dtype=float),
        ]
    )
    if len(times_s) == 1:
        states = state0[:, None]
    else:
        solution = solve_ivp(
            torque_free_rhs,
            (float(times_s[0]), float(times_s[-1])),
            state0,
            t_eval=times_s,
            args=(inertia, inertia_inv),
            rtol=1e-9,
            atol=1e-11,
        )
        states = solution.y
    quats_wxyz = states[:4].T
    quats_wxyz /= np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    rotations = Rotation.from_quat(
        np.column_stack([quats_wxyz[:, 1:], quats_wxyz[:, 0]])
    )
    momentum_body = states[4:].T
    omega_body = (inertia_inv @ momentum_body.T).T
    return {
        "times_s": times_s,
        "rotations_body_to_gal": rotations,
        "quaternions_wxyz": quats_wxyz,
        "momentum_body": momentum_body,
        "momentum_inertial": rotations.apply(momentum_body),
        "omega_body": omega_body,
        "kinetic_energy": 0.5
        * np.einsum("ti,ti->t", omega_body, momentum_body),
        "inertia": inertia,
    }


def interpolate_body_rotations(simulation, observation_times_s):
    """Interpolate body-to-Galactic rotations onto observation times."""
    times_s = np.asarray(simulation["times_s"], dtype=float)
    observation_times_s = np.asarray(observation_times_s, dtype=float)
    if len(times_s) == 1:
        return Rotation.from_quat(
            np.broadcast_to(
                simulation["rotations_body_to_gal"].as_quat()[0],
                (len(observation_times_s), 4),
            )
        )
    return Slerp(times_s, simulation["rotations_body_to_gal"])(
        observation_times_s
    )


@dataclass
class LunarCampaignResult:
    """Stacked outputs and diagnostics for a multi-spacecraft campaign."""

    spectra_K: np.ndarray
    masks: np.ndarray
    geometry: list
    body_rots: np.ndarray
    times: Time


class LunarCampaign:
    """Build and run a proposal-style multi-spacecraft lunar campaign."""

    def __init__(self, config, sky=None, sky_coeffs=None):
        self.config = config
        orbit_cfg = config["orbit"]
        spacecraft_cfg = config["spacecraft"]
        freq_cfg = config["frequency"]
        self.freqs_hz = (
            np.linspace(
                freq_cfg["min_mhz"], freq_cfg["max_mhz"], freq_cfg["nchan"]
            )
            * 1e6
        )
        self.orbit_normals_gal = make_ecliptic_orbit_normals(
            orbit_cfg.get("inclinations_deg", (-15, 15)),
            orbit_cfg.get("ascending_node_lon_deg", 0.0),
            orbit_cfg.get("equinox", "J2000"),
        )
        opening = spacecraft_cfg.get("opening_angle_deg", 90.0)
        angle = np.deg2rad(opening / 2.0)
        self.arm_axes_body = np.array(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [np.cos(angle), -np.sin(angle), 0.0],
            ]
        )
        self.beam = Beam.from_dipole(
            config["antenna"]["nside"],
            self.freqs_hz,
            arm_lengths_m=spacecraft_cfg["arm_lengths_m"],
            u_body=self.arm_axes_body,
            K=config["antenna"].get("n_modes", 3),
        )
        if sky is None:
            sky = Sky.from_gsm(
                config["sky"]["nside"],
                self.freqs_hz,
                n_modes=config["sky"].get("n_modes", 3),
                include_flat=True,
            )
        self.sky = sky
        self.sky_coeffs = (
            sky.init_coeffs() if sky_coeffs is None else np.asarray(sky_coeffs)
        )
        signal_cfg = config.get(
            "signal_21cm", {"enabled": True, "model_index": 0}
        )
        self.T21cm_K = np.zeros_like(self.freqs_hz, dtype=np.float32)
        if signal_cfg.get("enabled", True):
            self.T21cm_K = T21cmModel()(
                self.freqs_hz, model_index=signal_cfg.get("model_index", 0)
            ).astype(np.float32)
            sky_maps = self.sky.basis.deproject(self.sky_coeffs) + self.T21cm_K
            self.sky_coeffs = self.sky.basis.project(sky_maps).astype(
                np.float32
            )
        self.source_config = config.get(
            "sources", {"earth": {"enabled": False}, "sun": {"enabled": False}}
        )
        self.surface_model = UniformLunarSurface(
            config["surface"]["T_regolith_K"]
        )
        epoch = Time(orbit_cfg["epoch"])
        self.observers = [
            LunarOrbit(
                altitude=orbit_cfg["altitude_km"] * 1e3,
                rot_orbit_vec=normal,
                rot_spin_vec=[0.0, 0.0, 1.0],
                spin_period=0.0,
                t0=epoch,
            )
            for normal in self.orbit_normals_gal
        ]
        self.forward_models = [
            ForwardModel(
                obs, self.beam, self.sky, surface_model=self.surface_model
            )
            for obs in self.observers
        ]

    def make_times(self):
        """Return configured observation epochs."""
        cfg = self.config["integration"]
        offsets_s = np.linspace(
            0.0, cfg["duration_hours"] * 3600.0, cfg["ntimes"], endpoint=False
        )
        return self.observers[0].t0 + offsets_s * u.s

    def make_body_rotations(self, times):
        """Return per-spacecraft topocentric-to-body rotations."""
        offsets_s = (Time(times) - self.observers[0].t0).to_value(u.s)
        cfg = self.config["spacecraft"]
        inertia = crossed_rod_inertia(
            cfg["arm_lengths_m"],
            cfg["arm_masses_kg"],
            cfg.get("opening_angle_deg", 90.0),
        )
        max_time = max(float(np.max(offsets_s)), 1e-6)
        spin_period_s = cfg["spin_period_s"]
        momentum_direction = cfg["angular_momentum_direction_gal"]
        angular_momentum = angular_momentum_for_spin_period(
            inertia, momentum_direction, spin_period_s
        )
        self.attitude_inertia = inertia
        self.angular_momentum_gal = angular_momentum
        self.initial_angular_speed_rad_s = np.linalg.norm(
            np.linalg.solve(inertia, angular_momentum)
        )
        requested_step_s = self.config["integration"].get(
            "attitude_step_s", spin_period_s / 20.0
        )
        dt_s = min(requested_step_s, spin_period_s / 20.0)
        sim_times = np.arange(0.0, max_time + dt_s, dt_s)
        if sim_times[-1] < max_time:
            sim_times = np.append(sim_times, max_time)
        base = interpolate_body_rotations(
            integrate_torque_free(inertia, angular_momentum, sim_times),
            offsets_s,
        )
        phases = cfg.get(
            "attitude_phase_offsets_deg", [0.0] * len(self.observers)
        )
        body_rots = []
        for phase_deg in phases:
            phase = Rotation.from_rotvec(
                normalize_vector(momentum_direction) * np.deg2rad(phase_deg)
            )
            body_rots.append((phase * base).inv().as_matrix())
        return np.asarray(body_rots, dtype=np.float32)

    def run(self, times=None):
        """Simulate stacked spectra and retain geometry diagnostics."""
        times = self.make_times() if times is None else Time(times)
        body_rots = self.make_body_rotations(times)
        geometry, spectra, masks = [], [], []
        for index, fwd in enumerate(self.forward_models):
            geom = fwd.precompute_geometry(
                times=times, body_rots=body_rots[index]
            )
            geometry.append(geom)
            masks.append(np.asarray(geom["terrain_masks_jax"]))
            spectra.append(
                np.asarray(
                    fwd.simulate(self.sky_coeffs, self.beam.coeffs, geom=geom)
                )
            )
        return LunarCampaignResult(
            np.stack(spectra), np.stack(masks), geometry, body_rots, times
        )
