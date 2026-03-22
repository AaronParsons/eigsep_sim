"""
Ephemeris for a spacecraft in circular lunar orbit, and BLOOM-21cm mission
configuration.

:func:`circular_orbital_period` and :class:`LunarOrbit` provide the observer
interface used throughout eigsep_sim.

The mission-level classes (:class:`Antenna`, :class:`Observation`,
:class:`OrbiterMission`) are BLOOM-specific.  They consume :class:`LunarOrbit`
together with an external YAML configuration file.
"""

from __future__ import annotations

import numpy as np
import healpy
import yaml
import astropy.units as u
from astropy.time import Time
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from ._observer import Observer, ICRS2GAL
from .coord import rot_m
from .const import R_MOON, GM_MOON, c as _C
from .beam import (
    realized_efficiency as _realized_eff,
    gsm_like_tsky_K as _gsm_tsky,
)


# ── Orbital mechanics ─────────────────────────────────────────────────────

def circular_orbital_period(altitude):
    """
    Orbital period [s] for a circular lunar orbit at *altitude* metres above
    the surface.

    Uses Kepler's third law with the Moon's standard gravitational parameter
    GM_MOON = 4.9048695 × 10¹² m³ s⁻².

    Parameters
    ----------
    altitude : float
        Altitude above the lunar surface [m].

    Returns
    -------
    period : float
        Orbital period [s].
    """
    r = R_MOON + altitude
    return 2.0 * np.pi * np.sqrt(r ** 3 / GM_MOON)


class LunarOrbit(Observer):
    """
    Spacecraft in a circular lunar orbit with an independent spin.

    The orbital period is computed automatically from *altitude* via Kepler's
    third law (:func:`circular_orbital_period`).  The galactic frame is used
    as the inertial reference throughout, consistent with the GSM sky model.

    Parameters
    ----------
    altitude : float
        Orbital altitude above the lunar surface [m].
    rot_orbit_vec : array_like, shape (3,)
        Unit vector in the galactic frame normal to the orbital plane.
        The spacecraft orbits counter-clockwise around this axis.
    rot_spin_vec : array_like, shape (3,)
        Unit vector in the galactic frame about which the spacecraft spins.
    start_pos : array_like, shape (3,), optional
        Unit vector in the galactic frame giving the initial orbital
        position direction from the Moon centre.  Default: [1, 0, 0].
    spin_period : float
        Spacecraft spin period [s].  0 means no spin.
    t0 : `~astropy.time.Time` or str, optional
        Reference epoch: spacecraft is at ``start_pos`` with zero spin
        phase at this time.  Default: J2000.
    """

    def __init__(
        self,
        altitude,
        rot_orbit_vec,
        rot_spin_vec,
        start_pos=None,
        spin_period=0.0,
        t0=None,
    ):
        self.altitude = altitude
        self.orbital_radius = R_MOON + altitude
        self.orbital_period = circular_orbital_period(altitude)

        self.rot_orbit_vec = np.asarray(rot_orbit_vec, dtype=float)
        self.rot_orbit_vec /= np.linalg.norm(self.rot_orbit_vec)

        self.rot_spin_vec = np.asarray(rot_spin_vec, dtype=float)
        self.rot_spin_vec /= np.linalg.norm(self.rot_spin_vec)

        if start_pos is None:
            start_pos = np.array([1.0, 0.0, 0.0])
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_pos /= np.linalg.norm(self.start_pos)

        self.spin_period = float(spin_period)
        self.t0 = Time("J2000") if t0 is None else Time(t0)
        super().__init__()
        self.time = self.t0

        self._th_orbit = 0.0
        self._th_spin = 0.0

    def set_time(self, t):
        """Set the current epoch and update orbital and spin phases."""
        super().set_time(t)
        dt = (self.time - self.t0).to(u.s).value
        self._th_orbit = 2 * np.pi * dt / self.orbital_period
        self._th_spin = (
            2 * np.pi * dt / self.spin_period if self.spin_period != 0.0 else 0.0
        )

    def set_phases(self, th_orbit, th_spin=0.0):
        """
        Directly set orbital and spin phases in radians.

        Useful for looping over orbital/spin configurations without
        converting to absolute time.
        """
        self._th_orbit = float(th_orbit)
        self._th_spin = float(th_spin)

    def spacecraft_position(self):
        """
        Position of the spacecraft relative to the Moon centre,
        expressed in the galactic frame.

        Returns
        -------
        pos : ndarray, shape (3,)
            Position vector in metres.
        """
        R = rot_m(self._th_orbit, self.rot_orbit_vec)
        return R @ (self.start_pos * self.orbital_radius)

    def rot_gal2top(self):
        """
        3x3 rotation matrix mapping galactic unit vectors to spacecraft-
        body unit vectors.

        At zero spin phase the spacecraft frame is aligned with the
        galactic frame.  The spin rotation is applied counter-clockwise
        around ``rot_spin_vec``.
        """
        # R_spin maps topocentric → galactic, so gal2top = R_spin.T
        R_spin = rot_m(self._th_spin, self.rot_spin_vec)
        return R_spin.T

    def above_horizon(self, nside):
        """
        Boolean HEALPix mask (galactic frame) of pixels not occluded by
        the Moon.

        Returns
        -------
        mask : ndarray of bool, shape (npix,)
            True for pixels whose line of sight from the spacecraft does
            not intersect the lunar sphere.
        """
        from .utils import moon_surface_distance

        npix = healpy.nside2npix(nside)
        vecs = np.array(
            healpy.pix2vec(nside, np.arange(npix)), dtype=np.float64
        )  # (3, npix)
        pos = self.spacecraft_position()          # (3,) metres, from Moon centre
        d = float(np.linalg.norm(pos))
        moon_dir = pos / d                        # direction from Moon centre to spacecraft
        # angle between each pixel direction and moon_dir (i.e. outward radial)
        dot = np.clip(moon_dir @ vecs, -1.0, 1.0)
        angle = np.arccos(dot)
        dist = moon_surface_distance(angle, d)
        return np.isnan(dist)                     # NaN → no intersection → pixel visible


# ── BLOOM mission support ─────────────────────────────────────────────────
#
# Everything below supports the BLOOM-21cm lunar radiometer mission.
# The general eigsep_sim package only needs the code above.

# ── Geometry utilities ─────────────────────────────────────────────────────

def normalize(v, eps: float = 1e-15) -> np.ndarray:
    """Return a unit vector in the direction of *v*; raise if nearly zero."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def _skew(w) -> np.ndarray:
    """3×3 skew-symmetric (cross-product) matrix for vector *w*."""
    wx, wy, wz = w
    return np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]])


def _arm_axes(opening_angle_deg: float) -> list[np.ndarray]:
    """Body-frame unit vectors for the two arms of a symmetric X dipole."""
    a = np.deg2rad(opening_angle_deg / 2.0)
    return [
        normalize([np.cos(a),  np.sin(a), 0.0]),
        normalize([np.cos(a), -np.sin(a), 0.0]),
    ]


def _perp_to(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to *v*."""
    v = normalize(v)
    ref = np.array([1., 0., 0.]) if abs(v[0]) < 0.9 else np.array([0., 1., 0.])
    w = np.cross(v, ref)
    return w / np.linalg.norm(w)


# ── Rigid-body spacecraft dynamics ─────────────────────────────────────────

def rod_inertia_about_com(m: float, L: float, axis) -> np.ndarray:
    """Inertia tensor of a thin rod (mass *m*, length *L*) centred at origin."""
    u = normalize(axis)
    return (m * L ** 2 / 12.0) * (np.eye(3) - np.outer(u, u))


def make_x_inertia(
    arm_lengths=1.0,
    arm_masses=1.0,
    opening_angle_deg: float = 90.0,
    arm_axes=None,
) -> np.ndarray:
    """Inertia tensor for an X-shaped body of two thin rods crossing at centre."""
    if arm_axes is None:
        a = np.deg2rad(opening_angle_deg / 2.0)
        arm_axes = [
            np.array([np.cos(a),  np.sin(a), 0.0]),
            np.array([np.cos(a), -np.sin(a), 0.0]),
        ]
    arm_lengths = np.broadcast_to(arm_lengths, (len(arm_axes),))
    arm_masses  = np.broadcast_to(arm_masses,  (len(arm_axes),))
    I = np.zeros((3, 3))
    for u, L, m in zip(arm_axes, arm_lengths, arm_masses):
        I += rod_inertia_about_com(m, L, u)
    return I


def euler_rhs(t, y, I, Iinv) -> np.ndarray:
    """RHS for torque-free rigid-body ODE.  State y = [q_w, q_x, q_y, q_z, Lx, Ly, Lz]."""
    q = y[:4] / np.linalg.norm(y[:4])
    L = y[4:]
    omega = Iinv @ L
    w, x, yq, z = q
    ox, oy, oz = omega
    qdot = 0.5 * np.array([
        -x*ox - yq*oy - z*oz,
         w*ox + yq*oz -  z*oy,
         w*oy +  z*ox -  x*oz,
         w*oz +  x*oy - yq*ox,
    ])
    return np.concatenate([qdot, -np.cross(omega, L)])


def simulate_torque_free(
    I, L_inertial, t_final: float = 40.0, dt_sim: float = 0.002
) -> dict:
    """
    Integrate torque-free rigid-body equations from rest (body and inertial
    axes aligned at t = 0).

    Returns a dict with keys: t, rot, q, L_body, L_inert, omega_body,
    omega_inert, T, I.
    """
    I    = np.asarray(I, dtype=float)
    Iinv = np.linalg.inv(I)
    y0   = np.concatenate([[1., 0., 0., 0.], np.asarray(L_inertial, dtype=float)])
    t_eval = np.arange(0., t_final + 0.5 * dt_sim, dt_sim)

    sol = solve_ivp(euler_rhs, (0., t_final), y0,
                    t_eval=t_eval, args=(I, Iinv), rtol=1e-9, atol=1e-11)

    qs = sol.y[:4].T
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    L_body     = sol.y[4:].T
    omega_body = (Iinv @ L_body.T).T
    rots = Rotation.from_quat(np.column_stack([qs[:, 1], qs[:, 2], qs[:, 3], qs[:, 0]]))
    L_inert     = np.array([rots[i].apply(L_body[i])     for i in range(len(sol.t))])
    omega_inert = np.array([rots[i].apply(omega_body[i]) for i in range(len(sol.t))])
    return {
        "t": sol.t, "rot": rots, "q": qs,
        "L_body": L_body, "L_inert": L_inert,
        "omega_body": omega_body, "omega_inert": omega_inert,
        "T": 0.5 * np.einsum("...i,...i->...", omega_body, L_body),
        "I": I,
    }


# ── Antenna sub-object ─────────────────────────────────────────────────────

class Antenna:
    """
    Physical antenna + receiver front-end for an X-dipole spacecraft.

    Bundles dipole geometry, inertia, receiver impedance model, and all
    antenna/dynamics physics as methods that operate on stored parameters.
    """

    def __init__(
        self,
        arm_lengths: list[float],
        arm_masses: list[float],
        opening_angle_deg: float,
        l_hat_raw,
        r_loss_ohm: float,
        z_rx_ohm: float,
        x_scale: float,
        t_rx: float,
    ) -> None:
        self.arm_lengths       = arm_lengths
        self.arm_masses        = arm_masses
        self.opening_angle_deg = opening_angle_deg
        self.l_hat             = normalize(l_hat_raw)
        self.r_loss_ohm        = r_loss_ohm
        self.z_rx_ohm          = z_rx_ohm
        self.x_scale           = x_scale
        self.t_rx              = t_rx
        # derived geometry
        self.u_body  = np.array(_arm_axes(opening_angle_deg))          # (2, 3)
        self.inertia = make_x_inertia(arm_lengths, arm_masses, opening_angle_deg)

    # ── Antenna physics methods ────────────────────────────────────────────

    def kh(self, freq_mhz: float) -> np.ndarray:
        """Electrical half-lengths kh_d = π L_d f / c for each dipole (shape (2,))."""
        return np.pi * np.array(self.arm_lengths) * float(freq_mhz) * 1e6 / _C

    def realized_efficiency(self, freq_mhz: float) -> np.ndarray:
        """Per-dipole realised efficiency at *freq_mhz* (shape (2,))."""
        return np.array([
            _realized_eff(L, [freq_mhz],
                          r_loss_ohm=self.r_loss_ohm,
                          z_rx_ohm=self.z_rx_ohm,
                          x_scale=self.x_scale)[0]
            for L in self.arm_lengths
        ])

    def antenna_temperature_K(self, freq_mhz: float) -> np.ndarray:
        """Per-dipole delivered sky temperature at *freq_mhz* (shape (2,))."""
        return self.realized_efficiency(freq_mhz) * float(_gsm_tsky(freq_mhz))

    def receiver_margin_factor(self, freq_mhz: float) -> np.ndarray:
        """Per-dipole 2·Tant / Trx at *freq_mhz* (shape (2,))."""
        return 2.0 * self.antenna_temperature_K(freq_mhz) / self.t_rx

    def sigma_noise(
        self,
        freq_mhz: float,
        delta_nu: float,
        t_integration: float,
        t_gsm_avg: float | None = None,
    ) -> np.ndarray:
        """
        Per-dipole radiometer noise  σ_d = T_sys_d / sqrt(Δν · τ).

        Parameters
        ----------
        freq_mhz : float
            Observing frequency (MHz).
        delta_nu : float
            Channel bandwidth (Hz).
        t_integration : float
            Integration time per sample (s).
        t_gsm_avg : float, optional
            Mean GSM brightness at *freq_mhz* (K).  Defaults to the analytic
            ``gsm_like_tsky_K`` approximation.
        """
        if t_gsm_avg is None:
            t_gsm_avg = float(_gsm_tsky(freq_mhz))
        eta   = self.realized_efficiency(freq_mhz)
        T_sys = eta * t_gsm_avg + self.t_rx
        return T_sys / np.sqrt(delta_nu * t_integration)

    # ── Dynamics method ────────────────────────────────────────────────────

    def simulate(self, L_inertial, t_final: float = 40.0, dt_sim: float = 0.002) -> dict:
        """Torque-free spin simulation using ``self.inertia`` as the body tensor."""
        return simulate_torque_free(self.inertia, L_inertial, t_final, dt_sim)


# ── Observation sub-object ────────────────────────────────────────────────

class Observation:
    """
    Orbital and observing configuration for one BLOOM-21cm campaign.

    Bundles orbit geometry, sky/simulation parameters, and radiometer setup.
    All derived scheduling and spectral quantities are computed on construction.
    """

    def __init__(
        self,
        rot_orbit_vecs: list[np.ndarray],
        orbit_normals_frame: str,
        altitude: float,
        obs_epoch: Time,
        freq: float,
        nside: int,
        t_regolith: float,
        t_sun: float,
        n_days: int,
        n_obs: int,
        band_low_mhz: float,
        band_high_mhz: float,
        nchan: int,
        fixed_spin: bool,
        freq_min_mhz: float,
        freq_max_mhz: float,
        nchan_science: int,
        duty_cycle: float,
        attitude_knowledge_deg: float,
        spin_period_s: float,
    ) -> None:
        # orbit
        self.rot_orbit_vecs      = rot_orbit_vecs
        self.orbit_normals_frame = orbit_normals_frame
        self.altitude            = altitude
        self.obs_epoch           = obs_epoch
        self.n_orbits: int       = len(rot_orbit_vecs)
        self.t_orbital: float    = circular_orbital_period(altitude)
        # sky / simulation truth
        self.freq: float         = freq
        self.freq_mhz: float     = freq / 1e6
        self.nside: int          = nside
        self.npix: int           = healpy.nside2npix(nside)
        self.t_regolith: float   = t_regolith
        self.t_sun: float        = t_sun
        # radiometer schedule — stored parameters
        self.n_days: int             = n_days
        self.n_obs: int              = n_obs
        self.band_low_mhz: float     = band_low_mhz
        self.band_high_mhz: float    = band_high_mhz
        self.nchan: int              = nchan
        self.fixed_spin: bool        = fixed_spin
        # science frequency grid
        self.freq_min_mhz: float     = freq_min_mhz
        self.freq_max_mhz: float     = freq_max_mhz
        self.nchan_science: int      = nchan_science
        # duty cycle and attitude parameters
        self.duty_cycle: float           = duty_cycle
        self.attitude_knowledge_deg: float = attitude_knowledge_deg
        self.spin_period_s: float        = spin_period_s
        # derived — scheduling
        self.n_total: int           = self.n_orbits * n_obs
        self.n_rows: int            = self.n_total * 2
        self.bw_mhz: float          = band_high_mhz - band_low_mhz
        self.delta_nu: float        = (freq_max_mhz - freq_min_mhz) * 1e6 / nchan_science
        self.channel_width_khz: float = self.delta_nu / 1e3
        # derived — integration time
        #   Effective integration time per simulation observation: total on-sky
        #   time (n_days × duty_cycle) divided equally among all observations.
        #   Each simulation "observation" collapses many short physical snapshots
        #   with similar sky orientation into a single noise-averaged sample.
        _n_obs_total = self.n_orbits * n_obs
        self.t_integration: float = (n_days * 86400.0 * duty_cycle) / _n_obs_total
        #   Per-snapshot coherence limit: the beam drifts by attitude_knowledge_deg
        #   in this time due to the spacecraft spin.  t_integration should be much
        #   larger than t_snapshot for the ergodic-averaging model to be valid.
        self.t_snapshot: float = attitude_knowledge_deg / (360.0 / spin_period_s)

    def make_orbits(
        self,
        rot_spin_vec=(0, 0, 1),
        spin_period: float = 0.0,
    ) -> list[LunarOrbit]:
        """
        Create one :class:`LunarOrbit` per orbital plane defined in this
        Observation.

        Parameters
        ----------
        rot_spin_vec : array_like, shape (3,)
            Spin axis in the galactic frame (passed to :class:`LunarOrbit`).
        spin_period : float
            Spacecraft spin period [s].  0 means no spin.

        Returns
        -------
        orbits : list of LunarOrbit
        """
        return [
            LunarOrbit(
                altitude=self.altitude,
                rot_orbit_vec=v,
                rot_spin_vec=rot_spin_vec,
                start_pos=_perp_to(v),
                spin_period=spin_period,
                t0=self.obs_epoch,
            )
            for v in self.rot_orbit_vecs
        ]


# ── OrbiterMission ────────────────────────────────────────────────────────

class OrbiterMission:
    """
    Top-level BLOOM-21cm mission configuration loaded from a YAML file.

    Sub-objects
    -----------
    antenna     : Antenna
        Dipole geometry, receiver parameters, and antenna/dynamics methods.
    observation : Observation
        Orbital planes, sky model parameters, and radiometer schedule.

    Mission-level STM fields (spin_rate_rpm, science band, etc.) are stored
    directly on this object.

    Parameters
    ----------
    path : str
        Path to the BLOOM YAML configuration file (e.g. ``bloom_config.yaml``).
    """

    def __init__(self, path: str) -> None:
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        # ── Build Antenna ──────────────────────────────────────────────────
        sc, ant = raw["spacecraft"], raw["antenna"]
        self.antenna = Antenna(
            arm_lengths       = sc["arm_lengths_m"],
            arm_masses        = sc["arm_masses_kg"],
            opening_angle_deg = sc["opening_angle_deg"],
            l_hat_raw         = sc["l_hat_raw"],
            r_loss_ohm        = ant["r_loss_ohm"],
            z_rx_ohm          = ant["z_rx_ohm"],
            x_scale           = ant["x_scale"],
            t_rx              = ant["t_rx_K"],
        )

        # ── Build Observation (resolve orbit-normal frame first) ───────────
        obs   = raw["observation"]
        _on   = obs["orbit_normals"]
        _frame = _on.get("frame", "equatorial").lower()
        _vecs  = [np.array(v, dtype=float) for v in _on["vectors"]]
        if _frame in ("equatorial", "icrs"):
            rot_orbit_vecs = [ICRS2GAL @ v for v in _vecs]
        elif _frame == "galactic":
            rot_orbit_vecs = _vecs
        else:
            raise ValueError(
                f"orbit_normals.frame {_frame!r} not recognised; "
                "use 'equatorial' or 'galactic'."
            )

        self.observation = Observation(
            rot_orbit_vecs         = rot_orbit_vecs,
            orbit_normals_frame    = _frame,
            altitude               = obs["altitude_m"],
            obs_epoch              = Time(obs["obs_epoch"]),
            freq                   = obs["freq_hz"],
            nside                  = obs["nside"],
            t_regolith             = obs["t_regolith_K"],
            t_sun                  = obs["t_sun_K"],
            n_days                 = obs["n_days"],
            n_obs                  = obs["n_obs_per_orbit"],
            band_low_mhz           = obs["band_low_mhz"],
            band_high_mhz          = obs["band_high_mhz"],
            nchan                  = obs["nchan"],
            fixed_spin             = obs["fixed_spin"],
            freq_min_mhz           = obs["freq_min_mhz"],
            freq_max_mhz           = obs["freq_max_mhz"],
            nchan_science          = obs["nchan_science"],
            duty_cycle             = obs["duty_cycle"],
            attitude_knowledge_deg = obs["attitude_knowledge_deg"],
            spin_period_s          = obs["spin_period_s"],
        )

        # ── Mission / STM fields ───────────────────────────────────────────
        mis = raw["mission"]
        self.spin_rate_rpm: float               = mis["spin_rate_rpm"]
        self.eff_geom_rate_deg_s: float         = mis["eff_geom_rate_deg_s"]
        self.science_band_low_mhz: float        = mis["science_band_low_mhz"]
        self.science_band_high_mhz: float       = mis["science_band_high_mhz"]
        self.mission_duration_days: float       = mis["mission_duration_days"]
        self.synodic_month_days: float          = mis["synodic_month_days"]
        self.modulation_min: float | None       = mis.get("modulation_min")
        self.sky_frac_modulation: float | None  = mis.get("sky_fraction_meeting_modulation")
        self.trx_frac_of_tsky_max: float | None = mis.get("trx_frac_of_tsky_max")

    # ── Convenience bridges ────────────────────────────────────────────────

    @property
    def kh(self) -> np.ndarray:
        """Electrical half-lengths at the configured frequency (delegates to antenna)."""
        return self.antenna.kh(self.observation.freq_mhz)

    def sigma_noise(self, t_gsm_avg: float | None = None) -> np.ndarray:
        """
        Per-dipole radiometer noise at the configured frequency, bandwidth,
        and integration time (delegates to antenna.sigma_noise).
        """
        return self.antenna.sigma_noise(
            self.observation.freq_mhz,
            self.observation.delta_nu,
            self.observation.t_integration,
            t_gsm_avg,
        )

    def __repr__(self) -> str:
        o, a = self.observation, self.antenna
        return (
            f"OrbiterMission(freq={o.freq_mhz:.0f} MHz, nside={o.nside}, "
            f"alt={o.altitude/1e3:.0f} km, n_orbits={o.n_orbits}, "
            f"n_obs={o.n_obs}, Δν={o.channel_width_khz:.2f} kHz, "
            f"τ={o.t_integration} s, L={a.arm_lengths} m, Trx={a.t_rx} K)"
        )


# Backward-compatible alias
SimConfig = OrbiterMission
