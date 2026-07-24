"""Lunar regolith thermal/EM brightness model (Phase 2 realism work).

Companion to :mod:`ephemeris` (real Sun geometry) and :mod:`sources`
(Phase 1 external sources). Where Phase 1 modelled bright *external*
sources (Sun, Earth), Phase 2 models the *regolith itself*: at 50-150 MHz
the lunar surface is not an opaque blackbody at one uniform temperature --
(a) different points on the surface have different insolation history
(day/night, latitude) so brightness varies spatially and with the lunar
day/night cycle (~29.5 d synodic period), and (b) radio emission at these
frequencies originates from some depth below the surface (the regolith is
only weakly lossy), and the physical temperature at that depth is a
damped, phase-lagged version of the surface diurnal cycle -- so LOWER
frequencies (which probe deeper) should show a SMALLER, more phase-shifted
diurnal brightness swing than higher frequencies. That frequency-dependent
spectral structure is exactly the kind of subtle systematic that could
bias or mimic a 21cm global-signal trough if unmodelled -- this module
exists to quantify it, not to claim photometric-grade absolute accuracy.

Physical constants used (each independently citable, not fit as a set):

- Bond albedo ``bond_albedo=0.12``: standard lunar value, sets the
  subsolar equilibrium temperature via simple radiative balance
  (``subsolar_equilibrium_temperature_K``); with the solar constant this
  gives ~381 K, close to the commonly cited Diviner subsolar temperature
  ~387-400 K (the residual is emissivity<1, non-Lambertian scattering,
  etc., not modelled here).
- Deep isotherm ``T_deep_K=255``: the near-equatorial subsurface
  temperature (depth >~0.5-1 m) where diurnal forcing is fully damped and
  the interior heat flow dominates, ~250-260 K (Hayne et al. 2017,
  "Global Regolith Thermophysical Properties of the Moon from the Diviner
  Lunar Radiometer Experiment", JGR Planets; Vasavada et al.-type thermal
  models report similar values). This is also, self-consistently, close
  to what a symmetric day/night average of a ~387 K subsolar peak and a
  ~100 K nightside floor gives (~243 K) -- the classic periodic-heat-
  equation solution used below preserves the *mean* of the surface
  forcing at all depths, so using the empirical deep isotherm directly is
  both better-grounded and internally consistent.
- Thermal diffusivity ``thermal_diffusivity_m2_s=1e-9``: upper lunar
  regolith value (Hayne et al. 2017); combined with the ~29.5 d synodic
  period via the classic periodic-heat-equation skin depth
  ``delta = sqrt(diffusivity * period / pi)`` this gives ~3 cm, order-of-
  magnitude consistent with the ~4-10 cm diurnal skin depth reported
  there (real regolith diffusivity increases with depth/compaction, not
  captured by this single-diffusivity model).
- Dielectric constant ``eps_r=3.0`` / loss tangent ``loss_tangent=0.005``:
  representative of published in-situ/lab lunar regolith measurements
  (permittivity ~2.6-3.9, loss tangent ~0.003-0.013; e.g. Chang'E-5
  Lunar Penetrating Radar results, Siegler et al. 2020 JGR Planets). These
  were measured at 450 MHz-37 GHz; extrapolating the standard low-loss-
  dielectric penetration-depth formula down to 50-150 MHz assumes the loss
  tangent stays roughly frequency-independent, which lab studies broadly
  support for the *real* part of the dielectric constant, while the
  *imaginary* part (and hence loss tangent) is reported to decrease
  somewhat with INCREASING frequency -- so if anything this likely
  UNDERESTIMATES the true low-frequency penetration depth. Treat
  ``em_power_penetration_depth_m`` as order-of-magnitude, not a validated
  spectrum (same caveat level as ``sources.quiet_sun_temperature_K``).

Depth-averaging derivation (``regolith_brightness_temperature_K``): model
the surface diurnal forcing as a single sinusoid (fundamental harmonic)
T(0, phase) = T_deep + dT*cos(phase), dT = T_subsolar - T_deep. This loses
the true nonlinear day-side T^(1/4) lightcurve shape, but higher harmonics
of a periodic surface forcing damp with depth faster than the fundamental
(damping length for harmonic n scales as 1/sqrt(n)), so a depth-averaged
brightness is increasingly well approximated by the fundamental alone as
depth grows -- exactly the regime this module targets. The classic
solution to the 1-D heat equation under sinusoidal forcing is
T(z, phase) = T_deep + dT * exp(-z/d_th) * cos(phase - z/d_th), where
d_th is the diurnal thermal skin depth. Averaging this against the EM
power-deposition kernel w(z) = exp(-z/d_em) (d_em = the power e-folding
penetration depth -- for a passive lossy half-space, the emergent
brightness is a depth average of the physical temperature weighted by
where the observed power actually originates) gives, via
Re[exp(i*phase) * integral of exp(-z*(1/d_th + i/d_th + 1/d_em)) dz]:

    T_eff(phase) = T_deep + dT * amp(u) * cos(phase - shift(u))

with u = d_th/d_em, amp(u) = u/sqrt((1+u)^2+1), shift(u) = atan2(1, 1+u).
As u -> infinity (d_em << d_th, i.e. very high frequency / shallow depth),
amp -> 1 and shift -> 0: recovers the undamped surface fundamental. As
u -> 0 (d_em >> d_th, very low frequency / deep), amp -> 0: recovers the
deep isotherm. A correction term (see the function body) is added so the
high-frequency limit is EXACT (matches the true nonlinear
``surface_equilibrium_temperature_K``, not just its fundamental harmonic),
damped by the same amp(u) envelope -- conservative in that real higher
harmonics damp at least as fast, so this likely slightly OVERSTATES how
long daytime peak sharpness survives with depth, not understates it.

This is a sensitivity-study model, not a validated forward model of real
regolith emission: it deliberately isolates the two qualitative effects
that matter for the science question (frequency-dependent amplitude
damping and phase lag of the diurnal brightness swing) rather than trying
to reproduce an exact lightcurve.
"""

from __future__ import annotations

import numpy as np

from .const import c as C_LIGHT

SOLAR_CONSTANT_W_M2 = 1361.0
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
LUNAR_SYNODIC_DAY_S = 29.530589 * 86400.0


def subsolar_equilibrium_temperature_K(bond_albedo=0.12, solar_dist_au=1.0):
    """Blackbody radiative-equilibrium temperature at the sub-solar point.

    ``T_ss = [(1 - bond_albedo) * S0 / (solar_dist_au**2 * sigma)] ** 0.25``.
    See the module docstring for the ~381 K vs. ~387-400 K comparison.
    """
    S = SOLAR_CONSTANT_W_M2 / float(solar_dist_au) ** 2
    return ((1.0 - bond_albedo) * S / STEFAN_BOLTZMANN_W_M2_K4) ** 0.25


def surface_equilibrium_temperature_K(
    cos_zenith, bond_albedo=0.12, T_night_K=100.0, solar_dist_au=1.0
):
    """Instantaneous (z=0, infinite-frequency-limit) surface temperature.

    Day side: fast-rotator / negligible-thermal-inertia limit,
    ``T = T_subsolar * cos_zenith**0.25`` (valid for the Moon's very low
    SURFACE thermal inertia, though real Diviner data show a morning/
    evening asymmetry from finite inertia this ignores). Night side: a
    constant floor ``T_night_K`` (real nightside cooling is a slow
    exponential relaxation set by the same low thermal inertia; a floor
    is a coarse but adequate approximation here).

    Parameters
    ----------
    cos_zenith : array_like
        cos(solar zenith angle); any sign, <=0 treated as night.

    Returns
    -------
    T_K : ndarray, same shape as ``cos_zenith``
    """
    cos_zenith = np.asarray(cos_zenith, dtype=np.float64)
    T_ss = subsolar_equilibrium_temperature_K(bond_albedo, solar_dist_au)
    cz = np.clip(cos_zenith, 0.0, None)
    T_day = T_ss * cz**0.25
    return np.where(cos_zenith > 0, np.maximum(T_day, T_night_K), T_night_K)


def solar_geometry(pix_vecs, sun_dir):
    """cos(solar zenith angle) and local diurnal phase for surface points.

    Parameters
    ----------
    pix_vecs : ndarray, shape (..., 3)
        Unit "up"/outward-normal vectors of surface points, in the same
        frame as ``sun_dir`` (typically MCMF, body-fixed -- see
        :func:`eigsep_sim.observer._moon_icrs2mcmf`).
    sun_dir : ndarray, shape (..., 3) broadcastable with ``pix_vecs``
        Unit vector toward the Sun, same frame.

    Returns
    -------
    cos_zenith : ndarray
    phase_rad : ndarray
        Local hour angle from solar noon (0 = noon, +-pi = midnight),
        derived from the longitude difference about the frame's polar
        (z) axis. Exact for ``cos_zenith`` (a plain dot product); the
        longitude-difference shortcut for ``phase_rad`` neglects the
        Moon's ~1.5 deg obliquity to its orbit (negligible at this
        model's precision level).
    """
    pix_vecs = np.asarray(pix_vecs, dtype=np.float64)
    sun_dir = np.asarray(sun_dir, dtype=np.float64)
    cos_zenith = np.sum(pix_vecs * sun_dir, axis=-1)
    lon = np.arctan2(pix_vecs[..., 1], pix_vecs[..., 0])
    lon_sun = np.arctan2(sun_dir[..., 1], sun_dir[..., 0])
    phase = np.mod(lon - lon_sun + np.pi, 2 * np.pi) - np.pi
    return cos_zenith, phase


def em_power_penetration_depth_m(freqs_hz, eps_r=3.0, loss_tangent=0.005):
    """Power (intensity) e-folding penetration depth [m] in a low-loss
    dielectric half-space: ``lambda_0 / (2*pi*sqrt(eps_r)*loss_tangent)``.

    See the module docstring for the regolith parameter grounding and the
    extrapolation caveat from the measured 450 MHz-37 GHz range down to
    50-150 MHz.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    wavelength_m = C_LIGHT / freqs_hz
    return wavelength_m / (2.0 * np.pi * np.sqrt(eps_r) * loss_tangent)


def diurnal_thermal_skin_depth_m(thermal_diffusivity_m2_s=1e-9, period_s=None):
    """Thermal skin depth [m] for a periodic surface temperature forcing:
    ``sqrt(diffusivity * period / pi)``. Defaults match the upper lunar
    regolith and the synodic lunar day -- see the module docstring.
    """
    if period_s is None:
        period_s = LUNAR_SYNODIC_DAY_S
    return np.sqrt(thermal_diffusivity_m2_s * period_s / np.pi)


def regolith_brightness_temperature_K(
    cos_zenith,
    phase_rad,
    freqs_hz,
    bond_albedo=0.12,
    T_deep_K=255.0,
    thermal_diffusivity_m2_s=1e-9,
    eps_r=3.0,
    loss_tangent=0.005,
):
    """Frequency-dependent regolith brightness temperature [K].

    See the module docstring for the full derivation. Exact at both
    limits: as EM depth -> 0 (high frequency) this equals
    :func:`surface_equilibrium_temperature_K`; as EM depth -> infinity
    (low frequency / deep) this equals ``T_deep_K``.

    Parameters
    ----------
    cos_zenith, phase_rad : array_like, broadcastable
        From :func:`solar_geometry`.
    freqs_hz : array_like, shape (nfreq,)

    Returns
    -------
    T_b : ndarray, shape (*broadcast(cos_zenith, phase_rad).shape, nfreq)
    """
    cos_zenith, phase_rad = np.broadcast_arrays(
        np.asarray(cos_zenith, dtype=np.float64),
        np.asarray(phase_rad, dtype=np.float64),
    )
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    T_ss = subsolar_equilibrium_temperature_K(bond_albedo)
    dT = T_ss - T_deep_K

    delta_th = diurnal_thermal_skin_depth_m(thermal_diffusivity_m2_s)
    delta_em = em_power_penetration_depth_m(freqs_hz, eps_r, loss_tangent)  # (nfreq,)
    u = delta_th / delta_em  # (nfreq,) -- small u = deep/low-freq, large u = shallow/high-freq

    amp = u / np.sqrt((1.0 + u) ** 2 + 1.0)  # (nfreq,)
    shift = np.arctan2(1.0, 1.0 + u)  # (nfreq,)

    fundamental = dT * amp * np.cos(phase_rad[..., None] - shift)  # (..., nfreq)

    surface_exact = surface_equilibrium_temperature_K(cos_zenith, bond_albedo)  # (...,)
    surface_fundamental_z0 = dT * np.cos(phase_rad)  # (...,)
    correction = surface_exact - T_deep_K - surface_fundamental_z0  # (...,)

    return T_deep_K + fundamental + correction[..., None] * amp
