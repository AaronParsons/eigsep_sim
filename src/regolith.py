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


def regolith_reflectivity(freqs_hz, eps_r=3.0, loss_tangent=0.005):
    """Normal-incidence power reflectivity of the regolith half-space [-].

    ``R = |(n-1)/(n+1)|**2`` with complex refractive index
    ``n = sqrt(eps_r * (1 - 1j*loss_tangent))``, using the SAME dielectric
    parameters already grounded for :func:`em_power_penetration_depth_m`
    (Phase 2) rather than introducing new, independently-tuned numbers.
    With the default ``eps_r=3.0``, ``loss_tangent=0.005`` this gives
    ``R ~= 0.072``, nearly frequency-independent (the loss tangent's
    contribution is a small O(loss_tangent^2) correction to the dominant
    real-index reflectivity) -- consistent with a fixed, non-dispersive
    dielectric model. This is a single-interface, smooth-surface, normal-
    incidence placeholder: real regolith reflectivity is also a function of
    incidence angle (Fresnel angular dependence) and surface roughness
    (which suppresses coherent/specular reflection at short wavelengths
    relative to the roughness scale); see the module-level notes on
    ``specular_frac`` in :func:`specular_reflection_direction` /
    :func:`lambertian_hemisphere_weights` for why specular is nonetheless
    expected to dominate at these frequencies.

    Parameters
    ----------
    freqs_hz : array_like
        Unused except to broadcast the (currently frequency-independent)
        result to the caller's frequency grid shape; kept as an explicit
        argument so callers don't need special-case this function
        differently from the other spectrum functions in this module, and
        so a future angle/frequency-dependent refinement is a drop-in
        replacement.
    eps_r, loss_tangent : float
        Same dielectric parameters as :func:`em_power_penetration_depth_m`.

    Returns
    -------
    R : ndarray, same shape as ``freqs_hz``
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    n_complex = np.sqrt(complex(eps_r) * (1.0 - 1j * loss_tangent))
    R = np.abs((n_complex - 1.0) / (n_complex + 1.0)) ** 2
    return np.full_like(freqs_hz, R)


def specular_reflection_direction(view_dir, normal):
    """Direction a specularly-reflected ray toward ``view_dir`` came from.

    Standard mirror-image construction: ``source = 2*(view.n)*n - view``.
    Both ``view_dir`` (unit vector from the surface point TOWARD the
    observer) and the returned ``source`` direction (unit vector from the
    surface point toward whatever is being reflected, e.g. a point on the
    sky) point outward from the surface, into the same hemisphere as
    ``normal``.

    At 50-150 MHz (wavelength ~2-5 m), typical lunar regolith surface
    roughness (~mm-cm scale, e.g. Diviner/radar roughness studies) is far
    below the wavelength -- by the Rayleigh smoothness criterion the
    surface behaves as optically SMOOTH at these frequencies, so coherent
    specular reflection, not diffuse (Lambertian) scattering, should
    dominate. Callers mixing this with :func:`lambertian_hemisphere_weights`
    should default ``specular_frac`` close to 1 for that reason, not 0.5.

    Parameters
    ----------
    view_dir : ndarray, shape (..., 3)
        Unit vector(s) from the surface point toward the observer.
    normal : ndarray, shape (..., 3), broadcastable with ``view_dir``
        Unit outward surface normal(s), same frame as ``view_dir``.

    Returns
    -------
    source_dir : ndarray, shape matching ``broadcast(view_dir, normal)``
        Unit vector(s) toward the reflected source direction.
    """
    view_dir = np.asarray(view_dir, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    cos_i = np.sum(view_dir * normal, axis=-1, keepdims=True)
    source = 2.0 * cos_i * normal - view_dir
    return source / np.linalg.norm(source, axis=-1, keepdims=True)


def lambertian_hemisphere_weights(normal, pixel_dirs, pixel_omega):
    """Per-pixel Lambertian (diffuse) reflection weights for a surface point.

    Returns ``w[pixel] = max(0, normal . pixel_dirs[pixel]) * pixel_omega / pi``
    such that, for a sky brightness map ``T_sky`` sampled at ``pixel_dirs``
    (same units/shape convention throughout this codebase: brightness
    temperature, linear in the Rayleigh-Jeans regime used everywhere else
    here), the diffusely-reflected brightness temperature is
    ``R * (w @ T_sky)`` for reflectivity ``R``. Normalised so a spatially
    UNIFORM sky of brightness ``T0`` reflects to exactly ``R * T0``
    (``integral of cos(theta) over the hemisphere = pi``, the standard
    Lambertian identity) -- a useful energy-conservation sanity check.

    Parameters
    ----------
    normal : ndarray, shape (3,)
        Unit outward surface normal, in the same frame as ``pixel_dirs``.
    pixel_dirs : ndarray, shape (npix, 3)
        Unit vectors of the sky pixel grid (or a reduced set of spatial
        eigenmode-defining pixels) to integrate over.
    pixel_omega : float or ndarray, shape (npix,)
        Solid angle per pixel [sr] (e.g. ``healpy.nside2pixarea(nside)``
        for a uniform HEALPix grid).

    Returns
    -------
    weights : ndarray, shape (npix,)
    """
    normal = np.asarray(normal, dtype=np.float64)
    pixel_dirs = np.asarray(pixel_dirs, dtype=np.float64)
    cos_i = np.clip(pixel_dirs @ normal, 0.0, None)
    return cos_i * np.asarray(pixel_omega) / np.pi


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
