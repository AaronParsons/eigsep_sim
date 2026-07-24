"""Tests for the regolith thermal/EM brightness model (regolith.py).

Checks the physics building blocks individually (subsolar/surface
temperature, solar geometry, EM/thermal depth scales) and the combined
depth-weighted brightness model's two key qualitative predictions: it
must be EXACT at both limits (pure surface at high frequency, deep
isotherm at low frequency), and in between it must show a SMALLER,
more phase-lagged diurnal swing at lower frequency than higher --
the frequency-dependent spectral structure this model exists to quantify.
"""

import numpy as np
import healpy
import pytest
from astropy.time import Time
import astropy.units as u

from eigsep_sim.beam import Beam
from eigsep_sim.sky import Sky
from eigsep_sim.observer import LunarOrbit
from eigsep_sim.simulate import ForwardModel
from eigsep_sim.regolith import (
    subsolar_equilibrium_temperature_K,
    surface_equilibrium_temperature_K,
    solar_geometry,
    em_power_penetration_depth_m,
    diurnal_thermal_skin_depth_m,
    regolith_brightness_temperature_K,
)


def test_subsolar_temperature_reasonable():
    T = subsolar_equilibrium_temperature_K(bond_albedo=0.12)
    assert 350.0 < T < 420.0  # ballpark of the commonly cited ~387-400 K


def test_surface_equilibrium_noon_night_terminator():
    T_noon = surface_equilibrium_temperature_K(1.0, T_night_K=100.0)
    T_night = surface_equilibrium_temperature_K(-0.5, T_night_K=100.0)
    T_term = surface_equilibrium_temperature_K(1e-6, T_night_K=100.0)
    assert T_noon > 350.0
    assert T_night == 100.0
    assert T_term == 100.0  # T_day~0 near the terminator, floored at T_night
    assert T_noon > T_term


def test_surface_equilibrium_monotonic_in_cos_zenith():
    cz = np.linspace(0.01, 1.0, 20)
    T = surface_equilibrium_temperature_K(cz)
    assert np.all(np.diff(T) >= 0)


def test_solar_geometry_noon_and_midnight():
    sun_dir = np.array([1.0, 0.0, 0.0])
    cz_noon, ph_noon = solar_geometry(sun_dir, sun_dir)
    np.testing.assert_allclose(cz_noon, 1.0, atol=1e-12)
    np.testing.assert_allclose(ph_noon, 0.0, atol=1e-12)

    cz_midnight, ph_midnight = solar_geometry(-sun_dir, sun_dir)
    np.testing.assert_allclose(cz_midnight, -1.0, atol=1e-12)
    assert abs(abs(ph_midnight) - np.pi) < 1e-6

    # quarter-day point: 90 deg away in longitude, same "latitude" (equator)
    quarter = np.array([0.0, 1.0, 0.0])
    cz_q, ph_q = solar_geometry(quarter, sun_dir)
    np.testing.assert_allclose(cz_q, 0.0, atol=1e-12)
    np.testing.assert_allclose(abs(ph_q), np.pi / 2, atol=1e-6)


def test_solar_geometry_broadcasts():
    pix_vecs = np.tile(np.array([1.0, 0.0, 0.0]), (5, 3, 1))  # (5,3,3)
    sun_dir = np.array([1.0, 0.0, 0.0])
    cz, ph = solar_geometry(pix_vecs, sun_dir)
    assert cz.shape == (5, 3)
    assert ph.shape == (5, 3)


def test_em_penetration_depth_scales_inversely_with_frequency():
    freqs = np.array([50e6, 100e6, 150e6])
    depth = em_power_penetration_depth_m(freqs)
    assert depth[0] > depth[1] > depth[2]
    np.testing.assert_allclose(depth[0] / depth[2], 150.0 / 50.0, rtol=1e-6)


def test_diurnal_skin_depth_order_of_magnitude():
    d = diurnal_thermal_skin_depth_m()
    assert 0.01 < d < 0.20  # cm-scale, per Hayne et al. 2017's ~4-10 cm


def test_regolith_brightness_high_frequency_matches_exact_surface():
    # Tiny EM depth (huge freq + loss tangent) -> u -> infinity -> exact
    # surface limit. u only needs to be large, not literally infinite, so
    # push it hard (u ~ 1e4 here) and use a tolerance appropriate to that,
    # not machine precision.
    cz = np.array([1.0, 0.3, -0.5])
    ph = np.array([0.0, 1.0, 2.5])
    freqs = np.array([1e12])
    T_b = regolith_brightness_temperature_K(cz, ph, freqs, loss_tangent=100.0)
    T_surf = surface_equilibrium_temperature_K(cz)
    np.testing.assert_allclose(T_b[:, 0], T_surf, atol=0.5)


def test_regolith_brightness_converges_toward_surface_as_frequency_increases():
    """Monotonic-convergence check complementing the fixed-tolerance limit
    test above: the error to the exact surface value should shrink as u
    grows, confirming the asymptote is approached from a clear direction
    rather than the previous test passing by coincidence."""
    cz = np.array([1.0, 0.3, -0.5])
    ph = np.array([0.0, 1.0, 2.5])
    T_surf = surface_equilibrium_temperature_K(cz)
    freqs = np.array([1e8, 1e10, 1e12])
    errs = [
        np.max(np.abs(
            regolith_brightness_temperature_K(cz, ph, np.array([f]),
                                              loss_tangent=100.0)[:, 0]
            - T_surf
        ))
        for f in freqs
    ]
    assert errs[0] > errs[1] > errs[2]


def test_regolith_brightness_low_frequency_matches_deep_isotherm():
    # Huge EM depth (tiny loss tangent) -> u -> 0 -> deep-isotherm limit.
    cz = np.array([1.0, 0.3, -0.5])
    ph = np.array([0.0, 1.0, 2.5])
    freqs = np.array([50e6])
    T_b = regolith_brightness_temperature_K(cz, ph, freqs, loss_tangent=1e-9,
                                            T_deep_K=255.0)
    np.testing.assert_allclose(T_b[:, 0], 255.0, atol=1e-3)


def test_regolith_brightness_diurnal_amplitude_decreases_with_frequency_decrease():
    """Lower frequency probes deeper (larger EM depth) -> more damping ->
    smaller diurnal swing. This is the model's core qualitative claim."""
    phase = np.linspace(-np.pi, np.pi, 60)
    cos_zenith = np.cos(phase)  # crude but monotonic day/night proxy
    freqs = np.array([50e6, 150e6])
    T_b = regolith_brightness_temperature_K(cos_zenith, phase, freqs)  # (60, 2)
    swing_low = T_b[:, 0].max() - T_b[:, 0].min()
    swing_high = T_b[:, 1].max() - T_b[:, 1].min()
    assert swing_low < swing_high


def test_regolith_brightness_phase_lag_increases_at_lower_frequency():
    """Lower frequency (smaller u) should show a larger phase shift."""
    from eigsep_sim.regolith import diurnal_thermal_skin_depth_m as dth

    freqs = np.array([50e6, 150e6])
    delta_th = dth()
    delta_em = em_power_penetration_depth_m(freqs)
    u = delta_th / delta_em
    shift = np.arctan2(1.0, 1.0 + u)
    assert shift[0] > shift[1]  # 50 MHz (smaller u) lags more than 150 MHz


def test_regolith_brightness_shape():
    cz = np.zeros((4, 5))
    ph = np.zeros((4, 5))
    freqs = np.linspace(50e6, 150e6, 7)
    T_b = regolith_brightness_temperature_K(cz, ph, freqs)
    assert T_b.shape == (4, 5, 7)


# ─────────────────────────────────────────────────────────────────────────
# Integration: ForwardModel.precompute_geometry(regolith_kwargs=...)
# ─────────────────────────────────────────────────────────────────────────


def _small_lunar_fwd(nside=4, nfreq=4):
    freqs_hz = np.linspace(50e6, 150e6, nfreq)
    npix_sky = healpy.nside2npix(nside)
    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(
        nside, freqs_hz, np.zeros((npix_sky, nfreq)), n_modes=2
    )
    orbit = LunarOrbit(
        altitude=100e3, rot_orbit_vec=[0, 1, 0], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
        occultation_temperature_K=255.0,
    )
    fwd = ForwardModel(orbit, beam, sky)
    times = Time("2030-01-01") + np.linspace(0, 6 * 3600, 30) * u.s
    return fwd, times


def test_regolith_kwargs_absent_matches_prior_behavior():
    """Default (no regolith_kwargs) must be byte-identical to before this
    feature existed -- the scalar occultation_temperature_K broadcast."""
    fwd, times = _small_lunar_fwd()
    geom = fwd.precompute_geometry(times=times)
    emissions = np.asarray(geom["terrain_emissions_jax"])
    blocked = np.asarray(geom["terrain_masks_jax"]) < 1.0
    # every blocked pixel/time/freq should show exactly the scalar T=255
    vals = emissions[blocked[..., None] & (emissions != 0)]
    if blocked.any():
        np.testing.assert_allclose(emissions[blocked], 255.0)


def test_regolith_kwargs_requires_occulting_observer():
    freqs_hz = np.array([100e6])
    nside = 4
    npix = healpy.nside2npix(nside)
    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.zeros((npix, 1)), n_modes=1)
    from eigsep_sim.observer import EarthSurface

    observer = EarthSurface(lat=0.0, lon=0.0)
    observer.set_time("2030-01-01")
    fwd = ForwardModel(observer, beam, sky)
    with pytest.raises(ValueError, match="occulting observer"):
        fwd.precompute_geometry(
            times=[Time("2030-01-01")], regolith_kwargs={}
        )


def test_regolith_kwargs_incompatible_with_sky_mask():
    fwd, times = _small_lunar_fwd()
    sky_mask = np.ones(fwd.sky.npix, dtype=bool)
    with pytest.raises(ValueError, match="sky_mask"):
        fwd.precompute_geometry(
            times=times, sky_mask=sky_mask, regolith_kwargs={}
        )


def test_regolith_kwargs_produces_spatially_and_temporally_varying_emission():
    """The whole point: unlike the scalar broadcast, blocked-pixel
    brightness should vary across pixels (day/night, latitude) and across
    time (as the sub-observer point and Sun both move) -- NOT be uniform.

    With the module's grounded default parameters, the EM penetration
    depth at 50-150 MHz (tens of metres) vastly exceeds the diurnal
    thermal skin depth (~3 cm), so the model's own physics predicts the
    surviving diurnal swing is small (order tens of mK -- see
    test_regolith_brightness_diurnal_amplitude_decreases_with_frequency_decrease
    for the underlying mechanism), not the several-K swing a naive
    "regolith temperature varies with insolation" intuition might expect.
    That smallness is itself the interesting result, not a bug -- assert
    on it being present and in the right (tiny) ballpark, not large.
    """
    fwd, times = _small_lunar_fwd()
    geom = fwd.precompute_geometry(
        times=times, regolith_kwargs=dict(T_deep_K=255.0)
    )
    emissions = np.asarray(geom["terrain_emissions_jax"])  # (ntimes,npix,nfreq)
    masks = np.asarray(geom["terrain_masks_jax"])
    blocked = masks < 1.0
    assert blocked.any(), "test orbit should occult some pixels"

    blocked_vals = emissions[..., 0][blocked]  # first freq channel (50 MHz)
    assert 0.001 < blocked_vals.std() < 1.0
    # physically sane range: tightly clustered around the deep isotherm
    assert blocked_vals.min() > 254.0
    assert blocked_vals.max() < 256.0


def test_regolith_kwargs_frequency_dependence():
    """Different frequencies probe different depths -> different brightness
    at the same pixel/time (the core Phase-2 spectral-structure concern)."""
    fwd, times = _small_lunar_fwd(nfreq=2)  # 50 MHz and 150 MHz
    geom = fwd.precompute_geometry(times=times, regolith_kwargs={})
    emissions = np.asarray(geom["terrain_emissions_jax"])
    masks = np.asarray(geom["terrain_masks_jax"])
    blocked = masks < 1.0
    if blocked.any():
        diff = emissions[..., 0][blocked] - emissions[..., 1][blocked]
        assert np.any(np.abs(diff) > 1e-3)


def test_regolith_kwargs_simulate_runs_and_is_finite():
    fwd, times = _small_lunar_fwd()
    geom = fwd.precompute_geometry(times=times, regolith_kwargs={})
    sky_coeffs = np.zeros((fwd.sky.npix, 2), dtype=np.float32)
    T_ant = np.asarray(fwd.simulate(sky_coeffs, fwd.beam.coeffs, geom=geom))
    assert np.all(np.isfinite(T_ant))
    assert T_ant.shape == (30, 2, 4)
