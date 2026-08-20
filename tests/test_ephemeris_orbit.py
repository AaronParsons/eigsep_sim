#!/usr/bin/env python
"""
Test suite for observer.EphemerisLunarOrbit — position interpolation, period
estimation, and occultation geometry against a known synthetic trajectory.
"""

import numpy as np
import pytest
import healpy
from astropy.time import Time
import astropy.units as u

from eigsep_sim.observer import CircularLunarOrbit, EphemerisLunarOrbit
from eigsep_sim.const import R_MOON


def _synthetic_circular_ephemeris(altitude=100e3, n=400, t0=None):
    """Sample a CircularLunarOrbit's own trajectory to build a ground-truth
    (times, positions_m) table -- lets EphemerisLunarOrbit's interpolation
    be checked against a known-exact answer."""
    t0 = Time("2030-01-01") if t0 is None else t0
    truth = CircularLunarOrbit(altitude, [0, 0, 1], [0, 0, 1], t0=t0)
    times = t0 + np.linspace(0.0, 4.0, n) * u.hr
    positions_m = truth.spacecraft_position_stack(times)
    return truth, times, positions_m


def _synthetic_eccentric_ephemeris(a=R_MOON + 120e3, e=0.02, period_s=7000.0,
                                    n_per_orbit=25, n_orbits=6, t0=None):
    """Analytic two-body Keplerian ellipse (in-plane, z=0 -- no physical
    frame meaning, just a clean radius(t) with well-defined periapsis
    passages) -- gives a known-exact period and eccentricity to check
    EphemerisLunarOrbit's periapsis-based period estimator against, unlike
    the degenerate (constant-radius) circular case."""
    t0 = Time("2030-01-01") if t0 is None else t0
    dt = np.linspace(0.0, n_orbits * period_s, n_orbits * n_per_orbit,
                      endpoint=False)
    mean_anomaly = 2 * np.pi * dt / period_s
    ecc_anomaly = mean_anomaly.copy()
    for _ in range(50):
        ecc_anomaly -= (
            (ecc_anomaly - e * np.sin(ecc_anomaly) - mean_anomaly)
            / (1 - e * np.cos(ecc_anomaly))
        )
    x = a * (np.cos(ecc_anomaly) - e)
    y = a * np.sqrt(1 - e ** 2) * np.sin(ecc_anomaly)
    positions_m = np.column_stack([x, y, np.zeros_like(x)])
    times = t0 + dt * u.s
    return times, positions_m, period_s


def test_ephemeris_orbit_interpolation_matches_truth():
    """EphemerisLunarOrbit: interpolated position matches the sampled
    trajectory at held-out (non-grid) query times."""
    truth, times, positions_m = _synthetic_circular_ephemeris()
    orbit = EphemerisLunarOrbit(times, positions_m, [0, 0, 1])

    query_times = times[0] + np.array([37.0, 613.0, 1801.0]) * u.s
    expected = truth.spacecraft_position_stack(query_times)
    got = orbit.spacecraft_position_stack(query_times)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1.0)


def test_ephemeris_orbit_altitude_and_radius():
    """EphemerisLunarOrbit: mean altitude/radius match the (constant, for
    this synthetic case) truth."""
    _, times, positions_m = _synthetic_circular_ephemeris(altitude=100e3)
    orbit = EphemerisLunarOrbit(times, positions_m, [0, 0, 1])

    assert orbit.orbital_radius == pytest.approx(R_MOON + 100e3, rel=1e-6)
    assert orbit.altitude == pytest.approx(100e3, rel=1e-6)


def test_ephemeris_orbit_period_estimate():
    """EphemerisLunarOrbit: period estimated from periapsis spacing recovers
    the known period of an eccentric (non-degenerate radius) synthetic
    orbit to within a percent."""
    times, positions_m, true_period_s = _synthetic_eccentric_ephemeris()
    orbit = EphemerisLunarOrbit(times, positions_m, [0, 0, 1])

    assert orbit.orbital_period == pytest.approx(true_period_s, rel=1e-2)


def test_ephemeris_orbit_above_horizon_occultation():
    """EphemerisLunarOrbit: low orbit, Moon blocks some but not all sky --
    same sanity band as CircularLunarOrbit at the same altitude."""
    _, times, positions_m = _synthetic_circular_ephemeris(altitude=100e3)
    orbit = EphemerisLunarOrbit(times, positions_m, [0, 0, 1])
    orbit.set_time(times[len(times) // 2])

    mask = orbit.above_horizon(nside=4)
    npix = healpy.nside2npix(4)
    assert np.sum(mask) < npix
    assert np.sum(mask) > 0.5 * npix


def test_ephemeris_orbit_above_horizon_stack_matches_scalar():
    """EphemerisLunarOrbit: batched occultation masks match scalar masks."""
    _, times, positions_m = _synthetic_circular_ephemeris(altitude=100e3)
    orbit = EphemerisLunarOrbit(times, positions_m, [0, 0, 1])

    query_times = times[10:13]
    mask_stack = orbit.above_horizon_stack(query_times, nside=4)
    mask_loop = []
    for t in query_times:
        orbit.set_time(t)
        mask_loop.append(orbit.above_horizon(nside=4))

    np.testing.assert_array_equal(mask_stack, np.stack(mask_loop))


def test_ephemeris_orbit_out_of_bounds_raises():
    """EphemerisLunarOrbit: querying outside the loaded span raises
    ValueError instead of silently extrapolating."""
    _, times, positions_m = _synthetic_circular_ephemeris()
    orbit = EphemerisLunarOrbit(times, positions_m, [0, 0, 1])

    before = times[0] - 1.0 * u.hr
    after = times[-1] + 1.0 * u.hr
    with pytest.raises(ValueError):
        orbit.spacecraft_position_stack(before)
    with pytest.raises(ValueError):
        orbit.spacecraft_position_stack(after)

    orbit.set_time(after)
    with pytest.raises(ValueError):
        orbit.spacecraft_position()


def test_lunar_orbit_alias_is_circular_lunar_orbit():
    """LunarOrbit is CircularLunarOrbit (backward-compat alias)."""
    from eigsep_sim.observer import LunarOrbit

    assert LunarOrbit is CircularLunarOrbit
