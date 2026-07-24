#!/usr/bin/env python
"""
Test suite for observer.py — Earth surface, lunar surface, and orbital observers.

Tests verify:
1. Observer ABC basic functionality
2. EarthSurface rotation and horizon masks
3. LunarSurface rotation and horizon masks
4. LunarOrbit orbital mechanics and occultation
"""

import numpy as np
import pytest
import healpy
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body_barycentric
import astropy.units as u

from eigsep_sim.observer import (
    Observer, EarthSurface, LunarSurface, LunarOrbit,
    circular_orbital_period, ICRS2GAL, _moon_icrs2mcmf
)
from eigsep_sim.const import R_MOON, GM_MOON


def test_observer_abstract():
    """Observer: ABC cannot be instantiated."""
    obs = Observer()
    with pytest.raises(NotImplementedError):
        obs.rot_gal2top()
    with pytest.raises(NotImplementedError):
        obs.above_horizon(nside=4)


def test_icrs2gal_matrix():
    """ICRS2GAL: rotation matrix is valid."""
    # Check it's a 3x3 matrix
    assert ICRS2GAL.shape == (3, 3)
    # Check it's orthogonal (R^T @ R = I)
    identity = ICRS2GAL.T @ ICRS2GAL
    assert np.allclose(identity, np.eye(3))
    # Check determinant is 1 (proper rotation, not reflection)
    assert np.isclose(np.linalg.det(ICRS2GAL), 1.0)


def test_earth_surface_basic():
    """EarthSurface: basic construction at known location."""
    lat = 45.0  # degrees
    lon = 120.0  # degrees
    height = 1000.0  # meters

    obs = EarthSurface(lat, lon, height)

    assert np.isclose(obs.location.lat.deg, lat)
    assert np.isclose(obs.location.lon.deg, lon)
    assert np.isclose(obs.location.height.to(u.m).value, height)


def test_earth_surface_time():
    """EarthSurface: set_time updates internal state."""
    obs = EarthSurface(0, 0)
    assert obs.time is None

    t = Time("2000-01-01")
    obs.set_time(t)

    assert obs.time is not None
    assert obs.time.iso == "2000-01-01 00:00:00.000"


def test_earth_surface_rotation():
    """EarthSurface: rot_gal2top returns valid rotation matrix."""
    obs = EarthSurface(0, 0)
    obs.set_time("2000-01-01")

    R = obs.rot_gal2top()

    # Check shape
    assert R.shape == (3, 3)
    # Check orthogonal
    identity = R.T @ R
    assert np.allclose(identity, np.eye(3), atol=1e-10)
    # Check determinant
    assert np.isclose(np.linalg.det(R), 1.0)


def test_earth_surface_horizon():
    """EarthSurface: above_horizon returns bool mask."""
    obs = EarthSurface(45.0, 0.0)
    obs.set_time("2000-01-01")

    mask = obs.above_horizon(nside=4)

    npix = healpy.nside2npix(4)
    assert mask.shape == (npix,)
    assert mask.dtype == bool
    # At 45° latitude, roughly half the sky should be visible
    assert 0.3 < np.sum(mask) / npix < 0.7


def test_earth_surface_horizon_equator():
    """EarthSurface: at equator, roughly half sky visible."""
    obs = EarthSurface(lat=0.0, lon=0.0)
    obs.set_time("2000-01-01")

    mask = obs.above_horizon(nside=4)
    frac_visible = np.sum(mask) / len(mask)

    # At equator, about half sky should be visible
    assert 0.45 < frac_visible < 0.55


def test_earth_surface_horizon_pole():
    """EarthSurface: at pole, hemisphere visible."""
    obs = EarthSurface(lat=90.0, lon=0.0)
    obs.set_time("2000-01-01")

    mask = obs.above_horizon(nside=4)
    frac_visible = np.sum(mask) / len(mask)

    # At north pole, roughly hemisphere should be visible
    assert 0.4 < frac_visible < 0.6


def test_lunar_surface_basic():
    """LunarSurface: basic construction at known location."""
    lat = 0.0  # degrees
    lon = 0.0  # degrees

    obs = LunarSurface(lat, lon)

    assert obs.lat_rad == np.deg2rad(lat)
    assert obs.lon_rad == np.deg2rad(lon)


def test_lunar_surface_time():
    """LunarSurface: set_time updates internal state."""
    obs = LunarSurface(0, 0)
    assert obs.time is None

    t = Time("2000-01-01")
    obs.set_time(t)

    assert obs.time is not None


def test_lunar_surface_rotation():
    """LunarSurface: rot_gal2top returns valid rotation matrix."""
    obs = LunarSurface(0, 0)
    obs.set_time("2000-01-01")

    R = obs.rot_gal2top()

    # Check shape
    assert R.shape == (3, 3)
    # Check orthogonal
    identity = R.T @ R
    assert np.allclose(identity, np.eye(3), atol=1e-10)
    # Check determinant
    assert np.isclose(np.linalg.det(R), 1.0)


def test_lunar_surface_horizon():
    """LunarSurface: above_horizon returns bool mask."""
    obs = LunarSurface(0, 0)
    obs.set_time("2000-01-01")

    mask = obs.above_horizon(nside=4)

    npix = healpy.nside2npix(4)
    assert mask.shape == (npix,)
    assert mask.dtype == bool
    # Roughly hemisphere should be visible from lunar surface
    assert 0.4 < np.sum(mask) / npix < 0.6


def test_moon_icrs2mcmf_earth_libration_bounded():
    """
    _moon_icrs2mcmf: Earth's direction, expressed in the MCMF frame, must
    stay near a fixed point (real optical libration is ~8 deg in longitude,
    ~7 deg in latitude), not sweep across the sky.  This is a regression
    test for a rotation-convention bug that previously caused the computed
    sub-Earth point to sweep ~166 deg in longitude over a single month.
    """
    times = Time("2030-01-01") + np.linspace(0, 30, 15) * u.day
    moon_pos = get_body_barycentric("moon", times)
    earth_pos = get_body_barycentric("earth", times) - moon_pos
    xyz = earth_pos.xyz.to(u.m).value
    d_icrs = xyz / np.linalg.norm(xyz, axis=0)  # (3, ntimes)

    d_mcmf = np.empty_like(d_icrs)
    for i, t in enumerate(times):
        R = _moon_icrs2mcmf(t)
        d_mcmf[:, i] = R @ d_icrs[:, i]

    # Great-circle angle of each sample from the first sample.
    ref = d_mcmf[:, 0]
    cos_ang = np.clip(ref @ d_mcmf, -1.0, 1.0)
    ang = np.rad2deg(np.arccos(cos_ang))

    # Generous margin over the true ~10 deg combined libration amplitude.
    assert np.all(ang < 15.0)

    # Also sanity-check selenographic lon/lat stay near a fixed point.
    lon = np.rad2deg(np.arctan2(d_mcmf[1], d_mcmf[0]))
    lat = np.rad2deg(np.arcsin(np.clip(d_mcmf[2], -1, 1)))
    assert lon.max() - lon.min() < 20.0
    assert lat.max() - lat.min() < 20.0


def test_moon_icrs2mcmf_pole_stable():
    """
    _moon_icrs2mcmf: the Moon's pole direction (expressed back in ICRS)
    should be very stable over a month -- real precession of the lunar
    pole has an 18.6-year period, so month-to-month motion is tiny.  This
    guards against a "fix" that merely freezes the prime-meridian angle W
    to a wrong-but-constant value rather than correctly tracking Earth.
    """
    times = Time("2030-01-01") + np.linspace(0, 30, 6) * u.day
    poles = np.array(
        [_moon_icrs2mcmf(t).T @ np.array([0.0, 0.0, 1.0]) for t in times]
    )
    ref = poles[0]
    cos_ang = np.clip(poles @ ref, -1.0, 1.0)
    ang = np.rad2deg(np.arccos(cos_ang))

    # Real precessional wobble over a month is ~0.05 deg; use a tolerance
    # comfortably above that but far tighter than the ~10-15 deg libration
    # bound above.
    assert np.all(ang < 0.2)


def test_circular_orbital_period():
    """circular_orbital_period: compute period for given altitude."""
    altitude = 100e3  # 100 km
    period = circular_orbital_period(altitude)

    # Period should be reasonable (roughly 2 hours for lunar orbit)
    assert 6000 < period < 8000  # seconds


def test_circular_orbital_period_formula():
    """circular_orbital_period: verify Kepler's third law."""
    altitude = 0.0
    period = circular_orbital_period(altitude)

    # For altitude = 0, check P^2 = 4π^2 R^3 / GM
    r = R_MOON
    expected_period = 2 * np.pi * np.sqrt(r**3 / GM_MOON)

    assert np.isclose(period, expected_period)


def test_lunar_orbit_basic():
    """LunarOrbit: basic construction."""
    altitude = 100e3
    rot_orbit_vec = [0, 0, 1]
    rot_spin_vec = [0, 0, 1]

    orbit = LunarOrbit(altitude, rot_orbit_vec, rot_spin_vec)

    assert orbit.altitude == altitude
    assert np.allclose(orbit.rot_orbit_vec, [0, 0, 1])
    assert np.allclose(orbit.rot_spin_vec, [0, 0, 1])


def test_lunar_orbit_orbital_radius():
    """LunarOrbit: orbital radius is correct."""
    altitude = 100e3
    orbit = LunarOrbit(altitude, [0, 0, 1], [0, 0, 1])

    assert orbit.orbital_radius == R_MOON + altitude


def test_lunar_orbit_set_time():
    """LunarOrbit: set_time updates orbital and spin phases."""
    t0 = Time("2000-01-01")
    orbit = LunarOrbit(100e3, [0, 0, 1], [0, 0, 1], spin_period=3600.0, t0=t0)

    # At t0, phases should be zero
    orbit.set_time(t0)
    assert np.isclose(orbit._th_orbit, 0.0)
    assert np.isclose(orbit._th_spin, 0.0)

    # 1 second later
    t1 = t0 + (1 * u.s)
    orbit.set_time(t1)

    # Orbital phase should advance
    assert orbit._th_orbit > 0.0
    # Spin phase should advance
    assert orbit._th_spin > 0.0


def test_lunar_orbit_set_phases():
    """LunarOrbit: set_phases directly sets orbital and spin angles."""
    orbit = LunarOrbit(100e3, [0, 0, 1], [0, 0, 1])

    th_orbit = np.pi / 4
    th_spin = np.pi / 2

    orbit.set_phases(th_orbit, th_spin)

    assert np.isclose(orbit._th_orbit, th_orbit)
    assert np.isclose(orbit._th_spin, th_spin)


def test_lunar_orbit_spacecraft_position():
    """LunarOrbit: spacecraft position at phase 0 is along start_pos."""
    altitude = 100e3
    start_pos = [1, 0, 0]
    orbit = LunarOrbit(altitude, [0, 0, 1], [0, 0, 1], start_pos=start_pos)

    pos = orbit.spacecraft_position()

    # Should be along start_pos direction at orbital_radius distance
    expected = np.array(start_pos) * (R_MOON + altitude)
    assert np.allclose(pos, expected, atol=1e-6)


def test_lunar_orbit_spacecraft_position_90deg():
    """LunarOrbit: spacecraft position at 90° orbital phase."""
    altitude = 100e3
    orbit = LunarOrbit(altitude, [0, 0, 1], [0, 0, 1], start_pos=[1, 0, 0])

    orbit.set_phases(th_orbit=np.pi / 2)
    pos = orbit.spacecraft_position()

    # Should be at 90 degrees from start_pos in the orbital plane
    expected = np.array([0, 1, 0]) * (R_MOON + altitude)
    assert np.allclose(pos, expected, atol=1e-6)


def test_lunar_orbit_rotation():
    """LunarOrbit: rot_gal2top returns valid rotation matrix."""
    orbit = LunarOrbit(100e3, [0, 0, 1], [0, 0, 1])
    orbit.set_time("2000-01-01")

    R = orbit.rot_gal2top()

    # Check shape
    assert R.shape == (3, 3)
    # Check orthogonal
    identity = R.T @ R
    assert np.allclose(identity, np.eye(3), atol=1e-10)
    # Check determinant
    assert np.isclose(np.linalg.det(R), 1.0)


def test_lunar_orbit_rotation_stack_matches_scalar():
    """LunarOrbit: batched rotations match scalar rot_gal2top calls."""
    orbit = LunarOrbit(
        100e3,
        [0, 0, 1],
        [0, 0, 1],
        spin_period=3600.0,
        t0=Time("2000-01-01"),
    )
    times = orbit.t0 + np.array([0.0, 60.0, 120.0]) * u.s

    R_stack = orbit.rot_gal2top_stack(times)
    R_loop = []
    for t in times:
        orbit.set_time(t)
        R_loop.append(orbit.rot_gal2top())

    np.testing.assert_allclose(R_stack, np.stack(R_loop), atol=1e-7)


def test_lunar_orbit_above_horizon_no_occultation():
    """LunarOrbit: far from moon, all sky visible."""
    altitude = 100e3
    # Very high altitude for large radius
    orbit = LunarOrbit(5e7, [0, 0, 1], [0, 0, 1])
    orbit.set_time("2000-01-01")

    mask = orbit.above_horizon(nside=4)

    # Most sky should be visible
    assert np.sum(mask) > 0.9 * len(mask)


def test_lunar_orbit_above_horizon_stack_matches_scalar():
    """LunarOrbit: batched occultation masks match scalar masks."""
    orbit = LunarOrbit(
        100e3,
        [0, 0, 1],
        [0, 0, 1],
        spin_period=3600.0,
        t0=Time("2000-01-01"),
    )
    times = orbit.t0 + np.array([0.0, 60.0, 120.0]) * u.s

    mask_stack = orbit.above_horizon_stack(times, nside=4)
    mask_loop = []
    for t in times:
        orbit.set_time(t)
        mask_loop.append(orbit.above_horizon(nside=4))

    np.testing.assert_array_equal(mask_stack, np.stack(mask_loop))


def test_lunar_orbit_above_horizon_occultation():
    """LunarOrbit: low orbit, moon blocks some sky."""
    altitude = 100e3
    orbit = LunarOrbit(altitude, [0, 0, 1], [0, 0, 1])
    orbit.set_time("2000-01-01")

    mask = orbit.above_horizon(nside=4)

    npix = healpy.nside2npix(4)
    # Moon should block some pixels (not all sky visible)
    assert np.sum(mask) < npix
    # But significant portion of sky still visible (at least 50%)
    assert np.sum(mask) > 0.5 * npix


def test_lunar_orbit_normalize_vectors():
    """LunarOrbit: normalizes input vectors."""
    altitude = 100e3
    # Provide non-unit vectors
    rot_orbit_vec = [2, 0, 0]
    rot_spin_vec = [0, 3, 0]

    orbit = LunarOrbit(altitude, rot_orbit_vec, rot_spin_vec)

    assert np.isclose(np.linalg.norm(orbit.rot_orbit_vec), 1.0)
    assert np.isclose(np.linalg.norm(orbit.rot_spin_vec), 1.0)


def test_lunar_orbit_zero_spin_period():
    """LunarOrbit: zero spin period means no spin."""
    orbit = LunarOrbit(100e3, [0, 0, 1], [0, 0, 1], spin_period=0.0)
    orbit.set_time("2000-01-01")

    # Advance time
    t1 = orbit.time + (100 * u.s)
    orbit.set_time(t1)

    # Spin phase should remain 0
    assert orbit._th_spin == 0.0


if __name__ == "__main__":
    test_observer_abstract()
    test_icrs2gal_matrix()
    test_earth_surface_basic()
    test_earth_surface_time()
    test_earth_surface_rotation()
    test_earth_surface_horizon()
    test_earth_surface_horizon_equator()
    test_earth_surface_horizon_pole()
    test_lunar_surface_basic()
    test_lunar_surface_time()
    test_lunar_surface_rotation()
    test_lunar_surface_horizon()
    test_circular_orbital_period()
    test_circular_orbital_period_formula()
    test_lunar_orbit_basic()
    test_lunar_orbit_orbital_radius()
    test_lunar_orbit_set_time()
    test_lunar_orbit_set_phases()
    test_lunar_orbit_spacecraft_position()
    test_lunar_orbit_spacecraft_position_90deg()
    test_lunar_orbit_rotation()
    test_lunar_orbit_above_horizon_no_occultation()
    test_lunar_orbit_above_horizon_occultation()
    test_lunar_orbit_normalize_vectors()
    test_lunar_orbit_zero_spin_period()
    print("\n✓ All Phase 4 (observer.py) tests passed!")
