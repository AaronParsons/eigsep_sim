"""Tests for the Sun/Earth ephemeris helpers (ephemeris.py).

Sanity-checks the ephemeris backbone that Phase-1 external sources (Sun,
Earth) build on: body distances are physically reasonable, the inertial
(galactic) direction from the Moon sweeps at the Moon's own orbital rate
(NOT stationary -- that's a common mix-up with the body-fixed/MCMF frame,
where Earth *does* stay nearly fixed by tidal locking), and the occultation
geometry agrees with simple constructed cases.
"""

import numpy as np
from astropy.time import Time
import astropy.units as u

from eigsep_sim import (
    LunarOrbit, body_directions_gal, body_occulted_by_moon,
    moon_surface_intersection_mcmf,
)
from eigsep_sim.const import R_MOON


def test_distances_are_physically_reasonable():
    times = Time("2030-01-01") + np.linspace(0, 30, 30) * u.day
    dirs, dists = body_directions_gal(times, bodies=("sun", "earth"))
    # ~1 AU +- a percent or two (Earth's orbital eccentricity)
    assert np.all(np.abs(dists["sun"] - 1.496e11) / 1.496e11 < 0.02)
    # ~384,400 km +- lunar orbital eccentricity (~5.5%)
    assert np.all(np.abs(dists["earth"] - 3.844e8) / 3.844e8 < 0.06)
    np.testing.assert_allclose(
        np.linalg.norm(dirs["sun"], axis=1), 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(dirs["earth"], axis=1), 1.0, atol=1e-10
    )


def test_earth_direction_sweeps_in_inertial_frame():
    # Inertial (galactic) Earth-direction-from-Moon rotates with the Moon's
    # own ~27.3 day orbit around Earth -- it is NOT stationary in this
    # frame.  Over a 30-day span it should sweep most of a full circle.
    times = Time("2030-01-01") + np.linspace(0, 30, 60) * u.day
    dirs, _ = body_directions_gal(times, bodies=("earth",))
    d = dirs["earth"]
    angle_from_start = np.rad2deg(np.arccos(np.clip(d @ d[0], -1, 1)))
    assert angle_from_start.max() > 100.0


def test_occultation_matches_synthetic_geometry():
    orbit = LunarOrbit(
        altitude=1e5, rot_orbit_vec=[0, 0, 1], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    times = Time("2030-01-01") + np.linspace(0, 3600, 8) * u.s
    pos = orbit.spacecraft_position_stack(times)  # (ntimes, 3)
    d = np.linalg.norm(pos, axis=1)
    moon_dir = pos / d[:, None]  # Moon centre -> spacecraft

    # direction pointing away from the Moon (same side as spacecraft): visible
    visible_dir = moon_dir
    assert not body_occulted_by_moon(orbit, times, visible_dir).any()

    # direction pointing back through the Moon's centre: occulted
    hidden_dir = -moon_dir
    assert body_occulted_by_moon(orbit, times, hidden_dir).all()


def test_occultation_fraction_reasonable_for_equatorial_orbit():
    # A low, near-equatorial orbit should see Earth (fixed near the sub-
    # observer point on the near side, once corrected to MCMF -- but here
    # we just check that a real, slowly-moving external body is occulted
    # some but not all of the time over a multi-hour span, i.e. the
    # eclipse geometry is doing something nontrivial, not degenerate.
    orbit = LunarOrbit(
        altitude=1e5, rot_orbit_vec=[0, 1, 0], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    times = Time("2030-01-01") + np.linspace(0, 6 * 3600, 200) * u.s
    dirs, _ = body_directions_gal(times, bodies=("earth",))
    occ = body_occulted_by_moon(orbit, times, dirs["earth"])
    assert 0.0 < occ.mean() < 1.0


def test_moon_surface_intersection_nadir_direction_exact():
    """Looking straight at the Moon's centre, the near intersection point
    is exactly R_MOON along that line -- i.e. directly "below" the
    spacecraft. Before the MCMF rotation this must equal moon_dir exactly;
    after it, the returned vector must still be a unit vector."""
    orbit = LunarOrbit(
        altitude=1e5, rot_orbit_vec=[0, 0, 1], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    times = Time("2030-01-01") + np.linspace(0, 3600, 5) * u.s
    pos = orbit.spacecraft_position_stack(times)
    d = np.linalg.norm(pos, axis=1)
    moon_dir = pos / d[:, None]
    hidden_dir = (-moon_dir)[:, None, :]  # (ntimes, 1, 3)

    normals = moon_surface_intersection_mcmf(orbit, times, hidden_dir)
    assert normals.shape == (5, 1, 3)
    assert not np.any(np.isnan(normals))
    np.testing.assert_allclose(
        np.linalg.norm(normals[:, 0], axis=-1), 1.0, atol=1e-8
    )


def test_moon_surface_intersection_nonoccluded_direction_is_nan():
    orbit = LunarOrbit(
        altitude=1e5, rot_orbit_vec=[0, 0, 1], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    times = Time("2030-01-01") + np.linspace(0, 3600, 5) * u.s
    pos = orbit.spacecraft_position_stack(times)
    d = np.linalg.norm(pos, axis=1)
    moon_dir = pos / d[:, None]
    visible_dir = moon_dir[:, None, :]  # points away from the Moon: no hit

    normals = moon_surface_intersection_mcmf(orbit, times, visible_dir)
    assert np.all(np.isnan(normals))


def test_moon_surface_intersection_broadcasts_shared_pixel_grid():
    orbit = LunarOrbit(
        altitude=1e5, rot_orbit_vec=[0, 0, 1], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    # Short span: a *fixed* (non-tracking) shared direction stays valid as
    # "always hits"/"never hits" only while the spacecraft barely moves
    # (orbital period here is ~2 hr, so a few seconds is safely negligible).
    times = Time("2030-01-01") + np.linspace(0, 4, 4) * u.s
    pos = orbit.spacecraft_position_stack(times)
    d = np.linalg.norm(pos, axis=1)
    moon_dir0 = pos[0] / d[0]
    shared_dirs = np.array([-moon_dir0, moon_dir0])  # (2, 3): one hits, one misses

    normals = moon_surface_intersection_mcmf(orbit, times, shared_dirs)
    assert normals.shape == (4, 2, 3)
    assert not np.any(np.isnan(normals[:, 0]))  # hidden_dir-like: always hits
    assert np.all(np.isnan(normals[:, 1]))  # visible_dir-like: never hits
