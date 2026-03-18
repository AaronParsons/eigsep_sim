"""
Tests for observer ephemeris modules:
  _observer.py  (Observer base class, ICRS2GAL)
  earth_surface.py  (EarthSurface)
  lunar_surface.py  (LunarSurface, _moon_icrs2mcmf)
  lunar_orbit.py    (LunarOrbit)
"""

import numpy as np
import healpy
import pytest
import astropy.units as u
from astropy.time import Time

from eigsep_sim._observer import Observer, ICRS2GAL
from eigsep_sim.earth_surface import EarthSurface
from eigsep_sim.lunar_surface import LunarSurface, _moon_icrs2mcmf
from eigsep_sim.lunar_orbit import LunarOrbit, circular_orbital_period
from eigsep_sim.const import R_MOON, GM_MOON

# A fixed epoch used throughout
T0 = Time("2025-01-01T00:00:00", scale="utc")
T1 = Time("2025-01-01T06:00:00", scale="utc")  # 6 h later

NSIDE = 16
NPIX = healpy.nside2npix(NSIDE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rotation(R, atol=1e-10):
    """Return True iff R is a proper rotation matrix."""
    err_orth = np.max(np.abs(R @ R.T - np.eye(3)))
    err_det = abs(np.linalg.det(R) - 1.0)
    return err_orth < atol and err_det < atol


# ---------------------------------------------------------------------------
# ICRS2GAL
# ---------------------------------------------------------------------------

class TestICRS2GAL:
    def test_shape(self):
        assert ICRS2GAL.shape == (3, 3)

    def test_is_rotation_matrix(self):
        assert _is_rotation(ICRS2GAL)

    def test_galactic_north_pole(self):
        """ICRS north-galactic-pole direction maps to galactic (0, 0, 1)."""
        # NGP in ICRS: RA=192.859°, Dec=27.128°
        ra = np.deg2rad(192.859508)
        dec = np.deg2rad(27.128336)
        ngp_icrs = np.array([np.cos(dec) * np.cos(ra),
                              np.cos(dec) * np.sin(ra),
                              np.sin(dec)])
        ngp_gal = ICRS2GAL @ ngp_icrs
        # Should be close to galactic north pole [0, 0, 1]
        np.testing.assert_allclose(ngp_gal, [0, 0, 1], atol=1e-3)


# ---------------------------------------------------------------------------
# Observer base class
# ---------------------------------------------------------------------------

class TestObserverBase:
    def test_set_time(self):
        obs = Observer()
        assert obs.time is None
        obs.set_time(T0)
        assert isinstance(obs.time, Time)
        assert abs((obs.time - T0).to("s").value) < 1e-3

    def test_abstract_methods_raise(self):
        obs = Observer()
        with pytest.raises(NotImplementedError):
            obs.rot_gal2top()
        with pytest.raises(NotImplementedError):
            obs.above_horizon(NSIDE)


# ---------------------------------------------------------------------------
# EarthSurface
# ---------------------------------------------------------------------------

class TestEarthSurface:
    def _make(self, lat=37.9, lon=-122.2):
        """Hat Creek, CA, approximately."""
        obs = EarthSurface(lat=lat, lon=lon)
        obs.set_time(T0)
        return obs

    def test_rot_gal2top_is_rotation(self):
        obs = self._make()
        R = obs.rot_gal2top()
        assert R.shape == (3, 3)
        assert _is_rotation(R)

    def test_above_horizon_shape_and_dtype(self):
        obs = self._make()
        mask = obs.above_horizon(NSIDE)
        assert mask.shape == (NPIX,)
        assert mask.dtype == bool

    def test_above_horizon_roughly_half(self):
        """A non-polar site should see approximately half the sky."""
        obs = self._make()
        mask = obs.above_horizon(NSIDE)
        frac = mask.sum() / NPIX
        assert 0.35 < frac < 0.65

    def test_zenith_pixel_is_above_horizon(self):
        """The pixel nearest to the zenith direction should be above the horizon."""
        obs = self._make()
        R = obs.rot_gal2top()
        up_gal = R[2, :]           # z-row of gal2top: zenith direction in gal frame
        zenith_pix = healpy.vec2pix(NSIDE, *up_gal)
        mask = obs.above_horizon(NSIDE)
        assert mask[zenith_pix]

    def test_nadir_pixel_is_below_horizon(self):
        """The pixel nearest to nadir should be below the horizon."""
        obs = self._make()
        R = obs.rot_gal2top()
        nadir_gal = -R[2, :]
        nadir_pix = healpy.vec2pix(NSIDE, *nadir_gal)
        mask = obs.above_horizon(NSIDE)
        assert not mask[nadir_pix]

    def test_different_times_give_different_rotation(self):
        """Earth rotates, so gal2top should differ by ~6 h."""
        obs = self._make()
        R0 = obs.rot_gal2top().copy()
        obs.set_time(T1)
        R1 = obs.rot_gal2top()
        assert not np.allclose(R0, R1, atol=1e-3)

    def test_polar_site_above_horizon(self):
        """At the geographic north pole above_horizon should still return half-ish sky."""
        obs = EarthSurface(lat=90.0, lon=0.0)
        obs.set_time(T0)
        mask = obs.above_horizon(NSIDE)
        frac = mask.sum() / NPIX
        assert 0.35 < frac < 0.65


# ---------------------------------------------------------------------------
# LunarSurface
# ---------------------------------------------------------------------------

class TestMoonICRS2MCMF:
    def test_is_rotation_matrix(self):
        R = _moon_icrs2mcmf(T0)
        assert R.shape == (3, 3)
        assert _is_rotation(R)

    def test_changes_with_time(self):
        R0 = _moon_icrs2mcmf(T0)
        R1 = _moon_icrs2mcmf(T1)
        assert not np.allclose(R0, R1, atol=1e-6)


class TestLunarSurface:
    def _make(self, lat=0.0, lon=0.0):
        obs = LunarSurface(lat=lat, lon=lon)
        obs.set_time(T0)
        return obs

    def test_rot_gal2top_is_rotation(self):
        obs = self._make()
        R = obs.rot_gal2top()
        assert R.shape == (3, 3)
        assert _is_rotation(R)

    def test_above_horizon_shape_and_dtype(self):
        obs = self._make()
        mask = obs.above_horizon(NSIDE)
        assert mask.shape == (NPIX,)
        assert mask.dtype == bool

    def test_above_horizon_roughly_half(self):
        obs = self._make()
        frac = obs.above_horizon(NSIDE).sum() / NPIX
        assert 0.35 < frac < 0.65

    def test_zenith_above_horizon(self):
        obs = self._make()
        R = obs.rot_gal2top()
        zenith_pix = healpy.vec2pix(NSIDE, *R[2, :])
        assert obs.above_horizon(NSIDE)[zenith_pix]

    def test_nadir_below_horizon(self):
        obs = self._make()
        R = obs.rot_gal2top()
        nadir_pix = healpy.vec2pix(NSIDE, *(-R[2, :]))
        assert not obs.above_horizon(NSIDE)[nadir_pix]

    def test_different_sites_different_rotation(self):
        obs0 = self._make(lat=0.0, lon=0.0)
        obs1 = self._make(lat=45.0, lon=90.0)
        R0 = obs0.rot_gal2top()
        R1 = obs1.rot_gal2top()
        assert not np.allclose(R0, R1, atol=1e-3)

    def test_different_times_give_different_rotation(self):
        """Moon rotates slowly; 6 h difference should change the frame."""
        obs = self._make()
        R0 = obs.rot_gal2top().copy()
        obs.set_time(T1)
        R1 = obs.rot_gal2top()
        assert not np.allclose(R0, R1, atol=1e-3)


# ---------------------------------------------------------------------------
# LunarOrbit
# ---------------------------------------------------------------------------

class TestCircularOrbitalPeriod:
    def test_kepler_formula(self):
        """T² = 4π²r³/GM  →  T = 2π sqrt(r³/GM)."""
        alt = 100e3
        r = R_MOON + alt
        expected = 2 * np.pi * np.sqrt(r ** 3 / GM_MOON)
        np.testing.assert_allclose(circular_orbital_period(alt), expected, rtol=1e-12)

    def test_higher_altitude_longer_period(self):
        assert circular_orbital_period(200e3) > circular_orbital_period(100e3)

    def test_100km_approximately_two_hours(self):
        """100 km LLO period is ~7066 s (~1.96 h)."""
        T = circular_orbital_period(100e3)
        assert 6800 < T < 7300


class TestLunarOrbit:
    def _make(self, altitude=1e5, spin_period=0.0, th_spin=0.0):
        obs = LunarOrbit(
            altitude=altitude,
            rot_orbit_vec=[0, 0, 1],
            rot_spin_vec=[0, 0, 1],
            start_pos=[1, 0, 0],
            spin_period=spin_period,
            t0=T0,
        )
        obs.set_phases(0.0, th_spin)
        return obs

    def test_spacecraft_position_magnitude(self):
        obs = self._make(altitude=1e5)
        pos = obs.spacecraft_position()
        assert pos.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(pos), R_MOON + 1e5, rtol=1e-10)

    def test_spacecraft_position_at_t0(self):
        """At th_orbit=0, spacecraft should be at start_pos * orbital_radius."""
        obs = self._make()
        pos = obs.spacecraft_position()
        expected = np.array([1, 0, 0]) * obs.orbital_radius
        np.testing.assert_allclose(pos, expected, atol=1.0)

    def test_spacecraft_position_half_orbit(self):
        """After half orbit, spacecraft should be at -start_pos * orbital_radius."""
        obs = self._make()
        obs.set_phases(np.pi, 0.0)
        pos = obs.spacecraft_position()
        expected = np.array([-1, 0, 0]) * obs.orbital_radius
        np.testing.assert_allclose(pos, expected, atol=1.0)

    def test_rot_gal2top_zero_spin_is_identity(self):
        """With zero spin phase, spacecraft frame == galactic frame."""
        obs = self._make(spin_period=0.0)
        R = obs.rot_gal2top()
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rot_gal2top_is_rotation(self):
        obs = self._make(spin_period=100.0)
        obs.set_phases(0.5, 1.2)
        R = obs.rot_gal2top()
        assert _is_rotation(R)

    def test_rot_gal2top_varies_with_spin(self):
        obs = self._make(spin_period=100.0)
        obs.set_phases(0.0, 0.0)
        R0 = obs.rot_gal2top().copy()
        obs.set_phases(0.0, 1.0)
        R1 = obs.rot_gal2top()
        assert not np.allclose(R0, R1, atol=1e-6)

    def test_above_horizon_shape_and_dtype(self):
        obs = self._make()
        mask = obs.above_horizon(NSIDE)
        assert mask.shape == (NPIX,)
        assert mask.dtype == bool

    def test_above_horizon_far_from_moon(self):
        """Very far from Moon: almost all pixels should be visible."""
        obs = self._make(altitude=1e8)  # 100 000 km altitude
        mask = obs.above_horizon(NSIDE)
        frac = mask.sum() / NPIX
        assert frac > 0.99

    def test_above_horizon_near_moon(self):
        """Low orbit: Moon blocks ~1/3 of sky (angular radius ~71° at 100 km)."""
        obs = self._make(altitude=1e5)  # 100 km altitude
        mask = obs.above_horizon(NSIDE)
        frac = mask.sum() / NPIX
        assert 0.55 < frac < 0.80

    def test_set_time_updates_phases(self):
        obs = self._make()
        half_period = obs.orbital_period / 2.0
        obs.set_time(T0 + half_period * u.s)
        np.testing.assert_allclose(obs._th_orbit, np.pi, atol=1e-10)

    def test_set_time_spin_period_zero(self):
        """spin_period=0 should leave _th_spin=0 regardless of elapsed time."""
        obs = self._make(spin_period=0.0)
        obs.set_time(T0 + 3600.0 * u.s)
        assert obs._th_spin == 0.0

    def test_set_phases_directly(self):
        obs = self._make()
        obs.set_phases(1.23, 4.56)
        assert obs._th_orbit == pytest.approx(1.23)
        assert obs._th_spin == pytest.approx(4.56)
