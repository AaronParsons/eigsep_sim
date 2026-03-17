"""
Tests for eigsep_sim.src — coordinate helpers, disc_overlap_fraction,
SourceCatalog, and the ICRS2GAL rotation matrix from _observer.
"""

import numpy as np
import pytest
import healpy as hp
from eigsep_sim.src import (
    disc_overlap_fraction,
    radec_to_eqvec,
    random_points_on_sphere,
    SourceCatalog,
    Jy2K_nside,
)
from eigsep_sim._observer import ICRS2GAL


# ---------------------------------------------------------------------------
# ICRS2GAL rotation matrix
# ---------------------------------------------------------------------------

class TestICRS2GAL:
    def test_orthogonal(self):
        np.testing.assert_allclose(ICRS2GAL @ ICRS2GAL.T, np.eye(3), atol=1e-10)

    def test_determinant_one(self):
        np.testing.assert_allclose(np.linalg.det(ICRS2GAL), 1.0, atol=1e-10)

    def test_maps_known_direction(self):
        """The galactic centre (l=0, b=0) should map to a known ICRS direction."""
        # Galactic centre in galactic frame is (1, 0, 0)
        gc_gal = np.array([1.0, 0.0, 0.0])
        # ICRS2GAL @ v_icrs = v_gal  ⟹  v_icrs = ICRS2GAL.T @ v_gal
        gc_icrs = ICRS2GAL.T @ gc_gal
        # The galactic centre is near RA=266.4°, Dec=−28.9° in ICRS
        ra_deg = np.degrees(np.arctan2(gc_icrs[1], gc_icrs[0])) % 360
        dec_deg = np.degrees(np.arcsin(gc_icrs[2]))
        assert 260 < ra_deg < 272
        assert -32 < dec_deg < -26


# ---------------------------------------------------------------------------
# disc_overlap_fraction
# ---------------------------------------------------------------------------

class TestDiscOverlapFraction:
    def test_fracs_sum_to_one(self):
        crd = np.array([1, 0, 0], dtype=float)
        pix, frac = disc_overlap_fraction(nside=16, crd=crd, r_rad=0.1)
        np.testing.assert_allclose(frac.sum(), 1.0, rtol=1e-5)

    def test_very_small_disc_few_pixels(self):
        """A point-like disc should overlap only a small number of pixels."""
        crd = np.array([0, 0, 1], dtype=float)
        pix, frac = disc_overlap_fraction(nside=8, crd=crd, r_rad=1e-3)
        # Sub-pixel disc → fracs spread over at most a handful of pixels
        assert len(pix) <= 8
        assert frac.sum() == pytest.approx(1.0, rel=1e-5)

    def test_nonnegative_fracs(self):
        crd = np.array([0, 1, 0], dtype=float)
        pix, frac = disc_overlap_fraction(nside=8, crd=crd, r_rad=0.05)
        assert np.all(frac >= 0)

    def test_returns_ring_pixels(self):
        """Returned pixels should be valid ring-scheme indices for the given nside."""
        nside = 16
        crd = np.array([1, 0, 0], dtype=float)
        pix, frac = disc_overlap_fraction(nside=nside, crd=crd, r_rad=0.2)
        assert np.all(pix >= 0)
        assert np.all(pix < hp.nside2npix(nside))


# ---------------------------------------------------------------------------
# radec_to_eqvec
# ---------------------------------------------------------------------------

class TestRadecToEqvec:
    def test_north_pole(self):
        """Dec=90° should map to (0, 0, 1) regardless of RA."""
        v = radec_to_eqvec(0.0, np.pi / 2)
        np.testing.assert_allclose(v, [0, 0, 1], atol=1e-10)

    def test_unit_vector(self):
        ra = np.array([0.5, 1.0, 2.0])
        dec = np.array([0.1, -0.3, 0.5])
        v = radec_to_eqvec(ra, dec)
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Jy2K_nside
# ---------------------------------------------------------------------------

class TestJy2KNside:
    def test_positive_and_finite(self):
        freqs = np.linspace(50e6, 200e6, 10)
        vals = Jy2K_nside(nside=64, freqs_Hz=freqs)
        assert np.all(vals > 0)
        assert np.all(np.isfinite(vals))

    def test_increases_with_lower_frequency(self):
        """Rayleigh-Jeans: T ∝ 1/ν², so lower freq → higher K per Jy."""
        v_lo = Jy2K_nside(nside=64, freqs_Hz=np.array([50e6]))
        v_hi = Jy2K_nside(nside=64, freqs_Hz=np.array([200e6]))
        assert v_lo[0] > v_hi[0]


# ---------------------------------------------------------------------------
# random helpers
# ---------------------------------------------------------------------------

class TestRandomHelpers:
    def test_random_points_shape(self):
        pts = random_points_on_sphere(20)
        assert pts.shape == (20, 3)

    def test_random_points_unit_norm(self):
        pts = random_points_on_sphere(50)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# SourceCatalog
# ---------------------------------------------------------------------------

class TestSourceCatalog:
    def _make_cat(self, nside=8, nfreq=5):
        freqs = np.linspace(50e6, 200e6, nfreq)
        return SourceCatalog(nside=nside, freqs_Hz=freqs)

    def test_empty_on_construction(self):
        cat = self._make_cat()
        assert len(cat._ss_sources) == 0
        assert cat._crd_gal.shape == (0, 3)
        assert cat._Tsrc.shape[1] == 5

    def test_add_moon(self):
        cat = self._make_cat()
        cat.add_moon()
        assert len(cat._ss_sources) == 1
        src = cat._ss_sources[0]
        assert src.name == 'moon'
        assert src.radius_m > 0

    def test_add_sun(self):
        cat = self._make_cat()
        cat.add_sun()
        assert len(cat._ss_sources) == 1
        assert cat._ss_sources[0].name == 'sun'

    def test_update_positions(self):
        cat = self._make_cat()
        cat.add_moon()
        cat.update_positions('2024-06-01')
        src = cat._ss_sources[0]
        assert src.crd_gal is not None
        assert src.crd_gal.shape == (3,)
        norm = np.linalg.norm(src.crd_gal)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_moon_temperature(self):
        cat = self._make_cat()
        cat.add_moon(T_mean=220)
        freqs = np.linspace(50e6, 200e6, 5)
        T = cat._ss_sources[0].temperature(freqs)
        np.testing.assert_allclose(T, 220.0)

    def test_solar_system_source_angular_radius_before_update(self):
        cat = self._make_cat()
        cat.add_moon()
        # Before update, distance_m is None → angular radius = 0
        assert cat._ss_sources[0].angular_radius() == 0.0

    def test_solar_system_source_angular_radius_after_update(self):
        cat = self._make_cat()
        cat.add_moon()
        cat.update_positions('2024-06-01')
        r = cat._ss_sources[0].angular_radius()
        # Moon subtends ~0.26° ≈ 0.0045 rad
        assert 0.002 < r < 0.01

    def test_add_planets(self):
        cat = self._make_cat()
        cat.add_planets()
        assert len(cat._ss_sources) == 7  # mercury through neptune (no earth)
