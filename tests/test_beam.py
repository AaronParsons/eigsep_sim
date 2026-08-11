"""
Tests for eigsep_sim.beam — analytic dipole beams and the Beam class.
"""

import numpy as np
import pytest
import healpy
from eigsep_sim.beam import (
    short_dipole_beam,
    thin_dipole_beam,
    v_dipole_arm_axes,
    v_dipole_beam,
    v_dipole_pattern,
    analytic_dipole_beam,
    Beam,
)


# ---------------------------------------------------------------------------
# short_dipole_beam
# ---------------------------------------------------------------------------

class TestShortDipoleBeam:
    def test_output_shape(self):
        nside = 8
        freqs = np.array([100e6, 200e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside)
        assert bm.shape == (healpy.nside2npix(nside), 2)

    def test_zero_on_dipole_axis(self):
        """Power must be small along the dipole axis (dhat · rhat ≈ ±1)."""
        nside = 64  # Higher nside → pixel centres closer to exact poles
        freqs = np.array([100e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside, dipole_axis=(0, 0, 1))
        x, y, z = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
        # Minimum response should be at the most pole-ward pixels
        most_north = np.argmax(z)
        most_south = np.argmin(z)
        assert bm[most_north, 0] < 0.01
        assert bm[most_south, 0] < 0.01

    def test_max_perpendicular_to_axis(self):
        """Maximum response should be perpendicular to the dipole axis."""
        nside = 32
        freqs = np.array([100e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside, dipole_axis=(0, 0, 1))
        x, y, z = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
        # Equatorial pixels have z ≈ 0
        equatorial = np.abs(z) < 0.05
        assert bm[equatorial, 0].max() > 0.95 * bm[:, 0].max()

    def test_mean_two_thirds(self):
        """Solid-angle average of (1 − cos²θ) over the sphere equals 2/3."""
        nside = 64
        freqs = np.array([100e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside, dipole_axis=(0, 0, 1))
        mean = bm[:, 0].mean()
        np.testing.assert_allclose(mean, 2.0 / 3.0, rtol=0.01)

    def test_frequency_independent(self):
        """Short-dipole beam should be identical at all frequencies."""
        nside = 8
        freqs = np.array([50e6, 100e6, 200e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside)
        for i in range(1, freqs.size):
            np.testing.assert_array_equal(bm[:, 0], bm[:, i])

    def test_nonnegative(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside)
        assert np.all(bm >= 0)

    def test_horizon_clip(self):
        """With horizon_clip=True, all z < 0 pixels must be zero."""
        nside = 16
        freqs = np.array([100e6], dtype=np.float32)
        bm = short_dipole_beam(freqs, nside, horizon_clip=True)
        x, y, z = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
        assert np.all(bm[z < 0, 0] == 0.0)


# ---------------------------------------------------------------------------
# thin_dipole_beam
# ---------------------------------------------------------------------------

class TestThinDipoleBeam:
    def test_output_shape(self):
        nside = 8
        freqs = np.array([100e6, 200e6], dtype=np.float32)
        bm = thin_dipole_beam(freqs, nside, dipole_length=1.5)
        assert bm.shape == (healpy.nside2npix(nside), 2)

    def test_nonnegative(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm = thin_dipole_beam(freqs, nside)
        assert np.all(bm >= 0)

    def test_frequency_dependent(self):
        """Patterns at different frequencies must differ for a thin dipole."""
        nside = 16
        freqs = np.array([50e6, 150e6], dtype=np.float32)
        bm = thin_dipole_beam(freqs, nside, dipole_length=1.5)
        assert not np.allclose(bm[:, 0], bm[:, 1])

    def test_zero_on_axis(self):
        """Thin dipole response must be small near the dipole axis."""
        nside = 64
        freqs = np.array([100e6], dtype=np.float32)
        bm = thin_dipole_beam(freqs, nside, dipole_axis=(0, 0, 1))
        x, y, z = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
        most_north = np.argmax(z)
        assert bm[most_north, 0] < 0.01

    def test_horizon_clip(self):
        nside = 16
        freqs = np.array([100e6], dtype=np.float32)
        bm = thin_dipole_beam(freqs, nside, horizon_clip=True)
        x, y, z = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
        assert np.all(bm[z < 0, 0] == 0.0)


# ---------------------------------------------------------------------------
# v_dipole_beam
# ---------------------------------------------------------------------------

class TestVDipoleBeam:
    def test_output_shape(self):
        nside = 8
        freqs = np.array([100e6, 200e6], dtype=np.float32)
        bm = v_dipole_beam(freqs, nside, opening_angle_deg=90.0, dipole_length=6.0)
        assert bm.shape == (healpy.nside2npix(nside), 2)

    def test_nonnegative_and_finite(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm = v_dipole_beam(freqs, nside, opening_angle_deg=120.0, dipole_length=4.0)
        assert np.all(np.isfinite(bm))
        assert np.all(bm >= 0.0)

    def test_accepts_straight_limit(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm = v_dipole_beam(freqs, nside, opening_angle_deg=180.0, dipole_length=6.0)
        assert np.all(np.isfinite(bm))
        assert bm.max() > 0.0

    def test_opening_angle_changes_pattern(self):
        nside = 16
        freqs = np.array([100e6], dtype=np.float32)
        bm90 = v_dipole_beam(freqs, nside, opening_angle_deg=90.0, dipole_length=6.0)
        bm120 = v_dipole_beam(freqs, nside, opening_angle_deg=120.0, dipole_length=6.0)
        assert not np.allclose(bm90[:, 0], bm120[:, 0])

    def test_pattern_matches_beam_for_explicit_axes(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        length = 6.0
        axes = v_dipole_arm_axes(90.0)
        crd = np.stack(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))), axis=0)
        kh = np.pi * length * freqs[0] / 299792458.0
        pattern = v_dipole_pattern(kh, axes, crd)
        bm = v_dipole_beam(freqs, nside, arm_axes=axes, dipole_length=length)
        np.testing.assert_allclose(pattern, bm[:, 0], rtol=1e-6, atol=1e-7)

    def test_invalid_opening_angle_raises(self):
        with pytest.raises(ValueError, match="opening_angle_deg"):
            v_dipole_arm_axes(0.0)
        with pytest.raises(ValueError, match="opening_angle_deg"):
            v_dipole_arm_axes(181.0)


# ---------------------------------------------------------------------------
# analytic_dipole_beam dispatcher
# ---------------------------------------------------------------------------

class TestAnalyticDipoleBeam:
    def test_short_mode(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm_a = analytic_dipole_beam(freqs, nside, dipole_model='short')
        bm_b = short_dipole_beam(freqs, nside)
        np.testing.assert_array_equal(bm_a, bm_b)

    def test_thin_mode(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm_a = analytic_dipole_beam(freqs, nside, dipole_model='thin',
                                    dipole_length=2.0)
        bm_b = thin_dipole_beam(freqs, nside, dipole_length=2.0)
        np.testing.assert_array_equal(bm_a, bm_b)

    def test_v_mode(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        bm_a = analytic_dipole_beam(freqs, nside, dipole_model='v',
                                    dipole_length=6.0, opening_angle_deg=90.0)
        bm_b = v_dipole_beam(freqs, nside, dipole_length=6.0,
                             opening_angle_deg=90.0)
        np.testing.assert_array_equal(bm_a, bm_b)

    def test_unknown_model_raises(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown dipole_model"):
            analytic_dipole_beam(freqs, nside, dipole_model='invalid')

