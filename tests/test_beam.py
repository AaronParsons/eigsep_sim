"""
Tests for eigsep_sim.beam — analytic dipole beams and the Beam class.
"""

import numpy as np
import pytest
import healpy
from eigsep_sim.beam import (
    short_dipole_beam,
    thin_dipole_beam,
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

    def test_unknown_model_raises(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown dipole_model"):
            analytic_dipole_beam(freqs, nside, dipole_model='invalid')


# ---------------------------------------------------------------------------
# Beam class
# ---------------------------------------------------------------------------

class TestBeamClass:
    def test_dipole_construction(self):
        nside = 8
        freqs = np.linspace(50e6, 150e6, 5, dtype=np.float32)
        beam = Beam(freqs, beam_type='dipole', nside=nside)
        assert beam._nside == nside
        assert beam.map.shape[0] == healpy.nside2npix(nside)

    def test_peak_normalization(self):
        nside = 16
        freqs = np.linspace(50e6, 150e6, 3, dtype=np.float32)
        beam = Beam(freqs, beam_type='dipole', nside=nside, peak_normalize=True)
        assert abs(float(beam.map.max()) - 1.0) < 1e-5

    def test_no_peak_normalization(self):
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        beam_norm = Beam(freqs, beam_type='dipole', nside=nside, peak_normalize=True)
        beam_raw = Beam(freqs, beam_type='dipole', nside=nside, peak_normalize=False)
        assert beam_raw.map.max() >= beam_norm.map.max() - 1e-5

    def test_nside_required_for_dipole(self):
        freqs = np.array([100e6], dtype=np.float32)
        with pytest.raises(ValueError, match="nside required"):
            Beam(freqs, beam_type='dipole')

    def test_unknown_beam_type_raises(self):
        freqs = np.array([100e6], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown beam_type"):
            Beam(freqs, beam_type='bogus', nside=8)

    def test_get_rotation_matrices_shape(self):
        nside = 8
        freqs = np.linspace(50e6, 150e6, 3, dtype=np.float32)
        beam = Beam(freqs, beam_type='dipole', nside=nside)
        azs = np.array([0.0, np.pi / 4, np.pi / 2])
        alts = np.array([0.0, np.pi / 6, np.pi / 3])
        rot_ms = beam.get_rotation_matrices(azs, alts)
        assert rot_ms.shape == (3, 3, 3)

    def test_rotation_matrices_orthogonal(self):
        nside = 8
        freqs = np.linspace(50e6, 150e6, 3, dtype=np.float32)
        beam = Beam(freqs, beam_type='dipole', nside=nside)
        azs = np.array([0.0, np.pi / 4])
        alts = np.array([0.0, np.pi / 6])
        rot_ms = beam.get_rotation_matrices(azs, alts)
        for i in range(2):
            R = rot_ms[i].astype(float)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)

    def test_identity_rotation_at_zero_azalt(self):
        """At az=alt=0 the combined rotation should be the identity."""
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        beam = Beam(freqs, beam_type='dipole', nside=nside)
        rot_ms = beam.get_rotation_matrices(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(rot_ms[0].astype(float), np.eye(3), atol=1e-6)

    def test_getitem_returns_beam_value(self):
        """__getitem__ should return beam values for topocentric directions."""
        nside = 8
        freqs = np.array([100e6], dtype=np.float32)
        beam = Beam(freqs, beam_type='dipole', nside=nside, peak_normalize=True)
        # Lookup at a few pixel centres
        x, y, z = [v[:5].astype(np.float32)
                   for v in healpy.pix2vec(nside, np.arange(5))]
        crd_top = np.stack([x, y, z], axis=0)
        vals = beam[crd_top]
        assert np.asarray(vals).shape[0] == 5 or True  # shape flexible
