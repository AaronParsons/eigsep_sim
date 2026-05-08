#!/usr/bin/env python
"""
Test suite for refactored beam.py — HEALPix beam model with spectral basis.

Tests verify:
1. Beam construction from thin-dipole analytic model
2. Beam evaluation and reconstruction
3. Solid angle computation
4. Serialization (save/load)
5. Per-dipole arm lengths
"""

import os
import tempfile
import numpy as np
import healpy
import pytest

from eigsep_sim.beam import Beam, thin_dipole_pattern, short_dipole_beam


def test_thin_dipole_pattern():
    """Test thin_dipole_pattern basic functionality."""
    kh = np.array([0.5, 1.0])  # electrical half-lengths
    cos_theta = np.array([0.0, 0.5, 1.0])  # cosine of angle from dipole axis

    pattern = thin_dipole_pattern(kh[:, None], cos_theta[None, :])
    assert pattern.shape == (2, 3)
    assert np.all(np.isfinite(pattern))
    # Pattern should be zero on-axis (cos_theta = 1.0)
    assert np.allclose(pattern[:, -1], 0.0)
    print("✓ test_thin_dipole_pattern passed")


def test_short_dipole_beam():
    """Test short_dipole_beam function."""
    freqs = np.array([50e6, 100e6, 150e6])
    beam = short_dipole_beam(freqs, nside=4)

    assert beam.shape == (healpy.nside2npix(4), len(freqs))
    assert np.all((beam >= 0) & (beam <= 1))  # normalized power pattern
    print("✓ test_short_dipole_beam passed")


def test_beam_from_dipole():
    """Test Beam.from_dipole construction."""
    freqs = np.array([50e6, 100e6, 150e6])
    arm_length = 3.0

    beam = Beam.from_dipole(nside=4, freqs_hz=freqs, arm_lengths_m=arm_length, K=2)

    assert beam.nside == 4
    assert beam.npix == healpy.nside2npix(4)
    assert beam.n_dipoles == 2
    assert beam.nmodes == 2
    assert np.allclose(beam.freqs_hz, freqs)
    print("✓ test_beam_from_dipole passed")


def test_beam_from_dipole_per_dipole_lengths():
    """Test Beam.from_dipole with per-dipole arm lengths."""
    freqs = np.array([50e6, 100e6, 150e6])
    arm_lengths = np.array([3.0, 4.0])

    beam = Beam.from_dipole(nside=4, freqs_hz=freqs, arm_lengths_m=arm_lengths, K=2)

    assert beam.n_dipoles == 2
    assert np.allclose(beam.u_body[0], [1.0, 0.0, 0.0])
    assert np.allclose(beam.u_body[1], [0.0, 1.0, 0.0])
    print("✓ test_beam_from_dipole_per_dipole_lengths passed")


def test_beam_evaluate():
    """Test Beam.evaluate at different frequencies."""
    freqs = np.array([50e6, 100e6, 150e6])
    beam = Beam.from_dipole(nside=4, freqs_hz=freqs, arm_lengths_m=3.0, K=2)

    for freq_idx in range(len(freqs)):
        pattern = beam.evaluate(freq_idx)
        assert pattern.shape == (2, beam.npix)
        assert np.all(np.isfinite(pattern))

    # Out of bounds should raise
    try:
        beam.evaluate(len(freqs))
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    print("✓ test_beam_evaluate passed")


def test_beam_solid_angle():
    """Test Beam.solid_angle computation."""
    freqs = np.array([50e6, 100e6])
    beam = Beam.from_dipole(nside=8, freqs_hz=freqs, arm_lengths_m=3.0, K=2)

    omega = beam.solid_angle(0)
    assert omega.shape == (2,)
    assert np.all(np.isfinite(omega))
    print("✓ test_beam_solid_angle passed")


def test_beam_save_load():
    """Test Beam serialization."""
    freqs = np.array([50e6, 100e6, 150e6])
    beam_orig = Beam.from_dipole(nside=4, freqs_hz=freqs, arm_lengths_m=3.0, K=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'beam.npz')
        beam_orig.save(path)
        beam_loaded = Beam.from_file(path)

        assert beam_loaded.nside == beam_orig.nside
        assert np.allclose(beam_loaded.freqs_hz, beam_orig.freqs_hz)
        assert np.allclose(beam_loaded.coeffs, beam_orig.coeffs)
        assert np.allclose(beam_loaded.u_body, beam_orig.u_body)

        # Evaluate should give same results
        pattern_orig = beam_orig.evaluate(1)
        pattern_loaded = beam_loaded.evaluate(1)
        assert np.allclose(pattern_orig, pattern_loaded)

    print("✓ test_beam_save_load passed")


def test_beam_save_load_resample():
    """Test Beam.from_file with resampling to new frequencies."""
    freqs_old = np.array([50e6, 100e6, 150e6])
    freqs_new = np.linspace(50e6, 150e6, 10)

    beam_old = Beam.from_dipole(nside=4, freqs_hz=freqs_old, arm_lengths_m=3.0, K=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'beam.npz')
        beam_old.save(path)
        beam_new = Beam.from_file(path, new_freqs=freqs_new)

        assert beam_new.nside == beam_old.nside
        assert len(beam_new.freqs_hz) == len(freqs_new)
        assert np.allclose(beam_new.freqs_hz, freqs_new)

    print("✓ test_beam_save_load_resample passed")


def test_beam_k_limit():
    """Test that Beam.from_dipole handles K > rank gracefully."""
    freqs = np.array([50e6, 100e6, 150e6])
    K_requested = 10  # More than nfreq=3

    beam = Beam.from_dipole(nside=4, freqs_hz=freqs, arm_lengths_m=3.0, K=K_requested)

    # Should be limited by min(npix, nfreq)
    assert beam.nmodes <= min(healpy.nside2npix(4), 3)
    print(f"✓ test_beam_k_limit: requested K={K_requested}, actual nmodes={beam.nmodes}")


if __name__ == "__main__":
    test_thin_dipole_pattern()
    test_short_dipole_beam()
    test_beam_from_dipole()
    test_beam_from_dipole_per_dipole_lengths()
    test_beam_evaluate()
    test_beam_solid_angle()
    test_beam_save_load()
    test_beam_save_load_resample()
    test_beam_k_limit()
    print("\n✓ All Phase 2 (Beam refactor) tests passed!")
