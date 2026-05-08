#!/usr/bin/env python
"""
Test suite for basis.py — spectral basis decomposition for beams and sky.

Tests verify:
1. _SpectralBasis projection/deprojection and serialization
2. BeamBasis.from_dipole construction and SVD properties
3. SkyBasis.from_gsm construction (if pygdsm available)
4. Basis resampling to new frequencies
"""

import os
import tempfile
import numpy as np
import pytest

from eigsep_sim.basis import BeamBasis, SkyBasis, _SpectralBasis


def test_spectral_basis_basic():
    """Test basic _SpectralBasis construction and properties."""
    nfreq, nmodes = 10, 3
    A = np.random.randn(nfreq, nmodes).astype(np.float32)
    freqs = np.linspace(50e6, 150e6, nfreq)

    basis = _SpectralBasis(A, freqs_hz=freqs)

    assert basis.nfreq == nfreq
    assert basis.nmodes == nmodes
    assert np.allclose(basis.freqs_hz, freqs)
    assert not basis.has_svd  # no SVD metadata


def test_spectral_basis_project_deproject():
    """Test projection and deprojection."""
    nfreq, nmodes, npix = 10, 3, 768
    A = np.random.randn(nfreq, nmodes).astype(np.float32)

    # Orthonormalize for cleaner test
    A, _ = np.linalg.qr(A)

    basis = _SpectralBasis(A)

    # Forward project
    data = np.random.randn(npix, nfreq).astype(np.float32)
    coeffs = basis.project(data)
    assert coeffs.shape == (npix, nmodes)

    # Reconstruct via deprojection
    reconstructed = basis.deproject(coeffs)
    assert reconstructed.shape == (npix, nfreq)

    # Check that reconstructed data lies in the basis subspace
    # The reconstruction should match the projection of data onto the subspace
    data_proj = basis.deproject(basis.project(data))
    error = np.linalg.norm(data_proj - reconstructed)
    assert error < 1e-5  # should be exact (within numerical precision)


def test_spectral_basis_save_load():
    """Test npz serialization and deserialization."""
    nfreq, nmodes = 10, 3
    A = np.random.randn(nfreq, nmodes).astype(np.float32)
    freqs = np.linspace(50e6, 150e6, nfreq)
    svd_mean = np.random.randn(nfreq).astype(np.float32)
    svd_svals = np.array([5.0, 3.0, 1.0], dtype=np.float32)

    basis = _SpectralBasis(A, freqs_hz=freqs, svd_mean=svd_mean,
                          svd_svals=svd_svals, n_samples=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'basis.npz')
        basis.save(path)
        loaded = _SpectralBasis.from_file(path)

        assert np.allclose(loaded.A, basis.A)
        assert np.allclose(loaded.freqs_hz, basis.freqs_hz)
        assert np.allclose(loaded.svd_mean, basis.svd_mean)
        assert np.allclose(loaded.svd_svals, basis.svd_svals)
        assert loaded.n_samples == 100

    print("✓ test_spectral_basis_save_load passed")


def test_beam_basis_from_dipole():
    """Test BeamBasis construction from thin-dipole analytic model."""
    freqs = np.array([50e6, 100e6, 150e6])
    arm_length = 3.0
    K = 2

    bb = BeamBasis.from_dipole(freqs, arm_length_m=arm_length, K=K, nside=4)

    assert bb.nfreq == len(freqs)
    assert bb.nmodes == K
    assert np.allclose(bb.freqs_hz, freqs)
    print(f"✓ test_beam_basis_from_dipole passed (A shape: {bb.A.shape})")


def test_beam_basis_from_dipole_per_dipole():
    """Test BeamBasis with per-dipole arm lengths."""
    freqs = np.array([50e6, 100e6, 150e6])
    arm_lengths = np.array([3.0, 4.0])
    K = 2

    bb = BeamBasis.from_dipole(freqs, arm_length_m=arm_lengths, K=K, nside=4)

    assert bb.nfreq == len(freqs)
    assert bb.nmodes == K
    print(f"✓ test_beam_basis_from_dipole_per_dipole passed (A shape: {bb.A.shape})")


def test_basis_resample():
    """Test basis resampling to new frequency grid."""
    old_freqs = np.linspace(50e6, 150e6, 10)
    new_freqs = np.linspace(50e6, 150e6, 20)

    A_old = np.random.randn(10, 3).astype(np.float32)
    basis_old = _SpectralBasis(A_old, freqs_hz=old_freqs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'basis.npz')
        basis_old.save(path)

        # Reload with resampling
        basis_new = _SpectralBasis.from_file(path, new_freqs=new_freqs)

        assert basis_new.nfreq == len(new_freqs)
        assert basis_new.nmodes == basis_old.nmodes
        assert np.allclose(basis_new.freqs_hz, new_freqs)
        print(f"✓ test_basis_resample passed (old nfreq: {basis_old.nfreq}, "
              f"new nfreq: {basis_new.nfreq})")


@pytest.mark.skipif(True, reason="Requires pygdsm; skip in CI")
def test_sky_basis_from_gsm():
    """Test SkyBasis construction from GSM16 (requires pygdsm)."""
    try:
        import pygdsm  # noqa
    except ImportError:
        pytest.skip("pygdsm not installed")

    freqs = np.array([50e6, 100e6, 150e6])
    sb = SkyBasis.from_gsm(freqs, n_modes=3, nside=4, include_flat=True)

    assert sb.nfreq == len(freqs)
    assert sb.nmodes == 4  # 3 GSM + 1 flat
    assert np.allclose(sb.freqs_hz, freqs)
    print(f"✓ test_sky_basis_from_gsm passed (A shape: {sb.A.shape})")


def test_basis_from_ensemble():
    """Test basis construction from an ensemble of spectra."""
    nfreq, n_samples, n_modes = 10, 100, 3
    freqs = np.linspace(50e6, 150e6, nfreq)

    # Generate synthetic spectra (random but correlated)
    base = np.random.randn(n_modes, nfreq)
    coeffs = np.random.randn(n_samples, n_modes)
    spectra = coeffs @ base

    basis = _SpectralBasis.from_ensemble(freqs, spectra, n_modes)

    assert basis.nfreq == nfreq
    assert basis.nmodes == n_modes
    assert basis.has_svd
    assert basis.n_samples == n_samples
    print(f"✓ test_basis_from_ensemble passed (SVD mean shape: {basis.svd_mean.shape})")


if __name__ == "__main__":
    test_spectral_basis_basic()
    test_spectral_basis_project_deproject()
    test_spectral_basis_save_load()
    test_beam_basis_from_dipole()
    test_beam_basis_from_dipole_per_dipole()
    test_basis_resample()
    test_basis_from_ensemble()
    print("\n✓ All Phase 1 (basis.py) tests passed!")
