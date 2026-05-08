#!/usr/bin/env python
"""
Test suite for sky.py — Sky brightness descriptor with spectral basis.

Tests verify:
1. Sky basic construction and properties
2. Sky.from_map() factory
3. Sky.init_coeffs() initialization
4. SkyBasis integration
"""

import numpy as np
import pytest
import healpy

from eigsep_sim.sky import Sky
from eigsep_sim.basis import SkyBasis


def test_sky_basic():
    """Sky: basic construction."""
    nside = 4
    freqs_hz = np.array([50e6, 100e6, 150e6])
    A = np.random.randn(3, 2).astype(np.float32)
    basis = SkyBasis(A, freqs_hz=freqs_hz)

    sky = Sky(nside, freqs_hz, basis)

    assert sky.nside == nside
    assert np.allclose(sky.freqs_hz, freqs_hz)
    assert sky.basis is basis


def test_sky_properties():
    """Sky: npix and nmodes properties."""
    nside = 4
    nfreq = 3
    nmodes = 2
    freqs_hz = np.linspace(50e6, 150e6, nfreq)
    A = np.random.randn(nfreq, nmodes).astype(np.float32)
    basis = SkyBasis(A, freqs_hz=freqs_hz)

    sky = Sky(nside, freqs_hz, basis)

    assert sky.npix == healpy.nside2npix(nside)
    assert sky.nmodes == nmodes


def test_sky_from_map_basic():
    """Sky.from_map(): basic construction from synthetic map."""
    nside = 4
    npix = healpy.nside2npix(nside)
    nfreq = 5
    freqs_hz = np.linspace(50e6, 150e6, nfreq)
    n_modes = 2

    # Synthetic map: random data
    sky_map = np.random.randn(npix, nfreq).astype(np.float32)

    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=n_modes)

    assert sky.nside == nside
    assert np.allclose(sky.freqs_hz, freqs_hz)
    assert sky.nmodes == n_modes
    assert sky.basis is not None


def test_sky_from_map_rank_limited():
    """Sky.from_map(): handles rank limitation (K > min(npix, nfreq))."""
    nside = 2
    npix = healpy.nside2npix(nside)  # 48 pixels
    nfreq = 3
    freqs_hz = np.linspace(50e6, 150e6, nfreq)

    sky_map = np.random.randn(npix, nfreq).astype(np.float32)

    # Request more modes than available rank
    n_modes = 10
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=n_modes)

    # Should clamp to min(npix, nfreq) = 3
    assert sky.nmodes == min(npix, nfreq)


def test_sky_from_map_coefficients():
    """Sky.from_map(): generated coefficients have correct shape."""
    nside = 4
    npix = healpy.nside2npix(nside)
    nfreq = 5
    freqs_hz = np.linspace(50e6, 150e6, nfreq)
    n_modes = 2

    sky_map = np.random.randn(npix, nfreq).astype(np.float32)
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=n_modes)

    # Coefficients should project back to reconstruct map
    coeffs = sky.basis.project(sky_map)
    assert coeffs.shape == (npix, n_modes)


def test_sky_from_map_reconstruction():
    """Sky.from_map(): basis reconstructs the map in subspace."""
    nside = 4
    npix = healpy.nside2npix(nside)
    nfreq = 5
    freqs_hz = np.linspace(50e6, 150e6, nfreq)
    n_modes = 3

    sky_map = np.random.randn(npix, nfreq).astype(np.float32)
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=n_modes)

    # Project and reconstruct
    coeffs = sky.basis.project(sky_map)
    reconstructed = sky.basis.deproject(coeffs)

    # Reconstruction should have the same shape
    assert reconstructed.shape == (npix, nfreq)
    # Reconstruction should lie in the basis subspace (for full rank basis)
    # Check that projection of original map matches projection of reconstructed
    assert np.allclose(sky.basis.project(sky_map), sky.basis.project(reconstructed))


@pytest.mark.skipif(True, reason="Requires pygdsm; skip in CI")
def test_sky_from_gsm():
    """Sky.from_gsm(): basic construction."""
    try:
        import pygdsm  # noqa
    except ImportError:
        pytest.skip("pygdsm not installed")

    nside = 4
    freqs_hz = np.array([50e6, 100e6, 150e6])
    n_modes = 2

    sky = Sky.from_gsm(nside, freqs_hz, n_modes=n_modes, include_flat=True)

    assert sky.nside == nside
    assert np.allclose(sky.freqs_hz, freqs_hz)
    # With flat mode: n_modes + 1
    assert sky.nmodes == n_modes + 1


@pytest.mark.skipif(True, reason="Requires pygdsm; skip in CI")
def test_sky_from_gsm_no_flat():
    """Sky.from_gsm(): without flat mode."""
    try:
        import pygdsm  # noqa
    except ImportError:
        pytest.skip("pygdsm not installed")

    nside = 4
    freqs_hz = np.array([50e6, 100e6, 150e6])
    n_modes = 2

    sky = Sky.from_gsm(nside, freqs_hz, n_modes=n_modes, include_flat=False)

    assert sky.nmodes == n_modes


@pytest.mark.skipif(True, reason="Requires pygdsm; skip in CI")
def test_sky_init_coeffs():
    """Sky.init_coeffs(): generate initial coefficients from GSM."""
    try:
        import pygdsm  # noqa
    except ImportError:
        pytest.skip("pygdsm not installed")

    nside = 4
    npix = healpy.nside2npix(nside)
    freqs_hz = np.array([50e6, 100e6, 150e6])
    n_modes = 2

    sky = Sky.from_gsm(nside, freqs_hz, n_modes=n_modes, include_flat=True)

    coeffs = sky.init_coeffs()

    assert coeffs.shape == (npix, sky.nmodes)
    assert coeffs.dtype in (np.float32, np.float64)
    # Coefficients should be finite
    assert np.all(np.isfinite(coeffs))


@pytest.mark.skipif(True, reason="Requires pygdsm; skip in CI")
def test_sky_init_coeffs_reconstruction():
    """Sky.init_coeffs(): reconstructed sky has sensible values."""
    try:
        import pygdsm  # noqa
    except ImportError:
        pytest.skip("pygdsm not installed")

    nside = 4
    freqs_hz = np.array([50e6, 100e6, 150e6])
    n_modes = 3

    sky = Sky.from_gsm(nside, freqs_hz, n_modes=n_modes, include_flat=True)
    coeffs = sky.init_coeffs()

    # Reconstruct sky from coefficients
    reconstructed = sky.basis.deproject(coeffs)

    # Sky brightness should be positive (in Kelvin)
    assert np.all(reconstructed > 0)
    # Sky temperature should be in reasonable range (10 K to 10000 K)
    assert np.all(reconstructed < 10000)
    assert np.all(reconstructed > 1)


def test_sky_type_checking():
    """Sky: type checking on inputs."""
    nside = 4
    freqs_hz = np.array([50e6, 100e6, 150e6])
    A = np.random.randn(3, 2).astype(np.float32)
    basis = SkyBasis(A, freqs_hz=freqs_hz)

    # nside should be converted to int
    sky = Sky(np.float32(4.0), freqs_hz, basis)
    assert isinstance(sky.nside, (int, np.integer))

    # freqs_hz should be float64
    sky = Sky(4, freqs_hz, basis)
    assert sky.freqs_hz.dtype == np.float64


def test_sky_from_map_with_ensemble_basis():
    """Sky.from_map(): uses _SpectralBasis.from_ensemble internally."""
    nside = 4
    npix = healpy.nside2npix(nside)
    nfreq = 5
    freqs_hz = np.linspace(50e6, 150e6, nfreq)
    n_modes = 2

    sky_map = np.random.randn(npix, nfreq).astype(np.float32)
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=n_modes)

    # Basis should have SVD metadata
    assert sky.basis.has_svd


if __name__ == "__main__":
    test_sky_basic()
    test_sky_properties()
    test_sky_from_map_basic()
    test_sky_from_map_rank_limited()
    test_sky_from_map_coefficients()
    test_sky_from_map_reconstruction()
    test_sky_type_checking()
    test_sky_from_map_with_ensemble_basis()
    print("\n✓ All Phase 5 (sky.py) tests passed!")
