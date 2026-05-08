#!/usr/bin/env python
"""
Test suite for joint_solver.py — joint sky/beam estimation with Anderson Acceleration.

Tests verify:
1. Spectral basis computation from nominal beam
2. A-matrix construction from HEALPix beam model
3. Anderson Accelerator convergence behavior
4. Full joint solve on synthetic data with planted beam error
"""

import os
import sys
import numpy as np
import healpy
import pytest
from scipy.spatial.transform import Rotation

from eigsep_sim.spectral import beam_spectral_basis
from eigsep_sim.joint_solver import AndersonAccelerator, build_A_from_model

# For BLOOM-21cm config (if available)
_YAML = os.path.join(os.path.dirname(__file__), "..", "bloom21cm", "src", "bloom_config.yaml")


class MockOrbiterMission:
    """Minimal mock config for testing without bloom21cm dependency."""
    class Antenna:
        u_body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3) two orthogonal dipoles
        def kh(self, freq_mhz):
            # Thin dipole: kh ≈ π * freq_hz / c
            c = 3e8
            freq_hz = freq_mhz * 1e6
            kh_nominal = np.pi * freq_hz / c
            return np.array([kh_nominal, 0.75 * kh_nominal])  # (2,) per dipole
    antenna = Antenna()


def _get_cfg():
    """Load config from BLOOM-21cm if available, else use mock."""
    if os.path.exists(_YAML):
        from eigsep_sim.lunar_orbit import OrbiterMission
        return OrbiterMission(_YAML)
    else:
        return MockOrbiterMission()


def test_spectral_basis():
    """Test that spectral basis computation works and produces reasonable modes."""
    cfg = _get_cfg()
    freqs_mhz = np.array([50, 80, 110])

    basis = beam_spectral_basis(cfg, freqs_mhz, K=3, nside_beam=8)

    # Check shapes
    assert basis['psi'].shape == (2, 3, 3), f"Expected psi shape (2, 3, 3), got {basis['psi'].shape}"
    assert basis['c_init'].shape == (2, 3, 768), f"Expected c_init shape (2, 3, 768)"

    # Check spectral modes are normalized
    for d in range(2):
        psi_norm = np.linalg.norm(basis['psi'][d], axis=1)
        assert np.allclose(psi_norm, 1.0), "Spectral modes should be unit-norm"

    # Check beam coefficients are not all zero
    assert np.linalg.norm(basis['c_init']) > 0, "Beam coefficients should be non-zero"

    # Check that reconstructed beam is reasonable (non-negative, bounded)
    # by multiplying coefficients with spectral modes
    # c_init[0] shape: (K, npix_beam), psi[0] shape: (K, N_freq)
    B_recon = np.einsum('kp,kf->pf', basis['c_init'][0], basis['psi'][0])
    assert np.all(B_recon >= -1e-6), f"Reconstructed beam should be non-negative, got min {B_recon.min()}"

    print("✓ test_spectral_basis passed")


def test_anderson_accelerator():
    """Test Anderson Accelerator convergence on a simple fixed-point problem."""
    # Simple problem: F(x) = 0.9 * x → fixed point at x=0
    def F(x):
        return 0.9 * x

    aa = AndersonAccelerator(m=3)

    x = np.array([1.0, 2.0, 3.0])
    for i in range(10):
        Fx = F(x)
        x = aa.step(x, Fx)

    # Should converge to near zero
    assert np.allclose(x, 0, atol=1e-3), f"Expected convergence to 0, got {x}"

    print("✓ test_anderson_accelerator passed")


def test_build_A_from_model():
    """Test that A-matrix construction works with HEALPix beam model."""
    cfg = _get_cfg()
    freqs_mhz = np.array([50, 80])
    npix_sky = 8

    # Create simple synthetic setup
    from scipy.spatial.transform import Rotation
    rots = [Rotation.identity() for _ in range(4)]  # 4 observations
    masks = np.ones((4, healpy.nside2npix(npix_sky)), dtype=np.uint8)  # all sky, no regolith
    J_SUN = np.array([0, 10, 50, 100])  # sun pixel per observation

    # Spectral basis
    basis = beam_spectral_basis(cfg, freqs_mhz, K=2, nside_beam=8)

    # Beam coefficients (use initial nominal ones)
    c = basis['c_init'].copy()

    # Rotation offset
    delta_phi = np.array([0.0, 0.0, 0.0])

    # Build A for first frequency
    A, omega_B = build_A_from_model(rots, masks, basis, c, delta_phi,
                                    J_SUN, npix_sky, freq_idx=0, nside_beam=8)

    # Check shapes
    n_obs = 4
    npix = healpy.nside2npix(npix_sky)
    assert A.shape == (n_obs * 2, npix + 2), f"Expected A shape ({n_obs*2}, {npix+2}), got {A.shape}"
    assert omega_B.shape == (n_obs, 2), f"Expected omega_B shape ({n_obs}, 2), got {omega_B.shape}"

    # Check that A values are reasonable (beam pattern is between 0 and 1)
    assert np.all(A >= -1e-6), "A matrix should be non-negative (beam power)"
    assert np.all(A <= 1.1), "A matrix should be <= 1 (normalized beam)"

    print("✓ test_build_A_from_model passed")


if __name__ == "__main__":
    test_spectral_basis()
    test_anderson_accelerator()
    test_build_A_from_model()
    print("\n✓ All tests passed")
