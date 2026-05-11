"""
Tests for eigsep_sim.sim_jax — JAX kernels (_beam_sum, _src_sum) and SH helpers.

Simulator, sim_spin, and sim_azalt_sh were removed; their functionality is now
covered by ForwardModel (test_simulate.py) with the new coefficient-based API.
"""

import numpy as np
import pytest
import jax.numpy as jnp
import healpy

from eigsep_sim.sim_jax import (
    _beam_sum, _src_sum,
    _sh_coupling_modes, _sh_fft_spin,
)
from eigsep_sim.healpix import float_dtype


# ---------------------------------------------------------------------------
# _beam_sum
# ---------------------------------------------------------------------------

class TestBeamSum:
    """
    Analytically tractable case: uniform beam (all weights = 1) over a
    uniform sky of temperature T.  The beam-weighted integral is:

        num = Σ 1·T  =  npix·T
        den = Σ 1    =  npix
        T_ant = T
    """

    def _make_inputs(self, nside, nfreq, T_sky):
        npix = healpy.nside2npix(nside)
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        sky = jnp.full((npix, nfreq), T_sky, dtype=float_dtype)
        crds = jnp.array(
            healpy.pix2vec(nside, np.arange(npix)), dtype=float_dtype
        )
        rot_ms = jnp.eye(3, dtype=float_dtype)[None, ...]  # (1, 3, 3)
        return beam_map, sky, crds, rot_ms

    def test_recovers_sky_temperature(self):
        nside, nfreq, T = 8, 2, 100.0
        beam_map, sky, crds, rot_ms = self._make_inputs(nside, nfreq, T)
        num, den = _beam_sum(nside, beam_map, sky, crds, rot_ms)
        T_ant = np.asarray(num / den)
        np.testing.assert_allclose(T_ant[0], T, rtol=1e-5)

    def test_multiple_orientations_uniform(self):
        """Any rotation of a uniform beam over a uniform sky still gives T."""
        nside, nfreq, T = 8, 1, 42.0
        npix = healpy.nside2npix(nside)
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        sky = jnp.full((npix, nfreq), T, dtype=float_dtype)
        crds = jnp.array(
            healpy.pix2vec(nside, np.arange(npix)), dtype=float_dtype
        )
        n_orient = 4
        rot_ms = jnp.broadcast_to(
            jnp.eye(3, dtype=float_dtype), (n_orient, 3, 3)
        )
        num, den = _beam_sum(nside, beam_map, sky, crds, rot_ms)
        T_ant = np.asarray(num / den)
        np.testing.assert_allclose(T_ant[:, 0], T, rtol=1e-5)

    def test_zero_sky_gives_zero_num(self):
        nside, nfreq = 8, 2
        npix = healpy.nside2npix(nside)
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        sky = jnp.zeros((npix, nfreq), dtype=float_dtype)
        crds = jnp.array(
            healpy.pix2vec(nside, np.arange(npix)), dtype=float_dtype
        )
        rot_ms = jnp.eye(3, dtype=float_dtype)[None, ...]
        num, den = _beam_sum(nside, beam_map, sky, crds, rot_ms)
        np.testing.assert_allclose(np.asarray(num), 0.0, atol=1e-6)

    def test_output_shapes(self):
        nside, nfreq, n_orient = 8, 3, 5
        npix = healpy.nside2npix(nside)
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        sky = jnp.zeros((npix, nfreq), dtype=float_dtype)
        crds = jnp.array(
            healpy.pix2vec(nside, np.arange(npix)), dtype=float_dtype
        )
        rot_ms = jnp.broadcast_to(
            jnp.eye(3, dtype=float_dtype), (n_orient, 3, 3)
        )
        num, den = _beam_sum(nside, beam_map, sky, crds, rot_ms)
        assert num.shape == (n_orient, nfreq)
        assert den.shape == (n_orient, nfreq)


# ---------------------------------------------------------------------------
# _src_sum
# ---------------------------------------------------------------------------

class TestSrcSum:
    """
    Analytically tractable: uniform beam (all weights = 1), single source
    with flux T at the north pole.  The beam weight at that direction = 1,
    so the contribution is T·1 = T.
    """

    def test_single_source_uniform_beam(self):
        nside, nfreq = 8, 2
        npix = healpy.nside2npix(nside)
        T_src = 500.0
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        src_vecs = jnp.array([[0.0], [0.0], [1.0]], dtype=float_dtype)
        src_flux = jnp.full((1, nfreq), T_src, dtype=float_dtype)
        rot_ms = jnp.eye(3, dtype=float_dtype)[None, ...]
        num = _src_sum(nside, beam_map, src_vecs, src_flux, rot_ms)
        np.testing.assert_allclose(np.asarray(num[0]), T_src, rtol=1e-4)

    def test_multiple_sources_sum(self):
        """Multiple sources with uniform beam: result is sum of all fluxes."""
        nside, nfreq = 8, 1
        npix = healpy.nside2npix(nside)
        fluxes = np.array([10.0, 20.0, 30.0])
        # Three sources at well-separated directions
        src_vecs = jnp.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float_dtype
        ).T  # (3, 3)
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        src_flux = jnp.array(fluxes[:, None], dtype=float_dtype)
        rot_ms = jnp.eye(3, dtype=float_dtype)[None, ...]
        num = _src_sum(nside, beam_map, src_vecs, src_flux, rot_ms)
        # Each source gets weight ≈ 1, total ≈ sum(fluxes)
        np.testing.assert_allclose(
            float(num[0, 0]), fluxes.sum(), rtol=0.01
        )

    def test_output_shape(self):
        nside, nfreq, n_orient, n_src = 8, 3, 4, 2
        npix = healpy.nside2npix(nside)
        beam_map = jnp.ones((npix, nfreq), dtype=float_dtype)
        src_vecs = jnp.zeros((3, n_src), dtype=float_dtype)
        src_flux = jnp.zeros((n_src, nfreq), dtype=float_dtype)
        rot_ms = jnp.broadcast_to(
            jnp.eye(3, dtype=float_dtype), (n_orient, 3, 3)
        )
        num = _src_sum(nside, beam_map, src_vecs, src_flux, rot_ms)
        assert num.shape == (n_orient, nfreq)


# ---------------------------------------------------------------------------
# SH + FFT helpers
# ---------------------------------------------------------------------------

class TestSHHelpers:
    """Tests for the spherical-harmonic / FFT spin-sweep helpers."""

    def _make_alm(self, nside, value, lmax):
        """Return alm for a constant map = value."""
        import healpy
        npix = healpy.nside2npix(nside)
        m = np.full(npix, value, dtype=np.float64)
        return healpy.map2alm(m, lmax=lmax, use_pixel_weights=False)

    def test_coupling_modes_constant_maps(self):
        """For two constant maps, C_m should be zero for m > 0."""
        import healpy
        nside, lmax = 8, 16
        beam_alm = self._make_alm(nside, 1.0, lmax)
        sky_alm = self._make_alm(nside, 50.0, lmax)
        C_pos = _sh_coupling_modes(beam_alm, sky_alm, lmax)
        # For constant maps only l=0, m=0 is non-zero → C_m=0 for m>0
        np.testing.assert_allclose(np.abs(C_pos[1:]), 0.0, atol=1e-6)

    def test_coupling_modes_shape(self):
        import healpy
        nside, lmax = 8, 16
        alm = self._make_alm(nside, 1.0, lmax)
        C_pos = _sh_coupling_modes(alm, alm, lmax)
        assert C_pos.shape == (lmax + 1,)

    def test_fft_spin_constant_map(self):
        """Constant sky + constant beam → T_ant constant over spin sweep."""
        import healpy
        nside, lmax, n_phi = 8, 16, 32
        npix = healpy.nside2npix(nside)
        T_sky = 75.0
        beam_val = 1.0
        beam_alm = self._make_alm(nside, beam_val, lmax)
        sky_alm = self._make_alm(nside, T_sky, lmax)
        C_pos = _sh_coupling_modes(beam_alm, sky_alm, lmax)
        beam_solid_angle = beam_val * 4.0 * np.pi  # integral of 1 over sphere
        T_ant = _sh_fft_spin(C_pos, n_phi, beam_solid_angle)
        # Uniform sky → T_ant = T_sky for all spin angles
        np.testing.assert_allclose(T_ant, T_sky, rtol=0.01)

    def test_fft_spin_shape(self):
        import healpy
        nside, lmax, n_phi = 8, 16, 64
        alm = self._make_alm(nside, 1.0, lmax)
        C_pos = _sh_coupling_modes(alm, alm, lmax)
        T_ant = _sh_fft_spin(C_pos, n_phi, 4.0 * np.pi)
        assert T_ant.shape == (n_phi,)

    def test_fft_spin_matches_pixel_domain(self):
        """SH+FFT spin sweep must agree with pixel-domain scan at lmax=2*nside."""
        import healpy
        nside, lmax, n_phi = 8, 16, 8
        npix = healpy.nside2npix(nside)

        # Non-trivial sky: linear gradient in z
        pix_vecs = np.stack(healpy.pix2vec(nside, np.arange(npix)), axis=0)
        sky_np = (1.0 + pix_vecs[2]).astype(np.float32)   # T(n) = 1 + cos(theta)
        beam_np = np.ones(npix, dtype=np.float32)

        # SH+FFT
        beam_alm = healpy.map2alm(beam_np.astype(np.float64), lmax=lmax,
                                   use_pixel_weights=False)
        sky_alm = healpy.map2alm(sky_np.astype(np.float64), lmax=lmax,
                                  use_pixel_weights=False)
        C_pos = _sh_coupling_modes(beam_alm, sky_alm, lmax)
        beam_solid_angle = np.sum(beam_np) * (4.0 * np.pi / npix)
        T_fft = _sh_fft_spin(C_pos, n_phi, beam_solid_angle)

        # Pixel-domain reference (uniform beam → T_ant = mean sky)
        # For any spin of the uniform beam over this sky, T_ant = mean(sky)
        T_mean = float(np.mean(sky_np))
        # 1 + cos(theta) averaged over sphere = 1 (cos averages to 0)
        np.testing.assert_allclose(T_fft, T_mean, rtol=0.05)

