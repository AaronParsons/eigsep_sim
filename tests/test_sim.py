"""
Tests for eigsep_sim.sim — JAX kernels (_beam_sum, _src_sum) and Simulator.
"""

import numpy as np
import pytest
import jax.numpy as jnp
import healpy
from astropy.time import Time

from eigsep_sim.sim import _beam_sum, _src_sum, Simulator
from eigsep_sim.beam import Beam
from eigsep_sim.healpix import float_dtype


# ---------------------------------------------------------------------------
# Minimal mock observer — identity rotation, all pixels above horizon
# ---------------------------------------------------------------------------

class MockObserver:
    """Trivial observer for testing: identity gal→top, all sky visible."""
    def __init__(self):
        self.time = None

    def set_time(self, t):
        self.time = t

    def rot_gal2top(self):
        return np.eye(3, dtype=np.float32)

    def above_horizon(self, nside):
        return np.ones(healpy.nside2npix(nside), dtype=bool)


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
# Simulator
# ---------------------------------------------------------------------------

class TestSimulator:
    """Integration tests using MockObserver (no GSM, no terrain, no catalog)."""

    def _make_sim(self, nside=8, nfreq=4, monopole=None):
        freqs = np.linspace(50e6, 150e6, nfreq, dtype=np.float32)
        observer = MockObserver()
        beam = Beam(freqs, beam_type='dipole', nside=nside, peak_normalize=True)
        sim = Simulator(
            observer, freqs, beam,
            nside=nside, gsm=False,
            monopole=monopole,
        )
        return sim, freqs

    def test_trx_only_zero_sky(self):
        """With no sky emission, vis = Trx at all frequencies."""
        Trx = 75.0
        sim, freqs = self._make_sim()
        times = [Time('2024-01-01')]
        azalts = np.zeros((1, 2), dtype=np.float32)
        vis = sim.sim(times, azalts=azalts, Trx=Trx, S11=0.0, bandpass=1.0)
        assert vis.shape == (1, 1, len(freqs))
        np.testing.assert_allclose(vis[0, 0], Trx, rtol=0.05)

    def test_S11_scales_sky_contribution(self):
        """
        With Trx=0, S11=0 gives full T_ant; S11=0.5 gives half.
        vis = bandpass * (S12 * T_ant + Trx)
            S11=0.0 → vis_full = T_ant
            S11=0.5 → vis_half = 0.5 * T_ant
        """
        T_monopole = 100.0
        monopole = np.full(4, T_monopole, dtype=np.float32)
        sim, freqs = self._make_sim(monopole=monopole)
        times = [Time('2024-01-01')]
        azalts = np.zeros((1, 2), dtype=np.float32)

        vis_full = sim.sim(times, azalts=azalts, Trx=0.0, S11=0.0)
        vis_half = sim.sim(times, azalts=azalts, Trx=0.0, S11=0.5)
        np.testing.assert_allclose(vis_half, 0.5 * vis_full, rtol=1e-5)

    def test_bandpass_scaling(self):
        """bandpass=0.5 should halve all output values."""
        Trx = 50.0
        sim, freqs = self._make_sim()
        times = [Time('2024-01-01')]
        azalts = np.zeros((1, 2), dtype=np.float32)

        vis_1 = sim.sim(times, azalts=azalts, Trx=Trx, bandpass=1.0)
        vis_p = sim.sim(times, azalts=azalts, Trx=Trx, bandpass=0.5)
        np.testing.assert_allclose(vis_p, 0.5 * vis_1, rtol=1e-5)

    def test_output_shape_multiple_times_and_orients(self):
        sim, freqs = self._make_sim(nfreq=3)
        n_times = 2
        n_orient = 3
        times = [Time('2024-01-01'), Time('2024-01-02')]
        azalts = np.zeros((n_orient, 2), dtype=np.float32)
        vis = sim.sim(times, azalts=azalts, Trx=10.0)
        assert vis.shape == (n_times, n_orient, 3)

    def test_monopole_temperature_recovered(self):
        """
        A perfectly uniform sky of temperature T with a beam-weighted
        integral gives T_ant = T (beam cancels in numerator/denominator).
        """
        T_monopole = 150.0
        monopole = np.full(4, T_monopole, dtype=np.float32)
        sim, freqs = self._make_sim(monopole=monopole)
        times = [Time('2024-01-01')]
        azalts = np.zeros((1, 2), dtype=np.float32)
        vis = sim.sim(times, azalts=azalts, Trx=0.0, S11=0.0, bandpass=1.0)
        # Beam over uniform sky → T_ant = T_monopole
        np.testing.assert_allclose(vis[0, 0], T_monopole, rtol=0.05)
