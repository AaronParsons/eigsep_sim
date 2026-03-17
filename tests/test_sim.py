"""
Tests for eigsep_sim.sim — JAX kernels (_beam_sum, _src_sum) and Simulator.
"""

import numpy as np
import pytest
import jax.numpy as jnp
import healpy
from astropy.time import Time

from eigsep_sim.sim import (
    _beam_sum, _src_sum, Simulator,
    _sh_coupling_modes, _sh_fft_spin,
)
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


# ---------------------------------------------------------------------------
# Simulator.sim_spin
# ---------------------------------------------------------------------------

class TestSimulatorSpinSweep:
    """Tests for Simulator.sim_spin (SH+FFT spin-sweep method)."""

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

    def test_output_shape(self):
        n_phi = 16
        sim, freqs = self._make_sim(nfreq=3)
        times = [Time('2024-01-01'), Time('2024-01-02')]
        vis = sim.sim_spin(times, n_phi=n_phi, Trx=10.0)
        assert vis.shape == (2, n_phi, 3)

    def test_uniform_sky_constant_spin(self):
        """Uniform sky → T_ant should be (near-)constant across all spin angles."""
        T_monopole = 100.0
        monopole = np.full(4, T_monopole, dtype=np.float32)
        sim, freqs = self._make_sim(monopole=monopole)
        times = [Time('2024-01-01')]
        n_phi = 32
        vis = sim.sim_spin(times, n_phi=n_phi, Trx=0.0, S11=0.0)
        # All spin angles should give roughly the same T_ant
        assert vis.shape == (1, n_phi, 4)
        for fi in range(4):
            rms_var = float(np.std(vis[0, :, fi]))
            assert rms_var < 5.0, f"Spin variation too large at freq {fi}: {rms_var:.2f}"

    def test_trx_added(self):
        """Trx offset should appear uniformly in all spin angles."""
        Trx = 50.0
        sim, freqs = self._make_sim()
        times = [Time('2024-01-01')]
        vis_no_trx = sim.sim_spin(times, n_phi=8, Trx=0.0)
        vis_trx = sim.sim_spin(times, n_phi=8, Trx=Trx)
        np.testing.assert_allclose(vis_trx - vis_no_trx, Trx, atol=1.0)
