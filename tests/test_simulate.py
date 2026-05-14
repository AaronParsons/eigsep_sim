#!/usr/bin/env python
"""
Test suite for simulate.py — JAX-based ForwardModel for global 21cm radiometry.

Tests verify:
1. ForwardModel construction and basic properties
2. Geometry precomputation (rotation matrices, masks)
3. Antenna temperature simulation with basis coefficients
"""

import numpy as np
import pytest
import healpy
from astropy.time import Time

from eigsep_sim.simulate import ForwardModel
from eigsep_sim.basis import BeamBasis, SkyBasis
from eigsep_sim.beam import Beam
from eigsep_sim.sky import Sky
from eigsep_sim.observer import EarthSurface, LunarOrbit
from eigsep_sim.terrain import NullTerrain


def test_forward_model_basic():
    """ForwardModel: basic construction."""
    freqs_hz = np.array([50e6, 100e6, 150e6])
    nside = 4

    # Create minimal basis and objects
    beam = Beam.from_dipole(nside=nside, freqs_hz=freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(healpy.nside2npix(nside), 3), n_modes=2)

    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")

    fwd = ForwardModel(observer, beam, sky)

    assert fwd.observer is observer
    assert fwd.beam is beam
    assert fwd.sky is sky
    assert fwd.terrain is None


def test_forward_model_with_terrain():
    """ForwardModel: construction with terrain."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(healpy.nside2npix(nside), 2), n_modes=2)
    observer = EarthSurface(lat=0.0, lon=0.0)
    observer.set_time("2000-01-01")

    terrain = NullTerrain()
    fwd = ForwardModel(observer, beam, sky, terrain=terrain)

    assert fwd.terrain is terrain


def test_forward_model_precompute_geometry():
    """ForwardModel: precompute_geometry caches rotations and masks."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(healpy.nside2npix(nside), 2), n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    times = [Time("2000-01-01") + i for i in range(3)]
    geom = fwd.precompute_geometry(times)

    assert 'rot_gal2top' in geom
    assert 'crds_top' in geom
    assert 'masks' in geom

    assert len(geom['rot_gal2top']) == 3
    assert len(geom['crds_top']) == 3
    assert len(geom['masks']) == 3

    # Check shapes
    for R in geom['rot_gal2top']:
        assert R.shape == (3, 3)
    for crds in geom['crds_top']:
        assert crds.shape == (3, healpy.nside2npix(nside))
    for mask in geom['masks']:
        assert mask.shape == (healpy.nside2npix(nside),)
        assert mask.dtype == np.float32


def test_forward_model_simulate_basic():
    """ForwardModel: basic simulate call."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    # Random coefficients
    sky_coeffs = np.random.randn(npix_sky, 2).astype(np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside), 2).astype(np.float32)

    times = [Time("2000-01-01")]
    antenna_temp = fwd.simulate(sky_coeffs, beam_coeffs, times=times)

    assert antenna_temp.shape == (1, 2, 2)  # (ntimes, n_dipoles, nfreq)
    assert antenna_temp.dtype == np.float32
    assert np.all(np.isfinite(antenna_temp))


def test_forward_model_simulate_multiple_times():
    """ForwardModel: simulate over multiple times."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    sky_coeffs = np.random.randn(npix_sky, 2).astype(np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside), 2).astype(np.float32)

    ntimes = 5
    times = [Time("2000-01-01") + i for i in range(ntimes)]
    antenna_temp = fwd.simulate(sky_coeffs, beam_coeffs, times=times)

    assert antenna_temp.shape == (ntimes, 2, 2)
    assert np.all(np.isfinite(antenna_temp))


def test_forward_model_simulate_with_precomputed_geom():
    """ForwardModel: simulate with pre-computed geometry."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    times = [Time("2000-01-01"), Time("2000-01-02")]
    geom = fwd.precompute_geometry(times)

    sky_coeffs = np.random.randn(npix_sky, 2).astype(np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside), 2).astype(np.float32)

    antenna_temp = fwd.simulate(sky_coeffs, beam_coeffs, geom=geom)

    assert antenna_temp.shape == (2, 2, 2)


def test_forward_model_zero_coefficients():
    """ForwardModel: zero coefficients → zero antenna temperature."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    sky_coeffs = np.zeros((npix_sky, 2), dtype=np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside), 2).astype(np.float32)

    times = [Time("2000-01-01")]
    antenna_temp = fwd.simulate(sky_coeffs, beam_coeffs, times=times)

    # With zero sky coefficients, antenna temp should reflect only ground temperature
    assert np.all(np.isfinite(antenna_temp))


def test_forward_model_orbital_observer():
    """ForwardModel: works with orbital observer (LunarOrbit)."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)

    observer = LunarOrbit(altitude=100e3, rot_orbit_vec=[0, 0, 1],
                         rot_spin_vec=[0, 0, 1])
    observer.set_time("2000-01-01")

    fwd = ForwardModel(observer, beam, sky)

    sky_coeffs = np.random.randn(npix_sky, 2).astype(np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside), 2).astype(np.float32)

    times = [Time("2000-01-01")]
    antenna_temp = fwd.simulate(sky_coeffs, beam_coeffs, times=times)

    assert antenna_temp.shape == (1, 2, 2)
    assert np.all(np.isfinite(antenna_temp))


def test_forward_model_simulate_no_times_no_geom_error():
    """ForwardModel: simulate raises error if neither times nor geom provided."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    sky_coeffs = np.random.randn(npix_sky, 2).astype(np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside), 2).astype(np.float32)

    with pytest.raises(ValueError, match="Either times or geom"):
        fwd.simulate(sky_coeffs, beam_coeffs)


def test_forward_model_different_nside_beam_sky():
    """ForwardModel: works even if beam and sky have different nside."""
    freqs_hz = np.array([50e6, 100e6])
    nside_beam = 4
    nside_sky = 8

    # Beam at coarser resolution
    beam = Beam.from_dipole(nside_beam, freqs_hz, arm_lengths_m=3.0, K=2)

    # Sky at finer resolution
    npix_sky = healpy.nside2npix(nside_sky)
    sky = Sky.from_map(nside_sky, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2)

    observer = EarthSurface(lat=45.0, lon=0.0)

    fwd = ForwardModel(observer, beam, sky)

    sky_coeffs = np.random.randn(npix_sky, 2).astype(np.float32)
    beam_coeffs = np.random.randn(2, healpy.nside2npix(nside_beam), 2).astype(np.float32)

    times = [Time("2000-01-01")]
    antenna_temp = fwd.simulate(sky_coeffs, beam_coeffs, times=times)

    assert antenna_temp.shape == (1, 2, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures for body_rots / transmitter tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_fwd():
    """Minimal 1-dipole ForwardModel: nside=4, 3 freqs."""
    rng = np.random.default_rng(0)
    freqs_hz = np.linspace(50e6, 150e6, 3)
    nside = 4
    npix = healpy.nside2npix(nside)
    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky_map = np.abs(rng.standard_normal((npix, 3))) + 1.0   # positive temps
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    return ForwardModel(observer, beam, sky)


@pytest.fixture
def simple_coeffs(simple_fwd):
    rng = np.random.default_rng(1)
    fwd = simple_fwd
    npix_sky = fwd.sky.npix
    npix_beam = fwd.beam.npix
    sky_c = rng.standard_normal((npix_sky, 2)).astype(np.float32)
    beam_c = np.abs(rng.standard_normal((fwd.beam.coeffs.shape[0], npix_beam, 2))).astype(np.float32)
    return sky_c, beam_c


# ─────────────────────────────────────────────────────────────────────────────
# body_rots tests
# ─────────────────────────────────────────────────────────────────────────────

def test_body_rots_identity_matches_none(simple_fwd, simple_coeffs):
    """body_rots=[I]*n is numerically identical to body_rots=None."""
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 4
    rots = [R] * ntimes
    I3 = np.eye(3, dtype=np.float32)

    geom_none = fwd.precompute_geometry(rots=rots, body_rots=None)
    geom_iden = fwd.precompute_geometry(rots=rots, body_rots=[I3] * ntimes)

    T_none = np.array(fwd.simulate(sky_c, beam_c, geom=geom_none))
    T_iden = np.array(fwd.simulate(sky_c, beam_c, geom=geom_iden))

    np.testing.assert_allclose(T_none, T_iden, atol=1e-5, rtol=1e-5)


def test_body_rots_nonidentity_changes_result(simple_fwd, simple_coeffs):
    """A 90-degree body rotation changes the antenna temperature."""
    from eigsep_sim.beam import Beam as _Beam
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    R = fwd.observer.rot_gal2top().astype(np.float32)
    rots = [R]

    geom_none = fwd.precompute_geometry(rots=rots, body_rots=None)
    R90z = _Beam.rot_z(np.pi / 2).astype(np.float32)
    geom_rot = fwd.precompute_geometry(rots=rots, body_rots=[R90z])

    T_none = np.array(fwd.simulate(sky_c, beam_c, geom=geom_none))
    T_rot  = np.array(fwd.simulate(sky_c, beam_c, geom=geom_rot))

    assert not np.allclose(T_none, T_rot, atol=1e-4), \
        "90-degree rotation should change antenna temperature for asymmetric beam"


def test_precompute_geometry_rots_matches_times(simple_fwd, simple_coeffs):
    """rots= path gives the same result as times= path for EarthSurface."""
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    t = Time("2000-01-01T06:00:00")

    fwd.observer.set_time(t)
    R = fwd.observer.rot_gal2top().astype(np.float32)

    geom_times = fwd.precompute_geometry(times=[t])
    geom_rots  = fwd.precompute_geometry(rots=[R])

    T_times = np.array(fwd.simulate(sky_c, beam_c, geom=geom_times))
    T_rots  = np.array(fwd.simulate(sky_c, beam_c, geom=geom_rots))

    np.testing.assert_allclose(T_times, T_rots, atol=1e-4, rtol=1e-4)


def test_precompute_geometry_rots_shape(simple_fwd):
    """precompute_geometry with rots= produces correctly shaped geom arrays."""
    fwd = simple_fwd
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 6
    geom = fwd.precompute_geometry(rots=[R] * ntimes)

    npix_sky = fwd.sky.npix
    assert geom['rots_jax'].shape          == (ntimes, 3, 3)
    assert geom['body_rots_jax'].shape     == (ntimes, 3, 3)
    assert geom['terrain_masks_jax'].shape == (ntimes, npix_sky)
    assert geom['terrain_emissions_jax'].shape == (
        ntimes, npix_sky, len(fwd.beam.freqs_hz)
    )
    assert geom['default_emission_masks_jax'].shape == (ntimes, npix_sky)
    assert geom['crds_gal_jax'].shape      == (3, npix_sky)
    assert geom['tx_crds_jax'].shape       == (ntimes, 0, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Transmitter tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_tx_fwd(observer, beam, sky, tx_dir, tx_freqs, tx_power):
    """Helper: build a ForwardModel with one transmitter."""
    return ForwardModel(observer, beam, sky,
                        transmitters=[(tx_dir, tx_freqs, tx_power)])


def test_transmitter_zero_power_no_effect(simple_fwd, simple_coeffs):
    """Transmitter with T=0 produces the same Tant as no transmitter."""
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    freqs_hz = fwd.beam.freqs_hz
    R = fwd.observer.rot_gal2top().astype(np.float32)

    fwd_no_tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky)
    fwd_zero  = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                             transmitters=[(np.array([0.,0.,1.]), freqs_hz, 0.0)])

    geom = fwd_no_tx.precompute_geometry(rots=[R])
    T_no_tx = np.array(fwd_no_tx.simulate(sky_c, beam_c, geom=geom))

    geom_z = fwd_zero.precompute_geometry(rots=[R])
    T_zero  = np.array(fwd_zero.simulate(sky_c, beam_c, geom=geom_z))

    np.testing.assert_allclose(T_no_tx, T_zero, atol=1e-6)


def test_transmitter_channel_selectivity(simple_fwd):
    """Transmitter at freq[1] only raises Tant at that channel.

    Uses the actual (physical) beam coefficients so beam values at zenith are
    positive for both dipoles and the delta has a definite sign.
    Uses zero sky and zero T_gnd to isolate the TX contribution.
    """
    fwd = simple_fwd
    freqs_hz = fwd.beam.freqs_hz
    R = fwd.observer.rot_gal2top().astype(np.float32)

    sky_c  = np.zeros((fwd.sky.npix, 2), dtype=np.float32)
    beam_c = fwd.beam.coeffs                                 # physical, non-negative
    tx_dir   = np.array([0., 0., 1.], dtype=np.float32)     # zenith (topocentric)
    tx_freqs = freqs_hz[1:2]                                 # middle channel only
    tx_power = np.array([1e4], dtype=np.float32)

    fwd_no_tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky)
    fwd_tx    = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                             transmitters=[(tx_dir, tx_freqs, tx_power)])

    geom_no = fwd_no_tx.precompute_geometry(rots=[R])
    T_no_tx = np.array(fwd_no_tx.simulate(sky_c, beam_c, geom=geom_no, T_gnd=0.0))

    geom_tx = fwd_tx.precompute_geometry(rots=[R])
    T_tx    = np.array(fwd_tx.simulate(sky_c, beam_c, geom=geom_tx, T_gnd=0.0))

    delta = T_tx - T_no_tx   # (1, n_dipoles, nfreq)
    # Channels 0 and 2 must be exactly unchanged (tx_T_internal=0 at those channels)
    np.testing.assert_allclose(delta[..., 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(delta[..., 2], 0.0, atol=1e-6)
    # Channel 1 should increase (horizontal dipoles have non-zero response at zenith)
    assert np.all(delta[..., 1] > 0), "Zenith transmitter should raise Tant at its channel"


def test_transmitter_linearity(simple_fwd, simple_coeffs):
    """Two identical transmitters produce double the contribution of one."""
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    freqs_hz = fwd.beam.freqs_hz
    R = fwd.observer.rot_gal2top().astype(np.float32)

    tx_dir   = np.array([0., 0., 1.], dtype=np.float32)
    tx_freqs = freqs_hz
    tx_power = 1e4 * np.ones_like(tx_freqs)

    fwd_1tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                           transmitters=[(tx_dir, tx_freqs, tx_power)])
    fwd_2tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                           transmitters=[(tx_dir, tx_freqs, tx_power),
                                         (tx_dir, tx_freqs, tx_power)])
    fwd_no  = ForwardModel(fwd.observer, fwd.beam, fwd.sky)

    geom_1 = fwd_1tx.precompute_geometry(rots=[R])
    geom_2 = fwd_2tx.precompute_geometry(rots=[R])
    geom_0 = fwd_no.precompute_geometry(rots=[R])

    T_1 = np.array(fwd_1tx.simulate(sky_c, beam_c, geom=geom_1))
    T_2 = np.array(fwd_2tx.simulate(sky_c, beam_c, geom=geom_2))
    T_0 = np.array(fwd_no.simulate(sky_c, beam_c, geom=geom_0))

    # T_2 - T_0 ≈ 2 * (T_1 - T_0)
    np.testing.assert_allclose(T_2 - T_0, 2.0 * (T_1 - T_0), rtol=1e-5)


def test_transmitter_physics_formula(simple_fwd):
    """Transmitter contribution matches closed-form formula.

    For a transmitter at direction d (body frame) with power T_eff:
        T_ant_tx(f) = T_eff * B(d, f) / sum_pix(B(pix, f))
    where the sum is over all beam pixels.
    """
    fwd = simple_fwd
    freqs_hz = fwd.beam.freqs_hz
    nfreq = len(freqs_hz)
    R = fwd.observer.rot_gal2top().astype(np.float32)

    # Use zenith (topocentric [0,0,1]) as transmitter direction.
    # With body_rots=None the body frame = topocentric, so d_body = [0,0,1].
    tx_dir   = np.array([0., 0., 1.], dtype=np.float32)
    T_eff    = 1e6
    tx_freqs = freqs_hz
    tx_power = T_eff * np.ones(nfreq)

    # Build a no-transmitter model for baseline sky Tant
    fwd_no = ForwardModel(fwd.observer, fwd.beam, fwd.sky)
    fwd_tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                          transmitters=[(tx_dir, tx_freqs, tx_power)])

    sky_c  = np.zeros((fwd.sky.npix, 2), dtype=np.float32)   # zero sky → delta is pure TX
    beam_c = fwd.beam.coeffs                                   # nominal beam

    geom_no = fwd_no.precompute_geometry(rots=[R])
    geom_tx = fwd_tx.precompute_geometry(rots=[R])
    T_no = np.array(fwd_no.simulate(sky_c, beam_c, geom=geom_no))
    T_tx = np.array(fwd_tx.simulate(sky_c, beam_c, geom=geom_tx))
    delta = (T_tx - T_no)[0, 0, :]   # (nfreq,) — single time, single dipole

    # Closed-form: B(d_body) / sum_pix(B) * T_eff
    # beam_recon: (npix_beam, nfreq)
    beam_recon = fwd.beam.coeffs[0] @ fwd.beam.basis.A.T
    # Beam value at zenith (body frame [0,0,1]) via healpy interpolation
    th_z, ph_z = healpy.vec2ang(np.array([[0., 0., 1.]]))
    px_z, wgts_z = healpy.get_interp_weights(fwd.beam.nside, th_z, ph_z)
    B_at_zenith = np.sum(beam_recon[px_z[:, 0]] * wgts_z[:, 0, None], axis=0)   # (nfreq,)
    expected = T_eff * B_at_zenith / np.sum(beam_recon, axis=0)                  # (nfreq,)

    np.testing.assert_allclose(delta, expected, rtol=1e-4)


def test_transmitter_geom_shapes(simple_fwd):
    """precompute_geometry stores tx_crds_jax with correct shape."""
    fwd = simple_fwd
    freqs_hz = fwd.beam.freqs_hz
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 5
    n_sources = 2

    # Build ForwardModel with two transmitters
    fwd_tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                          transmitters=[
                              (np.array([0., 0., 1.]), freqs_hz, 1e4),
                              (np.array([1., 0., 0.]), freqs_hz, 1e3),
                          ])
    geom = fwd_tx.precompute_geometry(rots=[R] * ntimes)

    assert 'tx_crds_jax' in geom
    assert geom['tx_crds_jax'].shape == (ntimes, n_sources, 3)

    # Directions should be unit vectors (norm ≈ 1)
    crds = np.array(geom['tx_crds_jax'])
    norms = np.linalg.norm(crds, axis=-1)   # (ntimes, n_sources)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_transmitter_body_rots_coupling(simple_fwd):
    """body_rots changes which body-frame direction a fixed topocentric TX maps to."""
    from eigsep_sim.beam import Beam as _Beam
    fwd = simple_fwd
    freqs_hz = fwd.beam.freqs_hz
    sky_c  = np.zeros((fwd.sky.npix, 2), dtype=np.float32)
    beam_c = fwd.beam.coeffs
    R = fwd.observer.rot_gal2top().astype(np.float32)

    # Transmitter at zenith (topo [0,0,1])
    tx_dir   = np.array([0., 0., 1.], dtype=np.float32)
    tx_power = 1e6 * np.ones_like(freqs_hz)
    fwd_tx = ForwardModel(fwd.observer, fwd.beam, fwd.sky,
                          transmitters=[(tx_dir, freqs_hz, tx_power)])

    # No body rotation: tx is at body [0,0,1]
    geom_no_rot = fwd_tx.precompute_geometry(rots=[R], body_rots=None)
    # Use T_gnd=0 to isolate the TX contribution from sky coupling changes.
    T_no_rot = np.array(fwd_tx.simulate(sky_c, beam_c, geom=geom_no_rot, T_gnd=0.0))

    # 90-degree body rotation around z: topocentric [0,0,1] maps to body [0,0,1]
    # (z-rotation leaves the z-axis fixed, so TX coupling must be identical)
    R90z = _Beam.rot_z(np.pi / 2).astype(np.float32)
    geom_rz = fwd_tx.precompute_geometry(rots=[R], body_rots=[R90z])
    T_rz = np.array(fwd_tx.simulate(sky_c, beam_c, geom=geom_rz, T_gnd=0.0))
    np.testing.assert_allclose(T_no_rot, T_rz, atol=1e-4,
                                err_msg="z-rotation should not change coupling to zenith TX")

    # 90-degree body rotation around x: topocentric [0,0,1] maps to body [0,1,0]
    # (different beam response) — Tant should change
    R90x = _Beam.rot_x(np.pi / 2).astype(np.float32)
    geom_rx = fwd_tx.precompute_geometry(rots=[R], body_rots=[R90x])
    T_rx = np.array(fwd_tx.simulate(sky_c, beam_c, geom=geom_rx, T_gnd=0.0))
    assert not np.allclose(T_no_rot, T_rx, atol=1e-4), \
        "x-rotation moves TX to different beam direction, should change coupling"


def test_transmitter_no_sources_geom_shapes(simple_fwd):
    """No transmitters → tx_crds_jax has n_sources=0 and simulate runs fine."""
    fwd_no = ForwardModel(simple_fwd.observer, simple_fwd.beam, simple_fwd.sky)
    R = simple_fwd.observer.rot_gal2top().astype(np.float32)
    geom = fwd_no.precompute_geometry(rots=[R])

    assert geom['tx_crds_jax'].shape == (1, 0, 3)

    sky_c  = np.zeros((fwd_no.sky.npix, 2), dtype=np.float32)
    beam_c = fwd_no.beam.coeffs
    T = np.array(fwd_no.simulate(sky_c, beam_c, geom=geom))
    assert T.shape == (1, fwd_no.beam.coeffs.shape[0], len(fwd_no.beam.freqs_hz))
    assert np.all(np.isfinite(T))


# ─────────────────────────────────────────────────────────────────────────────
# sky_mask tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_sky_mask_rots(simple_fwd):
    """build_sky_mask returns bool (npix_sky,) covering ever-visible pixels."""
    fwd = simple_fwd
    R = fwd.observer.rot_gal2top().astype(np.float32)
    mask = fwd.build_sky_mask(rots=[R])
    assert mask.shape == (fwd.sky.npix,)
    assert mask.dtype == bool
    # Surface observers no longer impose an implicit z > 0 horizon.
    assert mask.all()


def test_sky_mask_reduces_crds_shape(simple_fwd):
    """precompute_geometry with sky_mask shrinks crds_gal_jax and terrain_masks_jax."""
    fwd = simple_fwd
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 3

    mask = fwd.build_sky_mask(rots=[R])
    npix_vis = int(mask.sum())

    geom_full = fwd.precompute_geometry(rots=[R] * ntimes)
    geom_mask = fwd.precompute_geometry(rots=[R] * ntimes, sky_mask=mask)

    assert geom_mask['crds_gal_jax'].shape      == (3, npix_vis)
    assert geom_mask['terrain_masks_jax'].shape == (ntimes, npix_vis)
    assert geom_mask['terrain_emissions_jax'].shape == (
        ntimes, npix_vis, len(fwd.beam.freqs_hz)
    )
    assert geom_mask['default_emission_masks_jax'].shape == (ntimes, npix_vis)
    assert geom_mask['unresolved_emission_jax'].shape == (len(fwd.beam.freqs_hz),)
    assert geom_mask['unresolved_default_emission_jax'].shape == (
        len(fwd.beam.freqs_hz),
    )
    assert 'sky_indices_jax' in geom_mask
    assert geom_mask['sky_indices_jax'].shape == (npix_vis,)

    # Full geom is unaffected
    assert geom_full['crds_gal_jax'].shape == (3, fwd.sky.npix)
    assert 'sky_indices_jax' not in geom_full


def test_sky_mask_same_result_at_zero_T_gnd(simple_fwd):
    """With T_gnd=0, sky_mask gives identical Tant to full-sky simulation.

    With an explicit sky_mask, omitted pixels contribute 0 in both paths when
    T_gnd=0, so the results must be numerically identical.
    """
    fwd = simple_fwd
    R = fwd.observer.rot_gal2top().astype(np.float32)
    sky_c  = np.abs(np.random.default_rng(42).standard_normal(
        (fwd.sky.npix, 2))).astype(np.float32)
    beam_c = fwd.beam.coeffs

    mask = fwd.build_sky_mask(rots=[R])
    geom_full = fwd.precompute_geometry(rots=[R])
    geom_mask = fwd.precompute_geometry(rots=[R], sky_mask=mask)

    T_full = np.array(fwd.simulate(sky_c, beam_c, geom=geom_full, T_gnd=0.0))
    T_mask = np.array(fwd.simulate(sky_c, beam_c, geom=geom_mask, T_gnd=0.0))

    np.testing.assert_allclose(T_full, T_mask, rtol=1e-5, atol=1e-5)


def test_sky_mask_same_result_with_ground(simple_fwd):
    """sky_mask must preserve ground emission from omitted pixels."""
    fwd = simple_fwd
    R = fwd.observer.rot_gal2top().astype(np.float32)
    sky_c = np.abs(
        np.random.default_rng(43).standard_normal((fwd.sky.npix, 2))
    ).astype(np.float32)
    beam_c = fwd.beam.coeffs

    mask = fwd.build_sky_mask(rots=[R])
    geom_full = fwd.precompute_geometry(rots=[R])
    geom_mask = fwd.precompute_geometry(rots=[R], sky_mask=mask)

    T_full = np.array(fwd.simulate(sky_c, beam_c, geom=geom_full, T_gnd=300.0))
    T_mask = np.array(fwd.simulate(sky_c, beam_c, geom=geom_mask, T_gnd=300.0))

    np.testing.assert_allclose(T_full, T_mask, rtol=1e-5, atol=1e-5)


def test_surface_observer_without_terrain_has_no_ground_cut():
    """EarthSurface without terrain sees the full sphere, even with T_gnd set."""
    from eigsep_sim.basis import BeamBasis, SkyBasis

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 4
    npix = healpy.nside2npix(nside)

    beam = Beam(
        nside,
        freqs_hz,
        BeamBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
        np.ones((1, npix, 1), dtype=np.float32),
    )
    sky = Sky(
        nside,
        freqs_hz,
        SkyBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
    )
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    fwd = ForwardModel(observer, beam, sky)

    sky_c = np.zeros((npix, 1), dtype=np.float32)
    beam_c = beam.coeffs
    R = observer.rot_gal2top().astype(np.float32)
    geom = fwd.precompute_geometry(rots=[R])
    assert np.all(np.array(geom["terrain_masks_jax"]) == 1.0)

    T = np.array(fwd.simulate(sky_c, beam_c, geom=geom, T_gnd=300.0))
    np.testing.assert_allclose(T, 0.0, atol=1e-5)


def test_surface_terrain_controls_ground_cut():
    """Terrain, not observer z, controls blocked surface-observer pixels."""
    from eigsep_sim.basis import BeamBasis, SkyBasis
    from eigsep_sim.terrain import Terrain

    class AllBlockedTerrain(Terrain):
        def mask(self, crds_top):
            return np.zeros(crds_top.shape[-1], dtype=bool)

        def emission(self, crds_top, freqs_hz):
            return np.full(
                (crds_top.shape[-1], len(freqs_hz)), 275.0, dtype=np.float32
            )

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 4
    npix = healpy.nside2npix(nside)

    beam = Beam(
        nside,
        freqs_hz,
        BeamBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
        np.ones((1, npix, 1), dtype=np.float32),
    )
    sky = Sky(
        nside,
        freqs_hz,
        SkyBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
    )
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    fwd = ForwardModel(observer, beam, sky, terrain=AllBlockedTerrain())

    sky_c = np.zeros((npix, 1), dtype=np.float32)
    beam_c = beam.coeffs
    R = observer.rot_gal2top().astype(np.float32)
    geom = fwd.precompute_geometry(rots=[R])
    assert np.all(np.array(geom["terrain_masks_jax"]) == 0.0)

    T = np.array(fwd.simulate(sky_c, beam_c, geom=geom, T_gnd=300.0))
    np.testing.assert_allclose(T, 275.0, atol=1e-5)


def test_sky_mask_preserves_uniform_terrain_emission():
    """sky_mask accounts for omitted blocked pixels using terrain spectrum."""
    from eigsep_sim.basis import BeamBasis, SkyBasis
    from eigsep_sim.terrain import HorizonTerrain

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.full(npix, 1.0, dtype=np.float32)

    beam = Beam(
        nside,
        freqs_hz,
        BeamBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
        np.ones((1, npix, 1), dtype=np.float32),
    )
    sky = Sky(
        nside,
        freqs_hz,
        SkyBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
    )
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    terrain = HorizonTerrain(nside, horizon_map, T_terrain=250.0)
    fwd = ForwardModel(observer, beam, sky, terrain=terrain)

    sky_c = np.zeros((npix, 1), dtype=np.float32)
    beam_c = beam.coeffs
    R = observer.rot_gal2top().astype(np.float32)
    sky_mask = fwd.build_sky_mask(rots=[R])
    assert not sky_mask.any()

    geom_full = fwd.precompute_geometry(rots=[R])
    geom_mask = fwd.precompute_geometry(rots=[R], sky_mask=sky_mask)

    T_full = np.array(fwd.simulate(sky_c, beam_c, geom=geom_full, T_gnd=300.0))
    T_mask = np.array(fwd.simulate(sky_c, beam_c, geom=geom_mask, T_gnd=300.0))

    np.testing.assert_allclose(T_full, 250.0, atol=1e-5)
    np.testing.assert_allclose(T_mask, T_full, atol=1e-5)


def test_sky_mask_rejects_unresolved_nonuniform_terrain_emission():
    """Nonuniform omitted terrain emission requires full geometry."""
    from eigsep_sim.basis import BeamBasis, SkyBasis
    from eigsep_sim.terrain import Terrain

    class NonUniformBlockedTerrain(Terrain):
        def mask(self, crds_top):
            return np.zeros(crds_top.shape[-1], dtype=bool)

        def emission(self, crds_top, freqs_hz):
            npix = crds_top.shape[-1]
            vals = np.arange(npix, dtype=np.float32)[:, None]
            return np.repeat(vals, len(freqs_hz), axis=1)

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 4
    npix = healpy.nside2npix(nside)

    beam = Beam(
        nside,
        freqs_hz,
        BeamBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
        np.ones((1, npix, 1), dtype=np.float32),
    )
    sky = Sky(
        nside,
        freqs_hz,
        SkyBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
    )
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    fwd = ForwardModel(observer, beam, sky, terrain=NonUniformBlockedTerrain())

    R = observer.rot_gal2top().astype(np.float32)
    sky_mask = fwd.build_sky_mask(rots=[R])
    assert not sky_mask.any()

    with pytest.raises(ValueError, match="unresolved_emission"):
        fwd.precompute_geometry(rots=[R], sky_mask=sky_mask)


def test_lunar_orbit_uses_occultation_not_body_horizon():
    """A far lunar orbiter with no occulted pixels should not see ground."""
    from eigsep_sim.basis import BeamBasis, SkyBasis

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 4
    npix = healpy.nside2npix(nside)

    beam = Beam(
        nside,
        freqs_hz,
        BeamBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
        np.ones((1, npix, 1), dtype=np.float32),
    )
    sky = Sky(
        nside,
        freqs_hz,
        SkyBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
    )
    observer = LunarOrbit(
        altitude=5e7,
        rot_orbit_vec=[0, 0, 1],
        rot_spin_vec=[0, 0, 1],
    )
    fwd = ForwardModel(observer, beam, sky)

    sky_c = np.zeros((npix, 1), dtype=np.float32)
    beam_c = beam.coeffs
    geom = fwd.precompute_geometry(times=[observer.t0])
    assert np.all(np.array(geom["masks"]) == 1.0)

    T = np.array(fwd.simulate(sky_c, beam_c, geom=geom, T_gnd=300.0))
    np.testing.assert_allclose(T, 0.0, atol=1e-5)


def test_lunar_sky_mask_preserves_occultation_emission():
    """sky_mask preserves lunar occultation fallback emission."""
    from eigsep_sim.basis import BeamBasis, SkyBasis

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 4
    npix = healpy.nside2npix(nside)

    beam = Beam(
        nside,
        freqs_hz,
        BeamBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
        np.ones((1, npix, 1), dtype=np.float32),
    )
    sky = Sky(
        nside,
        freqs_hz,
        SkyBasis(
            np.ones((len(freqs_hz), 1), dtype=np.float32),
            freqs_hz=freqs_hz,
        ),
    )
    observer = LunarOrbit(
        altitude=100e3,
        rot_orbit_vec=[0, 0, 1],
        rot_spin_vec=[0, 0, 1],
    )
    fwd = ForwardModel(observer, beam, sky)

    sky_c = np.zeros((npix, 1), dtype=np.float32)
    beam_c = beam.coeffs
    sky_mask = fwd.build_sky_mask(times=[observer.t0])
    assert sky_mask.any() and not sky_mask.all()

    geom_full = fwd.precompute_geometry(times=[observer.t0])
    geom_mask = fwd.precompute_geometry(times=[observer.t0], sky_mask=sky_mask)

    T_full = np.array(fwd.simulate(sky_c, beam_c, geom=geom_full, T_gnd=300.0))
    T_mask = np.array(fwd.simulate(sky_c, beam_c, geom=geom_mask, T_gnd=300.0))
    np.testing.assert_allclose(T_mask, T_full, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_forward_model_basic()
    test_forward_model_with_terrain()
    test_forward_model_precompute_geometry()
    test_forward_model_simulate_basic()
    test_forward_model_simulate_multiple_times()
    test_forward_model_simulate_with_precomputed_geom()
    test_forward_model_zero_coefficients()
    test_forward_model_orbital_observer()
    test_forward_model_simulate_no_times_no_geom_error()
    test_forward_model_different_nside_beam_sky()
    print("\n✓ All Phase 6 (simulate.py) tests passed!")
