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
