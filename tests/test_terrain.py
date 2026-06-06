#!/usr/bin/env python
"""
Test suite for terrain.py — terrain visibility and thermal emission models.

Tests verify:
1. NullTerrain (no-op): all sky visible, zero emission
2. HorizonTerrain: HEALPix horizon map with visibility and thermal properties
"""

import numpy as np
import pytest
import healpy

from eigsep_sim.terrain import (
    HORIZON_MODELS_NPZ,
    NullTerrain,
    HorizonTerrain,
)


def test_null_terrain_mask():
    """NullTerrain: all pixels visible."""
    terrain = NullTerrain()

    # Test (3, npix) format
    crds_top = np.random.randn(3, 100)
    mask = terrain.mask(crds_top)
    assert mask.shape == (100,)
    assert np.all(mask)  # All True

    # Test (npix, 3) format
    crds_top = np.random.randn(100, 3)
    mask = terrain.mask(crds_top)
    assert mask.shape == (100,)
    assert np.all(mask)


def test_null_terrain_emission():
    """NullTerrain: zero thermal emission."""
    terrain = NullTerrain()
    freqs_hz = np.array([50e6, 100e6, 150e6])

    # Test (3, npix) format
    crds_top = np.random.randn(3, 100)
    emission = terrain.emission(crds_top, freqs_hz)
    assert emission.shape == (100, 3)
    assert np.allclose(emission, 0.0)

    # Test (npix, 3) format
    crds_top = np.random.randn(100, 3)
    emission = terrain.emission(crds_top, freqs_hz)
    assert emission.shape == (100, 3)
    assert np.allclose(emission, 0.0)


def test_horizon_terrain_basic():
    """HorizonTerrain: basic construction and properties."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = (
        np.random.rand(npix).astype(np.float32) * 1e6
    )  # Random distances

    terrain = HorizonTerrain(nside, horizon_map, T_terrain=300.0)

    assert terrain.nside == nside
    assert terrain.npix == npix
    assert terrain.T_terrain == 300.0
    assert np.allclose(terrain.horizon_map, horizon_map)


def test_horizon_terrain_mask_open_sky():
    """HorizonTerrain: NaN pixels are visible."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.full(npix, np.nan, dtype=np.float32)  # All open sky

    terrain = HorizonTerrain(nside, horizon_map)
    crds_top = np.random.randn(3, npix)
    mask = terrain.mask(crds_top)

    assert np.all(mask)  # All visible


def test_horizon_terrain_mask_blocked():
    """HorizonTerrain: non-NaN pixels are blocked."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.ones(npix, dtype=np.float32) * 1e6  # All blocked

    terrain = HorizonTerrain(nside, horizon_map)
    crds_top = np.random.randn(3, npix)
    mask = terrain.mask(crds_top)

    assert np.all(~mask)  # All blocked


def test_horizon_terrain_mask_mixed():
    """HorizonTerrain: mixed visible/blocked pixels."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.array(
        [np.nan] * (npix // 2) + [1e6] * (npix - npix // 2), dtype=np.float32
    )

    terrain = HorizonTerrain(nside, horizon_map)
    crds_top = np.array(healpy.pix2vec(nside, np.arange(npix)))
    mask = terrain.mask(crds_top)

    assert np.sum(mask) == npix // 2  # Half visible
    assert np.sum(~mask) == npix - npix // 2  # Half blocked


def test_horizon_terrain_mask_interpolation():
    """HorizonTerrain: interpolation when nside_sky differs from nside."""
    nside_terrain = 4
    nside_sky = 8
    npix_terrain = healpy.nside2npix(nside_terrain)
    npix_sky = healpy.nside2npix(nside_sky)

    # Create a horizon map with half blocked, half open
    horizon_map = np.full(npix_terrain, np.nan, dtype=np.float32)
    horizon_map[: npix_terrain // 2] = 1e6  # Block half

    terrain = HorizonTerrain(nside_terrain, horizon_map, nside_sky=nside_sky)

    # Sky coordinates at nside_sky resolution
    crds_top = np.array(healpy.pix2vec(nside_sky, np.arange(npix_sky)))
    mask = terrain.mask(crds_top)

    # Check that output has correct shape and is boolean
    assert mask.shape == (npix_sky,)
    assert mask.dtype == bool
    # After interpolation, should have some visible and some blocked
    assert 0 < np.sum(mask) < npix_sky


def test_horizon_terrain_emission():
    """HorizonTerrain: emission for blocked pixels only."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.array(
        [np.nan] * (npix // 2) + [1e6] * (npix - npix // 2), dtype=np.float32
    )
    T_terrain = 300.0

    terrain = HorizonTerrain(nside, horizon_map, T_terrain=T_terrain)
    freqs_hz = np.array([50e6, 100e6, 150e6])
    crds_top = np.array(healpy.pix2vec(nside, np.arange(npix)))

    emission = terrain.emission(crds_top, freqs_hz)

    assert emission.shape == (npix, 3)
    # Visible pixels should have zero emission
    assert np.allclose(emission[: npix // 2], 0.0)
    # Blocked pixels should have T_terrain
    assert np.allclose(emission[npix // 2 :], T_terrain)


def test_horizon_terrain_mask_uses_coordinates_same_nside():
    """HorizonTerrain: same-nside rotated coordinates are not direct indexed."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.full(npix, np.nan, dtype=np.float32)
    horizon_map[0] = 1e6
    terrain = HorizonTerrain(nside, horizon_map)

    native_crds = np.array(healpy.pix2vec(nside, np.arange(npix)))
    mask_native = terrain.mask(native_crds)
    assert not mask_native[0]
    assert np.all(mask_native[1:])

    repeated_blocked_crds = np.repeat(native_crds[:, :1], npix, axis=1)
    mask_repeated = terrain.mask(repeated_blocked_crds)
    assert not np.any(mask_repeated)


def test_horizon_terrain_set_temperature():
    """HorizonTerrain: update terrain temperature."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.ones(npix, dtype=np.float32) * 1e6

    terrain = HorizonTerrain(nside, horizon_map, T_terrain=300.0)
    assert terrain.T_terrain == 300.0

    terrain.set_temperature(250.0)
    assert terrain.T_terrain == 250.0

    # Check that emission uses new temperature
    freqs_hz = np.array([50e6, 100e6])
    crds_top = np.random.randn(3, npix)
    emission = terrain.emission(crds_top, freqs_hz)
    assert np.allclose(emission, 250.0)


def test_horizon_terrain_invalid_shape():
    """HorizonTerrain: raise error on shape mismatch."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.ones(npix + 10, dtype=np.float32)  # Wrong size

    with pytest.raises(ValueError, match="inconsistent with nside"):
        HorizonTerrain(nside, horizon_map)


def test_horizon_terrain_from_packaged_model():
    """HorizonTerrain: load packaged Marjum horizon model by index."""
    terrain = HorizonTerrain.from_packaged_model(index=0)

    assert terrain.nside == 64
    assert terrain.horizon_map.shape == (healpy.nside2npix(64),)
    assert terrain.height == 1.0
    assert terrain.center.shape == (3,)
    assert terrain.metadata["index"] == 0
    assert terrain.metadata["path"] == HORIZON_MODELS_NPZ
    assert np.isfinite(terrain.horizon_map).any()
    assert np.isnan(terrain.horizon_map).any()


def test_horizon_terrain_from_packaged_model_nearest_height():
    """HorizonTerrain: select nearest packaged Marjum height slice."""
    terrain = HorizonTerrain.from_packaged_model(height=64.0, T_terrain=275.0)

    assert np.isclose(terrain.height, 63.77777777777778)
    assert terrain.T_terrain == 275.0
    assert terrain.metadata["index"] == 5


def test_horizon_terrain_from_file_rejects_ambiguous_selection():
    """HorizonTerrain: reject index and height together."""
    with pytest.raises(ValueError, match="either index or height"):
        HorizonTerrain.from_file(HORIZON_MODELS_NPZ, index=0, height=1.0)


def test_horizon_terrain_input_format():
    """HorizonTerrain: handle both (3, npix) and (npix, 3) coordinate formats."""
    nside = 4
    npix = healpy.nside2npix(nside)
    horizon_map = np.full(npix, np.nan, dtype=np.float32)

    terrain = HorizonTerrain(nside, horizon_map)

    # (3, npix) format
    crds_1 = np.random.randn(3, 50)
    mask_1 = terrain.mask(crds_1)
    assert mask_1.shape == (50,)

    # (npix, 3) format
    crds_2 = np.random.randn(50, 3)
    mask_2 = terrain.mask(crds_2)
    assert mask_2.shape == (50,)


if __name__ == "__main__":
    test_null_terrain_mask()
    test_null_terrain_emission()
    test_horizon_terrain_basic()
    test_horizon_terrain_mask_open_sky()
    test_horizon_terrain_mask_blocked()
    test_horizon_terrain_mask_mixed()
    test_horizon_terrain_mask_interpolation()
    test_horizon_terrain_emission()
    test_horizon_terrain_set_temperature()
    test_horizon_terrain_invalid_shape()
    test_horizon_terrain_input_format()
    print("\n✓ All Phase 3 (terrain.py) tests passed!")
