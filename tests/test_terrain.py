#!/usr/bin/env python
"""
Test suite for terrain.py — terrain visibility and thermal emission models.

Tests verify:
1. NullTerrain (no-op): all sky visible, zero emission
2. HorizonTerrain: HEALPix horizon map with visibility and thermal properties
3. LunarDisk: ray-sphere intersection for lunar occultation
"""

import numpy as np
import pytest
import healpy

from eigsep_sim.terrain import (
    HORIZON_MODELS_NPZ,
    NullTerrain,
    HorizonTerrain,
    LunarDisk,
)
from eigsep_sim.const import R_MOON, GM_MOON


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
    horizon_map = np.random.rand(npix).astype(np.float32) * 1e6  # Random distances

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
    horizon_map = np.array([np.nan] * (npix // 2) + [1e6] * (npix - npix // 2),
                          dtype=np.float32)

    terrain = HorizonTerrain(nside, horizon_map)
    crds_top = np.random.randn(3, npix)
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
    horizon_map[:npix_terrain // 2] = 1e6  # Block half

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
    horizon_map = np.array([np.nan] * (npix // 2) + [1e6] * (npix - npix // 2),
                          dtype=np.float32)
    T_terrain = 300.0

    terrain = HorizonTerrain(nside, horizon_map, T_terrain=T_terrain)
    freqs_hz = np.array([50e6, 100e6, 150e6])
    crds_top = np.random.randn(3, npix)

    emission = terrain.emission(crds_top, freqs_hz)

    assert emission.shape == (npix, 3)
    # Visible pixels should have zero emission
    assert np.allclose(emission[:npix//2], 0.0)
    # Blocked pixels should have T_terrain
    assert np.allclose(emission[npix//2:], T_terrain)


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


def test_lunar_disk_no_position():
    """LunarDisk: all sky visible when spacecraft position not set."""
    lunar = LunarDisk(nside=4)
    crds_top = np.random.randn(3, 100)

    mask = lunar.mask(crds_top)
    assert np.all(mask)  # All visible


def test_lunar_disk_update_position():
    """LunarDisk: update spacecraft position."""
    lunar = LunarDisk(nside=4, moon_radius_m=R_MOON, T_regolith=300.0)

    assert lunar.spacecraft_pos_gal is None

    pos = np.array([1e8, 0.0, 0.0])
    lunar.update(pos)

    assert np.allclose(lunar.spacecraft_pos_gal, pos)


def test_lunar_disk_far_away():
    """LunarDisk: sky visible when spacecraft is far from moon."""
    lunar = LunarDisk(nside=4, moon_radius_m=R_MOON)

    # Spacecraft far away
    pos = np.array([1e9, 0.0, 0.0])
    lunar.update(pos)

    crds_top = np.random.randn(3, 100)
    mask = lunar.mask(crds_top)

    # Most of the sky should be visible
    assert np.sum(mask) > 90


def test_lunar_disk_behind_moon():
    """LunarDisk: sky behind moon is blocked."""
    lunar = LunarDisk(nside=4, moon_radius_m=R_MOON)

    # Spacecraft at distance comparable to moon size
    pos = np.array([1e7, 0.0, 0.0])
    lunar.update(pos)

    # Create sky direction pointing toward moon (approximately)
    crds_top = np.zeros((3, 10))
    crds_top[0, :] = -1.0  # Point toward moon (along -x)
    crds_top = crds_top / np.linalg.norm(crds_top, axis=0)

    mask = lunar.mask(crds_top)

    # Some directions toward moon should be blocked
    assert np.sum(~mask) > 0


def test_lunar_disk_emission_unblocked():
    """LunarDisk: zero emission for visible pixels."""
    lunar = LunarDisk(nside=4, T_regolith=300.0)
    pos = np.array([1e9, 0.0, 0.0])
    lunar.update(pos)

    freqs_hz = np.array([50e6, 100e6])
    crds_top = np.random.randn(3, 100)

    emission = lunar.emission(crds_top, freqs_hz)

    # Should be mostly zero (very few rays blocked from far away)
    assert np.mean(emission) < 10.0


def test_lunar_disk_emission_blocked():
    """LunarDisk: lunar temperature for blocked pixels."""
    lunar = LunarDisk(nside=4, moon_radius_m=R_MOON, T_regolith=300.0)
    pos = np.array([1e7, 0.0, 0.0])
    lunar.update(pos)

    freqs_hz = np.array([50e6, 100e6])

    # Sky directions toward moon
    crds_top = np.zeros((3, 100))
    crds_top[0, :] = -1.0
    crds_top[1, :] = np.linspace(-0.5, 0.5, 100)
    crds_top[2, :] = np.linspace(-0.5, 0.5, 100)
    crds_top = crds_top / np.linalg.norm(crds_top, axis=0)

    emission = lunar.emission(crds_top, freqs_hz)

    # Some pixels should have lunar temperature
    assert np.any(emission > 100.0)


def test_lunar_disk_set_temperature():
    """LunarDisk: update regolith temperature."""
    lunar = LunarDisk(nside=4, T_regolith=300.0)
    assert lunar.T_regolith == 300.0

    lunar.set_temperature(250.0)
    assert lunar.T_regolith == 250.0


def test_lunar_disk_input_format():
    """LunarDisk: handle both (3, npix) and (npix, 3) coordinate formats."""
    lunar = LunarDisk(nside=4)
    pos = np.array([1e8, 0.0, 0.0])
    lunar.update(pos)

    # (3, npix) format
    crds_1 = np.random.randn(3, 50)
    mask_1 = lunar.mask(crds_1)
    assert mask_1.shape == (50,)

    # (npix, 3) format
    crds_2 = np.random.randn(50, 3)
    mask_2 = lunar.mask(crds_2)
    assert mask_2.shape == (50,)


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
    test_lunar_disk_no_position()
    test_lunar_disk_update_position()
    test_lunar_disk_far_away()
    test_lunar_disk_behind_moon()
    test_lunar_disk_emission_unblocked()
    test_lunar_disk_emission_blocked()
    test_lunar_disk_set_temperature()
    test_lunar_disk_input_format()
    test_horizon_terrain_input_format()
    print("\n✓ All Phase 3 (terrain.py) tests passed!")
