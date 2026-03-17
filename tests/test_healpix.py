"""
Tests for eigsep_sim.healpix — add2array, HealpixBase, HealpixMap, HPM.
"""

import numpy as np
import pytest
import healpy
from eigsep_sim.healpix import add2array, HealpixBase, HealpixMap, HPM


# ---------------------------------------------------------------------------
# add2array
# ---------------------------------------------------------------------------

class TestAdd2Array:
    def test_repeated_indices_accumulate(self):
        """add2array must accumulate repeated indices (unlike plain fancy assignment)."""
        a = np.zeros(3)
        ind = np.array([[0], [0], [1]])
        data = np.array([3.0, 4.0, 2.0])
        add2array(a, ind, data)
        assert a[0] == pytest.approx(7.0)
        assert a[1] == pytest.approx(2.0)
        assert a[2] == pytest.approx(0.0)

    def test_no_repeated_indices(self):
        a = np.zeros(5)
        ind = np.array([[1], [3]])
        data = np.array([1.5, 2.5])
        add2array(a, ind, data)
        assert a[1] == pytest.approx(1.5)
        assert a[3] == pytest.approx(2.5)
        assert a[0] == a[2] == a[4] == 0.0

    def test_all_to_same_index(self):
        a = np.zeros(2)
        ind = np.array([[0], [0], [0]])
        data = np.ones(3)
        add2array(a, ind, data)
        assert a[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# HealpixBase
# ---------------------------------------------------------------------------

class TestHealpixBase:
    def test_npix(self):
        for nside in (1, 4, 16):
            h = HealpixBase(nside=nside)
            assert h.npix() == healpy.nside2npix(nside)

    def test_order(self):
        h = HealpixBase(nside=8)
        assert h.order() == healpy.nside2order(8)

    def test_nside(self):
        h = HealpixBase(nside=32)
        assert h.nside() == 32

    def test_crd2px_vec(self):
        h = HealpixBase(nside=4)
        px = h.crd2px(1.0, 0.0, 0.0)
        assert isinstance(int(px), int)

    def test_px2crd_roundtrip(self):
        nside = 8
        h = HealpixBase(nside=nside)
        pix = np.arange(healpy.nside2npix(nside))
        xyz = h.px2crd(pix, ncrd=3)
        pix_back = h.crd2px(*xyz)
        np.testing.assert_array_equal(pix_back, pix)


# ---------------------------------------------------------------------------
# HPM
# ---------------------------------------------------------------------------

class TestHPM:
    def test_set_get_1d_map_constant(self):
        """Lookup of a constant map must return that constant at all directions."""
        nside = 4
        npix = healpy.nside2npix(nside)
        h = HPM(nside=nside, interp=False)
        data = np.full(npix, 7.5, dtype=np.float32)
        h.set_map(data)
        x, y, z = [v.astype(np.float32)
                   for v in healpy.pix2vec(nside, np.arange(npix))]
        vals = np.asarray(h[x, y, z])
        np.testing.assert_allclose(vals, 7.5, atol=1e-5)

    def test_set_get_2d_map_shape(self):
        """For a 2-D map (npix, nfreq), lookup at N directions returns (N, nfreq)."""
        nside = 4
        nfreq = 3
        n_dirs = 5
        npix = healpy.nside2npix(nside)
        h = HPM(nside=nside, interp=False)
        data = np.ones((npix, nfreq), dtype=np.float32)
        h.set_map(data)
        x, y, z = [v[:n_dirs].astype(np.float32)
                   for v in healpy.pix2vec(nside, np.arange(n_dirs))]
        vals = np.asarray(h[x, y, z])
        assert vals.ndim == 2
        assert vals.shape[-1] == nfreq

    def test_constant_map_interpolation(self):
        """Interpolation of a constant map should return that constant everywhere."""
        nside = 8
        npix = healpy.nside2npix(nside)
        h = HPM(nside=nside, interp=True)
        h.set_map(np.full(npix, 42.0, dtype=np.float32))
        x, y, z = [v.astype(np.float32)
                   for v in healpy.pix2vec(nside, np.arange(npix))]
        vals = np.asarray(h[x, y, z])
        np.testing.assert_allclose(vals, 42.0, rtol=1e-5)
