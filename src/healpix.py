"""
HEALPix map utilities.

Merges the HealpixBase / HealpixMap / Alm hierarchy (originally from aipy)
with JAX-accelerated interpolation (from healjax).  The HPM subclass is the
main entry point used by the rest of the package.
"""

from __future__ import annotations

import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
import healpy
from astropy.io import fits as pyfits

from healjax import get_interp_weights
import healjax
from healjax import INT_TYPE as int_dtype
from healjax import FLOAT_TYPE as float_dtype


# ---------------------------------------------------------------------------
# add2array — scatter-add that correctly handles repeated indices.
# Pure-Python / numpy replacement for the aipy C extension utils.add2array.
# ---------------------------------------------------------------------------

def add2array(a, ind, data):
    """
    Scatter-add *data* into *a* at the multi-dimensional indices in *ind*.

    Semantics match the original C extension: repeated indices accumulate
    (unlike plain NumPy fancy-index assignment which keeps only the last
    write for a repeated index).

    Parameters
    ----------
    a : ndarray
        Target array, modified in-place.
    ind : ndarray, shape (N, ndim(a))
        Multi-dimensional indices, one row per element of *data*.
    data : ndarray, shape (N,)
        Values to scatter-add.
    """
    if a.ndim == 1:
        np.add.at(a, ind[:, 0], data)
    else:
        idx = tuple(ind[:, j] for j in range(ind.shape[1]))
        np.add.at(a, idx, data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mk_arr(val, dtype=np.double):
    if type(val) is np.ndarray:
        return val.astype(dtype)
    return np.array(val, dtype=dtype).flatten()


HEALPIX_MODES = ('RING', 'NEST')

default_fits_format_codes = {
    np.bool_: 'L', np.uint8: 'B', np.int16: 'I', np.int32: 'J',
    np.int64: 'K', np.float32: 'E', np.float64: 'D',
    np.complex64: 'C', np.complex128: 'M',
}


# ---------------------------------------------------------------------------
# HealpixBase
# ---------------------------------------------------------------------------

class HealpixBase:
    """Functionality related to the HEALPix pixelisation."""

    def __init__(self, nside=1, scheme='RING'):
        self._nside = nside
        self._scheme = scheme

    def npix2nside(self, npix):
        return healpy.npix2nside(npix)

    def nest_ring_conv(self, px, scheme):
        """Translate pixel numbers to the given scheme ('NEST' or 'RING')."""
        mode = {'RING': healpy.nest2ring, 'NEST': healpy.ring2nest}
        if scheme != self._scheme:
            px = mode[scheme](self._nside, px)
        self._scheme = scheme
        return px

    def set_nside_scheme(self, nside=None, scheme=None):
        if nside is not None:
            pow2 = np.log2(nside)
            assert pow2 == np.around(pow2)
            self._nside = nside
        if scheme is not None:
            assert scheme in HEALPIX_MODES
            self._scheme = scheme

    def crd2px(self, c1, c2, c3=None, interpolate=False):
        """Convert coordinates to pixel indices.

        If only c1, c2 provided, read as theta, phi.
        If c1, c2, c3 provided, read as x, y, z.
        If *interpolate* is True, return (px, wgts) with 4 neighbours each.
        """
        is_nest = (self._scheme == 'NEST')
        if not interpolate:
            if c3 is None:
                return healpy.ang2pix(self._nside, c1, c2, nest=is_nest)
            else:
                return healpy.vec2pix(self._nside, c1, c2, c3, nest=is_nest)
        else:
            if c3 is not None:
                c1, c2 = healpy.vec2ang(np.array([c1, c2, c3]).T)
            px, wgts = healpy.get_interp_weights(self._nside, c1, c2, nest=is_nest)
            return px.T, wgts.T

    def px2crd(self, px, ncrd=3):
        """Convert pixel numbers to coordinates (ncrd=2 → theta/phi, ncrd=3 → x/y/z)."""
        is_nest = (self._scheme == 'NEST')
        assert ncrd in (2, 3)
        if ncrd == 2:
            return healpy.pix2ang(self._nside, px, nest=is_nest)
        else:
            return healpy.pix2vec(self._nside, px, nest=is_nest)

    def order(self):
        return healpy.nside2order(self._nside)

    def nside(self):
        return self._nside

    def npix(self):
        return healpy.nside2npix(self._nside)

    def scheme(self):
        return self._scheme


# ---------------------------------------------------------------------------
# Alm
# ---------------------------------------------------------------------------

class Alm:
    """Spherical-harmonic coefficients up to a given order."""

    def __init__(self, lmax, mmax, dtype=np.complex128):
        assert lmax >= mmax
        self._alm = healpy.Alm()
        self._lmax = lmax
        self._mmax = mmax
        self.dtype = dtype
        self.set_to_zero()

    def size(self):
        return self._alm.getsize(self._lmax, self._mmax)

    def set_to_zero(self):
        self.set_data(np.zeros(self.size(), dtype=self.dtype))

    def lmax(self):
        return self._lmax

    def mmax(self):
        return self._mmax

    def __getitem__(self, lm):
        l, m = lm
        return self.data[self._alm.getidx(self._lmax, l, m)]

    def __setitem__(self, lm, val):
        l, m = lm
        self.data[self._alm.getidx(self._lmax, l, m)] = val

    def to_map(self, nside, pixwin=False, fwhm=0.0, sigma=None, pol=True):
        return healpy.alm2map(self.get_data(), nside,
                              lmax=self._lmax, mmax=self._mmax,
                              pixwin=pixwin, fwhm=fwhm, sigma=sigma, pol=pol)

    def from_map(self, data, iter=3, pol=True, use_weights=False, gal_cut=0):
        data = healpy.map2alm(data, lmax=self._lmax, mmax=self._mmax,
                              iter=iter, pol=pol, use_weights=use_weights,
                              gal_cut=gal_cut)
        self.set_data(data)

    def lm_indices(self):
        return np.array([self._alm.getlm(self._lmax, i) for i in range(self.size())])

    def get_data(self):
        return self.data

    def set_data(self, data):
        assert data.size == self.size()
        self.data = data.astype(self.dtype)


# ---------------------------------------------------------------------------
# HealpixMap
# ---------------------------------------------------------------------------

class HealpixMap(HealpixBase):
    """Data array on a HEALPix sphere."""

    def __init__(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.double)
        interp = kwargs.pop('interp', False)
        fromfits = kwargs.pop('fromfits', None)
        HealpixBase.__init__(self, *args, **kwargs)
        self._use_interpol = interp
        if fromfits is None:
            self.set_map(np.zeros((self.npix(),), dtype=dtype),
                         scheme=self.scheme())
        else:
            self.from_fits(fromfits)

    def set_interpol(self, onoff):
        self._use_interpol = onoff

    def set_map(self, data, scheme='RING'):
        """Assign data; infers Nside from the first axis length."""
        try:
            nside = self.npix2nside(data.shape[0])
        except (AssertionError, ValueError):
            raise ValueError("First axis of data must have 12*N**2 elements.")
        self.set_nside_scheme(nside, scheme)
        self.map = data

    def get_map(self):
        return self.map

    def get_dtype(self):
        return self.map.dtype

    def change_scheme(self, scheme):
        assert scheme in ('RING', 'NEST')
        if scheme == self.scheme():
            return
        i = self.nest_ring_conv(np.arange(self.npix()), scheme)
        self[i] = self.map
        self.set_nside_scheme(self.nside(), scheme)

    def __getitem__(self, crd):
        """Access data via hpm[crd] where crd is pixel indices, (th,phi), or (x,y,z)."""
        if type(crd) is tuple:
            crd = [mk_arr(c, dtype=np.double) for c in crd]
            if self._use_interpol:
                px, wgts = self.crd2px(*crd, interpolate=True)
                wgts.shape += (1,) * (self.map.ndim - 1)
                return np.sum(self.map[px] * wgts, axis=1)
            else:
                px = self.crd2px(*crd)
        else:
            px = mk_arr(crd, dtype=np.int64)
        return self.map[px]

    def __setitem__(self, crd, val):
        """Assign data; repeated coordinates accumulate (scatter-add semantics)."""
        if type(crd) is tuple:
            crd = [mk_arr(c, dtype=np.double) for c in crd]
            px = self.crd2px(*crd)
        else:
            if type(crd) is np.ndarray:
                assert len(crd.shape) == 1
            px = mk_arr(crd, dtype=int)
        if px.size == 1:
            if type(val) is np.ndarray:
                val = mk_arr(val, dtype=self.map.dtype)
            self.map[px] = val
        else:
            m = np.zeros_like(self.map)
            px = px.reshape(px.size, 1)
            cnt = np.zeros(self.map.shape, dtype=np.bool_)
            val = mk_arr(val, dtype=m.dtype)
            add2array(m, px, val)
            add2array(cnt, px, np.ones(val.shape, dtype=np.bool_))
            self.map = np.where(cnt, m, self.map)

    def from_hpm(self, hpm):
        """Initialise from another HealpixMap, upgrading/downgrading resolution."""
        if hpm.nside() < self.nside():
            interpol = hpm._use_interpol
            hpm.set_interpol(True)
            px = np.arange(self.npix())
            th, phi = self.px2crd(px, ncrd=2)
            self[px] = hpm[th, phi].astype(self.get_dtype())
            hpm.set_interpol(interpol)
        elif hpm.nside() > self.nside():
            px = np.arange(hpm.npix())
            th, phi = hpm.px2crd(px, ncrd=2)
            self[th, phi] = hpm[px].astype(self.get_dtype())
        else:
            if hpm.scheme() == self.scheme():
                self.map = hpm.map.astype(self.get_dtype())
            else:
                i = self.nest_ring_conv(np.arange(self.npix()), hpm.scheme())
                self.map = hpm.map[i].astype(self.get_dtype())

    def from_alm(self, alm):
        self.set_map(alm.to_map(self.nside(), self.scheme()))

    def to_alm(self, lmax, mmax, iter=1):
        assert self.scheme() == 'RING'
        alm = Alm(lmax, mmax)
        alm.from_map(self.map, iter)
        return alm

    def from_fits(self, filename, hdunum=1, colnum=0):
        hdu = pyfits.open(filename)[hdunum]
        data = hdu.data.field(colnum)
        if not data.dtype.isnative:
            data.dtype = data.dtype.newbyteorder()
            data.byteswap(True)
        scheme = hdu.header['ORDERING'][:4]
        self.set_map(data, scheme=scheme)

    def to_fits(self, filename, format=None, clobber=True):
        if format is None:
            format = default_fits_format_codes[self.get_dtype().type]
        hdu0 = pyfits.PrimaryHDU()
        col0 = pyfits.Column(name='signal', format=format, array=self.map)
        cols = pyfits.ColDefs([col0])
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
        self._set_fits_header(tbhdu.header)
        pyfits.HDUList([hdu0, tbhdu]).writeto(filename, overwrite=clobber)

    def _set_fits_header(self, hdr):
        hdr['PIXTYPE'] = ('HEALPIX', 'HEALPIX pixelisation')
        scheme = 'NESTED' if self.scheme() == 'NEST' else self.scheme()
        hdr['ORDERING'] = (scheme, 'Pixel ordering scheme')
        hdr['NSIDE'] = (self.nside(), 'Resolution parameter for HEALPIX')
        hdr['FIRSTPIX'] = (0, 'First pixel # (0 based)')
        hdr['LASTPIX'] = (self.npix() - 1, 'Last pixel # (0 based)')
        hdr['INDXSCHM'] = ('IMPLICIT', 'Indexing: IMPLICIT or EXPLICIT')


# ---------------------------------------------------------------------------
# JAX-accelerated helpers
# ---------------------------------------------------------------------------

def vec2ang(c1, c2, c3):
    return healjax.vec2ang(c1, c2, c3)


def ang2pix(scheme, nside, c1, c2):
    px_flat = jax.vmap(
        lambda th, ph: healjax.ang2pix(scheme, nside, th, ph)
    )(c1.ravel(), c2.ravel())
    return px_flat.reshape(c1.shape)


def vec2pix(scheme, nside, c1, c2, c3):
    px_flat = jax.vmap(
        lambda x, y, z: healjax.vec2pix(scheme, nside, x, y, z)
    )(c1.ravel(), c2.ravel(), c3.ravel())
    return px_flat.reshape(c1.shape)


@partial(jax.jit, static_argnums=(0,))
def interpolate_map(nside, map_data, c1, c2, c3=None):
    """JAX-accelerated HEALPix interpolation."""
    if c3 is not None:
        c1, c2 = healjax.vec2ang(c1, c2, c3)
    px, wgts = get_interp_weights(c1, c2, nside)
    slicing = (slice(None),) * wgts.ndim + (None,) * (map_data.ndim - 1)
    return jnp.sum(map_data[px] * wgts[slicing], axis=0)


@partial(jax.jit, static_argnums=(0,))
def rotate_interpolate_and_sum(nside, map_data, sky, crds, rot_ms):
    """JAX-accelerated rotation / interpolation / beam-weighted sum."""
    def body(_, rot_m):
        tx, ty, tz = rot_m @ crds
        wgt = interpolate_map(nside, map_data, tx, ty, tz)
        val = jnp.sum(wgt * sky, axis=0) / jnp.sum(wgt, axis=0)
        return None, val
    _, data_out = jax.lax.scan(body, None, rot_ms)
    return data_out  # (nrots, nfreq)


# ---------------------------------------------------------------------------
# HPM — HealpixMap with JAX-accelerated interpolation
# ---------------------------------------------------------------------------

class HPM(HealpixMap):

    def __init__(self, *args, **kwargs):
        HealpixMap.__init__(self, *args, **kwargs)
        scheme = self._scheme.lower()
        self.jax_ang2pix = jax.jit(partial(ang2pix, scheme, self._nside))
        self.jax_vec2pix = jax.jit(partial(vec2pix, scheme, self._nside))
        self.jax_vec2ang = jax.jit(vec2ang)

    def crd2px(self, c1, c2, c3=None, interpolate=False):
        """Coordinate → pixel conversion using JAX (non-interpolating path)
        or healpy (interpolating path)."""
        is_nest = (self._scheme == 'NEST')
        if not interpolate:
            if c3 is None:
                return self.jax_ang2pix(c1, c2)
            else:
                return self.jax_vec2pix(c1, c2, c3)
        else:
            if c3 is not None:
                c1, c2 = self.jax_vec2ang(c1, c2, c3)
            assert not is_nest  # NEST not supported via JAX path
            px, wgts = get_interp_weights(c1, c2, self._nside)
            return px.T, wgts.T

    def __getitem__(self, crd):
        """Access data via hpm[crd] (pixel indices, (th,phi), or (x,y,z))."""
        if type(crd) is tuple:
            crd = [mk_arr(c, dtype=np.double) for c in crd]
            if self._use_interpol:
                return interpolate_map(self._nside, self.map, *crd)
            else:
                px = self.crd2px(*crd)
        else:
            px = mk_arr(crd, dtype=np.int64)
        return self.map[px]

    def set_map(self, data, scheme='RING'):
        """Assign data; infers Nside from the first axis length."""
        try:
            nside = self.npix2nside(data.shape[0])
        except (AssertionError, ValueError):
            raise ValueError("First axis of data must have 12*N**2 elements.")
        self.set_nside_scheme(nside, scheme)
        self.map = data

    def rotate_interpolate_and_sum(self, sky, crds, rot_ms, chunk_size=16):
        """Beam-weighted sky integral for each rotation matrix in *rot_ms*."""
        data_out = []
        for i in range(0, rot_ms.shape[0], chunk_size):
            data_out.append(
                rotate_interpolate_and_sum(
                    self._nside, self.map, sky, crds, rot_ms[i:i + chunk_size]
                )
            )
        return np.concatenate(data_out, axis=0)
