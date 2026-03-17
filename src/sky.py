import numpy as np
import healpy
from scipy.constants import c as C, k as k_B
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body, Galactic
from aipy.src import get_catalog

from .hpm import HPM
from .const import Jy_mks


class SkyModel(HPM):
    def __init__(self, freqs, gsm=True, nside=64, dtype='float32',
                 srcs=['cyg', 'cas', 'vir', 'crab', 'her', 'cen', 'pic'],
                 ):
        HPM.__init__(self, nside=nside, interp=True)
        self.freqs = freqs.astype(dtype)
        self.dtype = dtype
        self._gsm_map = 0
        self._src_map = 0
        if gsm:
            self._load_gsm()
        if srcs is not None:
            self._load_srcs(srcs)
        self._update_map()

    def _load_gsm(self):
        from pygdsm import GlobalSkyModel16 as GSM16
        gsm = GSM16(freq_unit='Hz',
                    data_unit='TRJ',
                    resolution='lo',
                    include_cmb=True)
        gsm_data = gsm.generate(self.freqs).astype(self.dtype).T
        gsm_nside = healpy.npix2nside(gsm_data.shape[1])
        if gsm_nside != self.nside:
            gsm_hpm = HPM(nside=gsm_nside)
            gsm_hpm.set_map(gsm_data)
            x, y, z = self.px2crd()
            gsm_data = gsm_hpm[x, y, z]
        self.set_map(gsm_data)
        self._gsm_map = self.map

    def _update_map(self):
        self.map = self._gsm_map + self._src_map

    def _load_srcs(self, srcs):
        catalog = get_catalog(srcs)
        self.map = np.array(self.npix(), dtype=self.dtype)
        for k in srcs:
            src = catalog[k]
            assert src._epoch == 36525.0  # corresponds to J2000
            src.update_jys(self.freqs / 1e9)
            crd = SkyCoord(ra=src._ra * u.rad, dec=src._dec * u.rad,
                           equinox='J2000', frame='icrs')
            gcrd = crd.galactic.cartesian
            self[gcrd.x, gcrd.y, gcrd.z] = (src.get_jys() * Jy_mks /
                    (2 * k_B * self.px_area() * self.freqs**2 / C**2))
        self._src_map = self.map

    def px_area(self):
        return 4 * np.pi / self.npix()

    def px2crd(self, px_inds=None):
        if px_inds is None:
            px_inds = np.arange(self.npix())
        v = np.array(self.px2crd(px_inds), dtype=self.dtype)
        return v

    def set_time(self, date):
        self.time = Time(date)

    def inc_time(self, seconds):
        self.time += seconds  # XXX check units

    def set_lunar_orbit(self, rel_moon_pos, t0, rot_orbit_vec):
        if t0 is None:
            t0 = self.time
        self.t0 = t0
        self.pos_at_t0 = v.astype(self.dtype)

    def moon_pos(self):
        return get_body('moon', self.time).transform_to(Galactic()).cartesian

    def earth_pos(self):
        return get_body('earth', self.time).transform_to(Galactic()).cartesian
