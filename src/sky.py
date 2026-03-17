import numpy as np
import healpy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from aipy.src import get_catalog

from .healpix import HPM
from .const import c as C, k as k_B, Jy


class SkyModel(HPM):
    def __init__(self, freqs, observer=None, gsm=True, nside=64, dtype='float32',
                 srcs=['cyg', 'cas', 'vir', 'crab', 'her', 'cen', 'pic'],
                 ):
        HPM.__init__(self, nside=nside, interp=True)
        self.freqs = freqs.astype(dtype)
        self.dtype = dtype
        self.observer = observer
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
        if gsm_nside != self._nside:
            gsm_hpm = HPM(nside=gsm_nside)
            gsm_hpm.set_map(gsm_data)
            npix = healpy.nside2npix(self._nside)
            x, y, z = [v.astype(self.dtype)
                       for v in healpy.pix2vec(self._nside, np.arange(npix))]
            gsm_data = np.asarray(gsm_hpm[x, y, z])
        self.set_map(gsm_data)
        self._gsm_map = self.map

    def _update_map(self):
        self.map = self._gsm_map + self._src_map

    def _load_srcs(self, srcs):
        catalog = get_catalog(srcs)
        self._src_map = np.zeros(
            (self.npix(), self.freqs.size), dtype=self.dtype
        )
        for k in srcs:
            src = catalog[k]
            assert src._epoch == 36525.0  # corresponds to J2000
            src.update_jys(self.freqs / 1e9)
            crd = SkyCoord(ra=src._ra * u.rad, dec=src._dec * u.rad,
                           equinox='J2000', frame='icrs')
            gcrd = crd.galactic.cartesian
            px = self.crd2px(float(gcrd.x.value), float(gcrd.y.value),
                             float(gcrd.z.value))
            self._src_map[px] += (src.get_jys() * Jy /
                                  (2 * k_B * self.px_area() * self.freqs**2 / C**2))

    def px_area(self):
        return 4 * np.pi / self.npix()

    # ------------------------------------------------------------------
    # Time and observer management

    def set_time(self, t):
        """Set the current epoch; also updates the attached observer."""
        self.time = Time(t)
        if self.observer is not None:
            self.observer.set_time(t)

    def inc_time(self, seconds):
        """Advance the current epoch by *seconds*."""
        self.time = self.time + seconds * u.s
        if self.observer is not None:
            self.observer.set_time(self.time)

    def set_observer(self, observer):
        """Attach an observer (EarthSurface, LunarSurface, or LunarOrbit)."""
        self.observer = observer

    # ------------------------------------------------------------------
    # Topocentric projection

    def rot_gal2top(self):
        """
        3x3 rotation matrix mapping galactic unit vectors to topocentric
        unit vectors, via the attached observer.
        """
        if self.observer is None:
            raise ValueError("No observer attached. Call set_observer() first.")
        return self.observer.rot_gal2top()

    def above_horizon(self, nside=None):
        """
        Boolean HEALPix mask (galactic frame) for pixels above the horizon.
        """
        if self.observer is None:
            raise ValueError("No observer attached. Call set_observer() first.")
        if nside is None:
            nside = self._nside
        return self.observer.above_horizon(nside)

    def topocentric_map(self, nside=None):
        """
        Sky brightness in a topocentric HEALPix frame.

        For each topocentric pixel direction the galactic sky map is
        interpolated at the corresponding galactic direction.  Pixels
        below the horizon (or occluded by the Moon for a lunar-orbit
        observer) are set to zero.

        Parameters
        ----------
        nside : int, optional
            HEALPix nside for the output topocentric grid.  Defaults to
            the SkyModel's own nside.

        Returns
        -------
        sky_topo : ndarray, shape (npix_topo, nfreq) or (npix_topo,)
            Sky brightness in the topocentric frame.
        """
        if self.observer is None:
            raise ValueError("No observer attached. Call set_observer() first.")
        if nside is None:
            nside = self._nside

        R = self.rot_gal2top()  # (3, 3): gal -> top, so R.T: top -> gal
        npix = healpy.nside2npix(nside)
        x_top, y_top, z_top = [v.astype(np.float64)
                                for v in healpy.pix2vec(nside, np.arange(npix))]

        # Convert topocentric pixel directions to galactic
        RT = R.T  # top -> gal
        x_gal = RT[0, 0]*x_top + RT[0, 1]*y_top + RT[0, 2]*z_top
        y_gal = RT[1, 0]*x_top + RT[1, 1]*y_top + RT[1, 2]*z_top
        z_gal = RT[2, 0]*x_top + RT[2, 1]*y_top + RT[2, 2]*z_top

        # Interpolate galactic sky map at those directions
        sky_topo = np.asarray(self[x_gal, y_gal, z_gal])

        # Zero out below-horizon / occluded pixels
        mask = self.observer.above_horizon(nside)
        sky_topo[~mask] = 0.0
        return sky_topo
