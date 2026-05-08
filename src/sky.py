"""
Sky brightness models with spectral basis decomposition.

Provides:
- Sky: descriptor for sky brightness (metadata only, no coefficients)
- SkyModel: legacy class (deprecated, for backward compatibility)
"""

import numpy as np
import healpy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from .basis import SkyBasis
from .const import c as C, k_B, Jy


class Sky:
    """
    Sky brightness model using spectral basis decomposition.

    A pure descriptor storing metadata: HEALPix resolution, frequencies, and
    spectral basis. Coefficients (spatial basis weights) live externally in
    the params dict, enabling a clean separation between model description
    and parameter state.

    Parameters
    ----------
    nside : int
        HEALPix resolution of sky coefficients.
    freqs_hz : ndarray, shape (nfreq,)
        Frequencies [Hz] at which the basis is defined.
    basis : SkyBasis
        Spectral basis for sky decomposition.
    """

    def __init__(self, nside, freqs_hz, basis):
        """
        Initialize a Sky descriptor.

        Parameters
        ----------
        nside : int
            HEALPix resolution.
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].
        basis : SkyBasis
            Spectral basis (contains projection matrix A).
        """
        self.nside = int(nside)
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        self.basis = basis

    @property
    def npix(self):
        """Number of HEALPix pixels."""
        return healpy.nside2npix(self.nside)

    @property
    def nmodes(self):
        """Number of basis modes."""
        return self.basis.nmodes

    @classmethod
    def from_gsm(cls, nside, freqs_hz, n_modes=5, include_flat=True):
        """
        Build sky from Global Sky Model (GSM16) via SkyBasis.

        Loads GSM16 via pygdsm, performs SVD to extract dominant spectral modes,
        and optionally appends a flat (constant) mode for common-mode degeneracy.

        Parameters
        ----------
        nside : int
            HEALPix resolution for sky coefficients.
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].
        n_modes : int, optional
            Number of GSM eigenmodes to retain (default 5).
        include_flat : bool, optional
            If True, append a normalized flat mode orthogonalized against
            GSM modes (default True).

        Returns
        -------
        Sky
            Sky descriptor with SkyBasis built from GSM.
        """
        # Build basis from GSM
        basis = SkyBasis.from_gsm(freqs_hz, n_modes=n_modes, nside=nside,
                                   include_flat=include_flat)
        return cls(nside, freqs_hz, basis)

    @classmethod
    def from_map(cls, nside, freqs_hz, sky_map, n_modes=5):
        """
        Build sky from a pre-computed (npix, nfreq) map via SVD.

        Performs SVD on the map to extract n_modes dominant spectral patterns.

        Parameters
        ----------
        nside : int
            HEALPix resolution.
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].
        sky_map : ndarray, shape (npix, nfreq)
            Sky brightness map.
        n_modes : int, optional
            Number of SVD modes to retain (default 5).

        Returns
        -------
        Sky
            Sky descriptor with basis built from the map.
        """
        basis = SkyBasis.from_ensemble(freqs_hz, sky_map, n_modes=n_modes)
        return cls(nside, freqs_hz, basis)

    def init_coeffs(self):
        """
        Generate initial sky coefficients from GSM.

        Returns
        -------
        coeffs : ndarray, shape (npix, nmodes)
            Initial basis coefficients, obtained by projecting GSM onto basis.
        """
        try:
            import pygdsm
        except ImportError:
            raise ImportError("pygdsm required for init_coeffs(); install via "
                            "`pip install pygdsm`")

        # Load GSM16 and resample to our nside
        freqs_mhz = self.freqs_hz / 1e6
        gsm = pygdsm.GlobalSkyModel16(freq_unit='MHz', resolution='lo')
        gsm_maps = []
        for f_mhz in freqs_mhz:
            m = gsm.query(f_mhz)
            m_hp = healpy.ud_grade(m, self.nside)
            gsm_maps.append(m_hp)
        gsm_maps = np.array(gsm_maps).T  # (npix, nfreq)

        # Project GSM onto basis
        coeffs = self.basis.project(gsm_maps)  # (npix, nmodes)
        return coeffs


# Legacy support: keep SkyModel for backward compatibility
try:
    from .healpix import HPM
    from aipy.src import get_catalog

    class SkyModel(HPM):
        """
        DEPRECATED: Use Sky + SkyBasis instead.

        Legacy sky brightness model combining GSM and bright point sources.
        """
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
            gsm_nside = healpy.npix2nside(gsm_data.shape[0])
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

        def rot_gal2top(self):
            """3x3 rotation matrix from galactic to topocentric via observer."""
            if self.observer is None:
                raise ValueError("No observer attached. Call set_observer() first.")
            return self.observer.rot_gal2top()

        def above_horizon(self, nside=None):
            """Boolean HEALPix mask for pixels above the horizon."""
            if self.observer is None:
                raise ValueError("No observer attached. Call set_observer() first.")
            if nside is None:
                nside = self._nside
            return self.observer.above_horizon(nside)

        def topocentric_map(self, nside=None):
            """Sky brightness in topocentric HEALPix frame."""
            if self.observer is None:
                raise ValueError("No observer attached. Call set_observer() first.")
            if nside is None:
                nside = self._nside

            R = self.rot_gal2top()
            npix = healpy.nside2npix(nside)
            x_top, y_top, z_top = [v.astype(np.float64)
                                    for v in healpy.pix2vec(nside, np.arange(npix))]

            RT = R.T
            x_gal = RT[0, 0]*x_top + RT[0, 1]*y_top + RT[0, 2]*z_top
            y_gal = RT[1, 0]*x_top + RT[1, 1]*y_top + RT[1, 2]*z_top
            z_gal = RT[2, 0]*x_top + RT[2, 1]*y_top + RT[2, 2]*z_top

            sky_topo = np.asarray(self[x_gal, y_gal, z_gal])

            mask = self.observer.above_horizon(nside)
            sky_topo[~mask] = 0.0
            return sky_topo

except ImportError:
    # If HPM or aipy not available, SkyModel cannot be defined
    SkyModel = None
