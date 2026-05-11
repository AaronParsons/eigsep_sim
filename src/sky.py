"""
Sky brightness models with spectral basis decomposition.

Provides:
- Sky: descriptor for sky brightness (metadata only, no coefficients)
"""

import numpy as np
import healpy

from .basis import SkyBasis
from .const import DTYPE_R_NPY


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
            m = gsm.generate(f_mhz)  # Returns map at nside=1024
            m_hp = healpy.ud_grade(m, self.nside)
            gsm_maps.append(m_hp)
        gsm_maps = np.array(gsm_maps).T  # (npix, nfreq)

        # Project GSM onto basis
        coeffs = self.basis.project(gsm_maps)  # (npix, nmodes)
        return coeffs

