"""
Antenna beam model.

The Beam class wraps a HEALPix beam pattern (HPM) and adds azimuth/altitude
rotation so the beam can be steered in the topocentric frame without
reloading data.  load_beam interpolates a stored NPZ beam pattern onto the
requested frequency grid.
"""

import healpy
import numpy as np
from scipy.interpolate import interp1d

from .hpm import HPM
from .coord import rot_m

BEAM_NPZ = 'eigsep_vivaldi.npz'

_real_dtype = np.float32


def load_beam(freqs, filename=BEAM_NPZ):
    """
    Load and interpolate a HEALPix beam pattern from an NPZ file.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz] at which to evaluate the beam.
    filename : str
        Path to NPZ file containing 'freqs' (Hz) and 'bm' (nfreq, npix).

    Returns
    -------
    bm : ndarray, shape (npix, nfreq), float32
    """
    npz = np.load(filename)
    bm = npz['bm'].T  # (npix, nfreq)
    mdl_interp = interp1d(npz['freqs'], bm, kind='cubic',
                          fill_value=0, bounds_error=False)
    return mdl_interp(freqs).astype(_real_dtype)


class Beam(HPM):
    """
    Antenna beam pattern stored as a HEALPix map with rotation support.

    The beam is defined in a fixed antenna frame.  Az/alt rotation matrices
    are applied on every pixel access so the beam can be steered without
    reloading the data.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    filename : str
        Path to NPZ file containing the beam pattern.
    peak_normalize : bool
        If True, normalise the beam to its peak value across all pixels
        and frequencies.
    """

    def __init__(self, freqs, filename=BEAM_NPZ, peak_normalize=True):
        bm_data = load_beam(freqs, filename=filename)
        self.freqs = freqs
        if peak_normalize:
            bm_data /= bm_data.max()
        nside_beam = healpy.npix2nside(bm_data.shape[0])
        self.set_az(0)
        self.set_alt(0)
        HPM.__init__(self, nside_beam, interp=True)
        self.set_map(bm_data)

    def set_az(self, theta, az_vec=None):
        """Set the azimuth rotation angle (radians) about az_vec."""
        if az_vec is None:
            az_vec = np.array([0, 0, 1], dtype=_real_dtype)
        self.az = theta
        self.rot_az = rot_m(theta, az_vec)

    def set_alt(self, theta, alt_vec=None):
        """Set the altitude rotation angle (radians) about alt_vec."""
        if alt_vec is None:
            alt_vec = np.array([1, 0, 0], dtype=_real_dtype)
        self.alt = theta
        self.rot_alt = rot_m(theta, alt_vec)

    def get_rotation_matrices(self, azs, alts):
        """
        Compute combined az/alt rotation matrices for arrays of angles.

        Parameters
        ----------
        azs, alts : ndarray
            Azimuth and altitude angles (radians), same shape.

        Returns
        -------
        rot_ms : ndarray, shape (*azs.shape, 3, 3)
        """
        n_rots = azs.size
        rot_ms = np.empty((n_rots, 3, 3), dtype=_real_dtype)
        for cnt in range(n_rots):
            self.set_az(azs.flat[cnt])
            self.set_alt(alts.flat[cnt])
            rot_ms[cnt] = self.rot_az.dot(self.rot_alt)
        return rot_ms.reshape(azs.shape + (3, 3))

    def __getitem__(self, crd_top):
        """
        Interpolate the beam at topocentric directions *crd_top*, applying
        the current az/alt rotation before lookup.

        Parameters
        ----------
        crd_top : array_like, shape (3, N)
            Topocentric unit vectors.
        """
        rot = self.rot_az.dot(self.rot_alt)
        bx, by, bz = rot.dot(crd_top)
        return HPM.__getitem__(self, (bx, by, bz))
