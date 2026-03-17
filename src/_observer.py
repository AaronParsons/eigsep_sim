"""
Abstract base class for observer ephemeris modules.

All concrete observers (EarthSurface, LunarSurface, LunarOrbit) inherit
from Observer and must implement rot_gal2top() and above_horizon().
"""

import numpy as np
import healpy
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u


def _icrs2gal_matrix():
    """Constant 3x3 rotation matrix from ICRS to galactic frame.

    Column i is the galactic representation of the i-th ICRS basis vector,
    so v_gal = _ICRS2GAL @ v_icrs.
    """
    R = np.empty((3, 3))
    for i, v in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        c = SkyCoord(CartesianRepresentation(*v, unit=u.one), frame='icrs')
        R[:, i] = c.galactic.cartesian.xyz.value
    return R


ICRS2GAL = _icrs2gal_matrix()


class Observer:
    """
    Abstract base class for observer ephemeris.

    Subclasses must implement:
      - rot_gal2top()   : 3x3 rotation matrix, galactic → topocentric
      - above_horizon() : boolean HEALPix mask of visible pixels

    The base class provides:
      - self.time       : current epoch (astropy Time or None)
      - set_time(t)     : stores Time(t) in self.time
    """

    def __init__(self):
        self.time = None

    def set_time(self, t):
        """Set the current epoch."""
        self.time = Time(t)

    def rot_gal2top(self):
        """
        3x3 rotation matrix mapping galactic unit vectors to topocentric
        unit vectors (x=east, y=north, z=up for surface observers).

        Returns
        -------
        R : ndarray, shape (3, 3)
        """
        raise NotImplementedError

    def above_horizon(self, nside):
        """
        Boolean HEALPix mask (galactic frame) of pixels that are visible
        to this observer (above the horizon, or not occluded).

        Parameters
        ----------
        nside : int

        Returns
        -------
        mask : ndarray of bool, shape (npix,)
        """
        raise NotImplementedError
