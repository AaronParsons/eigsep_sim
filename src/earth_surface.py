"""
Ephemeris for a ground-based observer.

Provides the rotation from galactic to local topocentric coordinates
(x=east, y=north, z=up) at any time, given a geographic position.
"""

import numpy as np
import healpy
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, CartesianRepresentation
import astropy.units as u


def _icrs2gal_matrix():
    R = np.empty((3, 3))
    for i, v in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        c = SkyCoord(CartesianRepresentation(*v, unit=u.one), frame='icrs')
        R[:, i] = c.galactic.cartesian.xyz.value
    return R


_ICRS2GAL = _icrs2gal_matrix()


class EarthSurface:
    """
    Observer on the Earth's surface.

    Parameters
    ----------
    lat : float
        Geodetic latitude, degrees.
    lon : float
        Longitude, degrees east.
    height : float
        Height above the ellipsoid, metres.
    """

    def __init__(self, lat, lon, height=0.0):
        self.location = EarthLocation(
            lat=lat * u.deg, lon=lon * u.deg, height=height * u.m
        )
        self.time = None

    def set_time(self, t):
        """Set the current epoch."""
        self.time = Time(t)

    def rot_gal2top(self):
        """
        3x3 rotation matrix mapping galactic unit vectors to local
        topocentric unit vectors (x=east, y=north, z=up).
        """
        gcrs = self.location.get_gcrs(self.time)
        # GCRS cartesian components are aligned with ICRS axes
        up = gcrs.cartesian.xyz.value.copy()
        up /= np.linalg.norm(up)
        # GCRS z-axis is the celestial intermediate pole ≈ geographic north
        pole = np.array([0.0, 0.0, 1.0])
        east = np.cross(pole, up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)
        # columns of top2icrs are [east, north, up] expressed in ICRS
        top2icrs = np.column_stack([east, north, up])
        top2gal = _ICRS2GAL @ top2icrs
        return top2gal.T  # gal2top = top2gal^{-1} = top2gal.T

    def above_horizon(self, nside):
        """
        Boolean HEALPix mask (galactic frame) of pixels above the horizon.

        Returns
        -------
        mask : ndarray of bool, shape (npix,)
        """
        R = self.rot_gal2top()
        vecs = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
        return (R[2, :] @ vecs) > 0
