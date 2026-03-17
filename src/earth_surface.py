"""
Ephemeris for a ground-based observer.

Provides the rotation from galactic to local topocentric coordinates
(x=east, y=north, z=up) at any time, given a geographic position.
"""

import numpy as np
import healpy
from astropy.coordinates import EarthLocation
import astropy.units as u

from ._observer import Observer, ICRS2GAL


class EarthSurface(Observer):
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
        super().__init__()
        self.location = EarthLocation(
            lat=lat * u.deg, lon=lon * u.deg, height=height * u.m
        )

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
        top2gal = ICRS2GAL @ top2icrs
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
