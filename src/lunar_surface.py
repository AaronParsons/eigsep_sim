"""
Ephemeris for an observer on the lunar surface.

The Moon's orientation in ICRS is modelled with the IAU WGCCRE 2015 mean
rotation parameters.  The local topocentric frame has x=east, y=north,
z=up (away from the Moon centre).
"""

import numpy as np
import healpy

from ._observer import Observer, ICRS2GAL


def _rotmat_x(a):
    """Right-handed rotation by angle a (radians) about the x-axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotmat_z(a):
    """Right-handed rotation by angle a (radians) about the z-axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _moon_icrs2mcmf(t):
    """
    3x3 rotation matrix mapping ICRS to Moon body-fixed (MCMF) frame.

    Uses the IAU WGCCRE 2015 mean orientation model.  Returns R such
    that v_mcmf = R @ v_icrs.
    """
    d = t.tdb.jd - 2451545.0      # TDB days from J2000.0
    T = d / 36525.0               # Julian centuries TDB

    # Fundamental arguments (IAU WGCCRE 2015, Table 2)
    E1  = np.deg2rad(125.045 -  0.0529921 * d)
    E2  = np.deg2rad(250.089 -  0.1059842 * d)
    E3  = np.deg2rad(260.008 + 13.012009  * d)
    E4  = np.deg2rad(176.625 + 13.340716  * d)
    E5  = np.deg2rad(357.529 +  0.9856003 * d)
    E6  = np.deg2rad(311.589 + 26.4057084 * d)
    E7  = np.deg2rad(134.963 + 13.0649930 * d)
    E8  = np.deg2rad(276.617 +  0.3287146 * d)
    E9  = np.deg2rad( 34.226 +  1.7484877 * d)
    E10 = np.deg2rad( 15.134 -  0.1589763 * d)
    E11 = np.deg2rad(119.743 +  0.0036096 * d)
    E12 = np.deg2rad(239.961 +  0.1643573 * d)
    E13 = np.deg2rad( 25.053 + 12.9590088 * d)

    alpha0 = np.deg2rad(
        269.9949 + 0.0031 * T
        - 3.8787 * np.sin(E1)  - 0.1204 * np.sin(E2)
        + 0.0700 * np.sin(E3)  - 0.0172 * np.sin(E4)
        + 0.0072 * np.sin(E6)  - 0.0052 * np.sin(E10)
        + 0.0043 * np.sin(E13)
    )
    delta0 = np.deg2rad(
        66.5392 + 0.013 * T
        + 1.5419 * np.cos(E1) + 0.0239 * np.cos(E2)
        - 0.0278 * np.cos(E3) + 0.0068 * np.cos(E4)
        - 0.0029 * np.cos(E6) + 0.0009 * np.cos(E7)
        + 0.0008 * np.cos(E10) - 0.0009 * np.cos(E13)
    )
    W = np.deg2rad(
        38.3213 + 13.17635815 * d - 1.4e-12 * d ** 2
        - 3.5610 * np.sin(E1)  - 0.1208 * np.sin(E2)
        - 0.0642 * np.sin(E3)  + 0.0158 * np.sin(E4)
        + 0.0252 * np.sin(E5)  - 0.0066 * np.sin(E6)
        - 0.0047 * np.sin(E7)  - 0.0046 * np.sin(E8)
        + 0.0028 * np.sin(E9)  + 0.0052 * np.sin(E10)
        + 0.0040 * np.sin(E11) + 0.0019 * np.sin(E12)
        - 0.0044 * np.sin(E13)
    )

    # ICRS → MCMF:  Rz(W) @ Rx(90° − δ₀) @ Rz(90° + α₀)
    return _rotmat_z(W) @ _rotmat_x(np.pi / 2 - delta0) @ _rotmat_z(np.pi / 2 + alpha0)


class LunarSurface(Observer):
    """
    Observer on the lunar surface.

    Parameters
    ----------
    lat : float
        Selenographic latitude, degrees.
    lon : float
        Selenographic longitude, degrees east.
    """

    def __init__(self, lat, lon):
        super().__init__()
        self.lat_rad = np.deg2rad(lat)
        self.lon_rad = np.deg2rad(lon)
        # Observer "up" direction in MCMF (body-fixed Moon frame)
        clat = np.cos(self.lat_rad)
        slat = np.sin(self.lat_rad)
        self._up_mcmf = np.array(
            [clat * np.cos(self.lon_rad),
             clat * np.sin(self.lon_rad),
             slat]
        )

    def rot_gal2top(self):
        """
        3x3 rotation matrix mapping galactic unit vectors to local
        topocentric unit vectors (x=east, y=north, z=up).
        """
        icrs2mcmf = _moon_icrs2mcmf(self.time)
        mcmf2icrs = icrs2mcmf.T
        up_icrs = mcmf2icrs @ self._up_mcmf
        # Moon's north pole direction in ICRS
        pole_icrs = mcmf2icrs @ np.array([0.0, 0.0, 1.0])
        east_icrs = np.cross(pole_icrs, up_icrs)
        east_icrs /= np.linalg.norm(east_icrs)
        north_icrs = np.cross(up_icrs, east_icrs)
        top2icrs = np.column_stack([east_icrs, north_icrs, up_icrs])
        top2gal = ICRS2GAL @ top2icrs
        return top2gal.T

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
