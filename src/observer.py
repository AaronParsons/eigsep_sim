"""
Observer classes for different vantage points: Earth surface, lunar surface, lunar orbit.

All observers inherit from the Observer ABC and must implement:
- rot_gal2top(): rotation matrix from galactic to topocentric frame
- above_horizon(): boolean HEALPix mask of visible sky pixels
"""

from __future__ import annotations

import numpy as np
import healpy
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation, EarthLocation
import astropy.units as u
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp

from .coord import rot_m
from .const import R_MOON, GM_MOON


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


def circular_orbital_period(altitude):
    """
    Orbital period [s] for a circular lunar orbit at *altitude* metres above
    the surface.

    Uses Kepler's third law with the Moon's standard gravitational parameter
    GM_MOON = 4.9048695 × 10¹² m³ s⁻².

    Parameters
    ----------
    altitude : float
        Altitude above the lunar surface [m].

    Returns
    -------
    period : float
        Orbital period [s].
    """
    r = R_MOON + altitude
    return 2.0 * np.pi * np.sqrt(r ** 3 / GM_MOON)


class LunarOrbit(Observer):
    """
    Spacecraft in a circular lunar orbit with an independent spin.

    The orbital period is computed automatically from *altitude* via Kepler's
    third law (:func:`circular_orbital_period`).  The galactic frame is used
    as the inertial reference throughout, consistent with the GSM sky model.

    Parameters
    ----------
    altitude : float
        Orbital altitude above the lunar surface [m].
    rot_orbit_vec : array_like, shape (3,)
        Unit vector in the galactic frame normal to the orbital plane.
        The spacecraft orbits counter-clockwise around this axis.
    rot_spin_vec : array_like, shape (3,)
        Unit vector in the galactic frame about which the spacecraft spins.
    start_pos : array_like, shape (3,), optional
        Unit vector in the galactic frame giving the initial orbital
        position direction from the Moon centre.  Default: [1, 0, 0].
    spin_period : float
        Spacecraft spin period [s].  0 means no spin.
    t0 : `~astropy.time.Time` or str, optional
        Reference epoch: spacecraft is at ``start_pos`` with zero spin
        phase at this time.  Default: J2000.
    """

    def __init__(
        self,
        altitude,
        rot_orbit_vec,
        rot_spin_vec,
        start_pos=None,
        spin_period=0.0,
        t0=None,
    ):
        self.altitude = altitude
        self.orbital_radius = R_MOON + altitude
        self.orbital_period = circular_orbital_period(altitude)

        self.rot_orbit_vec = np.asarray(rot_orbit_vec, dtype=float)
        self.rot_orbit_vec /= np.linalg.norm(self.rot_orbit_vec)

        self.rot_spin_vec = np.asarray(rot_spin_vec, dtype=float)
        self.rot_spin_vec /= np.linalg.norm(self.rot_spin_vec)

        if start_pos is None:
            start_pos = np.array([1.0, 0.0, 0.0])
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_pos /= np.linalg.norm(self.start_pos)

        self.spin_period = float(spin_period)
        self.t0 = Time("J2000") if t0 is None else Time(t0)
        super().__init__()
        self.time = self.t0

        self._th_orbit = 0.0
        self._th_spin = 0.0

    def set_time(self, t):
        """Set the current epoch and update orbital and spin phases."""
        super().set_time(t)
        dt = (self.time - self.t0).to(u.s).value
        self._th_orbit = 2 * np.pi * dt / self.orbital_period
        self._th_spin = (
            2 * np.pi * dt / self.spin_period if self.spin_period != 0.0 else 0.0
        )

    def set_phases(self, th_orbit, th_spin=0.0):
        """
        Directly set orbital and spin phases in radians.

        Useful for looping over orbital/spin configurations without
        converting to absolute time.
        """
        self._th_orbit = float(th_orbit)
        self._th_spin = float(th_spin)

    def spacecraft_position(self):
        """
        Position of the spacecraft relative to the Moon centre,
        expressed in the galactic frame.

        Returns
        -------
        pos : ndarray, shape (3,)
            Position vector in metres.
        """
        R = rot_m(self._th_orbit, self.rot_orbit_vec)
        return R @ (self.start_pos * self.orbital_radius)

    def rot_gal2top(self):
        """
        3x3 rotation matrix mapping galactic unit vectors to spacecraft-
        body unit vectors.

        At zero spin phase the spacecraft frame is aligned with the
        galactic frame.  The spin rotation is applied counter-clockwise
        around ``rot_spin_vec``.
        """
        # R_spin maps topocentric → galactic, so gal2top = R_spin.T
        R_spin = rot_m(self._th_spin, self.rot_spin_vec)
        return R_spin.T

    def above_horizon(self, nside):
        """
        Boolean HEALPix mask (galactic frame) of pixels not occluded by
        the Moon.

        Returns
        -------
        mask : ndarray of bool, shape (npix,)
            True for pixels whose line of sight from the spacecraft does
            not intersect the lunar sphere.
        """
        from .utils import moon_surface_distance

        npix = healpy.nside2npix(nside)
        vecs = np.array(
            healpy.pix2vec(nside, np.arange(npix)), dtype=np.float64
        )  # (3, npix)
        pos = self.spacecraft_position()          # (3,) metres, from Moon centre
        d = float(np.linalg.norm(pos))
        moon_dir = pos / d                        # direction from Moon centre to spacecraft
        # angle between each pixel direction and moon_dir (i.e. outward radial)
        dot = np.clip(moon_dir @ vecs, -1.0, 1.0)
        angle = np.arccos(dot)
        dist = moon_surface_distance(angle, d)
        return np.isnan(dist)                     # NaN → no intersection → pixel visible
