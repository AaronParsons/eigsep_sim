"""
Ephemeris for a spacecraft in circular lunar orbit.

The spacecraft attitude is defined by a spin rotation applied to an
initial frame aligned with the galactic frame.  The orbital position
is tracked separately and used for lunar occultation masking.
"""

import numpy as np
import healpy
from astropy.time import Time
import astropy.units as u

from .coord import rot_m
from .const import R_MOON


class LunarOrbit:
    """
    Spacecraft in a circular lunar orbit with an independent spin.

    The galactic frame is used as the inertial reference throughout,
    consistent with the GSM sky model.

    Parameters
    ----------
    altitude : float
        Orbital altitude above the lunar surface, metres.
    rot_orbit_vec : array_like, shape (3,)
        Unit vector in the galactic frame normal to the orbital plane.
        The spacecraft orbits counter-clockwise around this axis.
    rot_spin_vec : array_like, shape (3,)
        Unit vector in the galactic frame about which the spacecraft spins.
    start_pos : array_like, shape (3,), optional
        Unit vector in the galactic frame giving the initial orbital
        position direction from the Moon centre.  Default: [1, 0, 0].
    orbital_period : float
        Orbital period in seconds.  Default: 7200 (2 h, ~100 km altitude).
    spin_period : float
        Spacecraft spin period in seconds.  0 means no spin.
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
        orbital_period=7200.0,
        spin_period=0.0,
        t0=None,
    ):
        self.altitude = altitude
        self.orbital_radius = R_MOON + altitude

        self.rot_orbit_vec = np.asarray(rot_orbit_vec, dtype=float)
        self.rot_orbit_vec /= np.linalg.norm(self.rot_orbit_vec)

        self.rot_spin_vec = np.asarray(rot_spin_vec, dtype=float)
        self.rot_spin_vec /= np.linalg.norm(self.rot_spin_vec)

        if start_pos is None:
            start_pos = np.array([1.0, 0.0, 0.0])
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_pos /= np.linalg.norm(self.start_pos)

        self.orbital_period = float(orbital_period)
        self.spin_period = float(spin_period)
        self.t0 = Time("J2000") if t0 is None else Time(t0)
        self.time = self.t0

        self._th_orbit = 0.0
        self._th_spin = 0.0

    def set_time(self, t):
        """Set the current epoch and update orbital and spin phases."""
        self.time = Time(t)
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
