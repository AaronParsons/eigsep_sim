"""Solar-system body ephemeris for external radio sources (Sun, Earth, ...).

Companion to :mod:`observer` — where ``Observer`` subclasses give the
spacecraft's *own* position/attitude, this module gives the position of
everything else (Sun, Earth) relative to the Moon, in the same inertial
galactic frame used throughout the rest of the package (``ICRS2GAL``).

Uses astropy's built-in low-precision analytical ephemeris
(``get_body_barycentric``) — no external kernel download required, and
accurate to well under a HEALPix pixel at ``nside <= 64`` (~1 deg), which is
all that's needed to place an extended source (Sun, Earth) on the sky and
gate it by lunar occultation.  Swap in a JPL DE44x kernel via
``astropy.coordinates.solar_system_ephemeris.set('jpl')`` (requires the
``jplephem`` package and a downloaded kernel) if higher precision is ever
needed — not expected for this use case.

Note on frames: the *inertial* (galactic) direction from the Moon to Earth
sweeps a full circle roughly once per sidereal month (~27.3 days, the
Moon's orbital period around Earth) — it is only in the Moon **body-fixed**
frame (MCMF) that Earth appears nearly stationary (that's what tidal
locking means).  The Sun's inertial direction from the Moon sweeps once per
synodic month (~29.5 days); in the MCMF frame this sweep *is* the lunar
day/night cycle relevant to surface heating.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import astropy.units as u

from .observer import ICRS2GAL, _moon_icrs2mcmf
from .const import R_MOON, R_SUN, R_EARTH

_BODY_RADII_M = {"sun": R_SUN, "earth": R_EARTH}


def body_directions_gal(times, bodies=("sun", "earth")):
    """Unit vectors + distances from the Moon's centre to ``bodies``.

    Parameters
    ----------
    times : Time-like, shape (ntimes,)
    bodies : sequence of str
        Names understood by ``astropy.coordinates.get_body_barycentric``
        (e.g. ``"sun"``, ``"earth"``).

    Returns
    -------
    dirs : dict[str, ndarray (ntimes, 3)]
        Unit vectors, galactic frame, Moon centre -> body.
    dists : dict[str, ndarray (ntimes,)]
        Distances [m].
    """
    times = Time(times)
    moon_pos = get_body_barycentric("moon", times)
    dirs, dists = {}, {}
    for body in bodies:
        rel = get_body_barycentric(body, times) - moon_pos
        xyz = rel.xyz.to(u.m).value  # (3, ntimes)
        d = np.linalg.norm(xyz, axis=0)
        u_icrs = xyz / d
        u_gal = ICRS2GAL @ u_icrs  # (3, ntimes)
        dirs[body] = np.ascontiguousarray(u_gal.T, dtype=np.float64)  # (ntimes, 3)
        dists[body] = d
    return dirs, dists


def body_angular_radius(body, dist_m):
    """Angular radius [rad] of ``body`` at distance ``dist_m`` (small-angle)."""
    return _BODY_RADII_M[body] / np.asarray(dist_m)


def body_occulted_by_moon(orbit, times, body_dir_gal):
    """Boolean mask, shape (ntimes,): True where the Moon blocks the line of
    sight from ``orbit``'s spacecraft position to ``body_dir_gal[t]``.

    Same eclipse geometry as ``LunarOrbit.above_horizon_stack``, evaluated
    for one time-varying direction (an external body) instead of a fixed
    HEALPix grid.  Parallax between "from Moon centre" and "from spacecraft"
    is neglected, consistent with how the rest of the pipeline treats the
    sky as being at infinity (valid here too: orbital radius is ~1e-2x the
    Earth distance and ~1e-5x the Sun distance).

    Parameters
    ----------
    orbit : LunarOrbit
    times : Time-like, shape (ntimes,)
    body_dir_gal : ndarray, shape (ntimes, 3)
        Unit vectors, galactic frame (e.g. from ``body_directions_gal``).

    Returns
    -------
    occulted : ndarray of bool, shape (ntimes,)
    """
    pos = orbit.spacecraft_position_stack(times)  # (ntimes, 3) m
    d = np.linalg.norm(pos, axis=1)
    moon_dir = pos / d[:, None]
    dot = np.sum(moon_dir * body_dir_gal, axis=1)
    limb_dot = -np.sqrt(np.maximum(0.0, 1.0 - (R_MOON / d) ** 2))
    return dot <= limb_dot


def moon_surface_intersection_mcmf(orbit, times, sky_dirs_gal):
    """Where an occulted line of sight actually terminates on the Moon.

    For each (time, sky direction) finds the near intersection of the ray
    from the spacecraft along that direction with the Moon's sphere, and
    returns it as a body-fixed (MCMF) unit surface-normal vector -- i.e.
    the selenographic point a blocked sky pixel's brightness should
    physically be evaluated at (see :mod:`regolith`), rather than treating
    every occulted pixel as an undifferentiated single temperature.

    Parameters
    ----------
    orbit : LunarOrbit
    times : Time-like, shape (ntimes,)
    sky_dirs_gal : ndarray, shape (ntimes, npix, 3) or (npix, 3)
        Galactic-frame unit vectors for each sky pixel; a (npix, 3) array
        is broadcast across all times.

    Returns
    -------
    normals_mcmf : ndarray, shape (ntimes, npix, 3)
        Body-fixed unit surface-normal vectors at the intersection point.
        ``NaN`` where the ray does not actually intersect the sphere (not
        occulted) -- callers should already have an occultation mask (e.g.
        from ``LunarOrbit.above_horizon_stack``) and only use this where
        that mask is False.
    """
    times = Time(times)
    ntimes = len(times)
    pos = orbit.spacecraft_position_stack(times)  # (ntimes, 3) m
    d = np.linalg.norm(pos, axis=1)  # (ntimes,)

    sky_dirs_gal = np.asarray(sky_dirs_gal, dtype=np.float64)
    if sky_dirs_gal.ndim == 2:
        sky_dirs_gal = np.broadcast_to(
            sky_dirs_gal, (ntimes,) + sky_dirs_gal.shape
        )

    # Ray-sphere intersection: |pos + t*u|^2 = R_MOON^2, near root, t>0
    # (a mathematically valid root behind the spacecraft, e.g. for a ray
    # pointing away from the Moon, does not count as a hit).
    pos_dot_u = np.einsum("ti,tpi->tp", pos, sky_dirs_gal)  # (ntimes, npix)
    disc = pos_dot_u**2 - (d[:, None] ** 2 - R_MOON**2)
    t_near = -pos_dot_u - np.sqrt(np.maximum(disc, 0.0))  # (ntimes, npix)
    valid = (disc >= 0.0) & (t_near > 0.0)

    intersection_gal = (
        pos[:, None, :] + t_near[..., None] * sky_dirs_gal
    )  # (ntimes, npix, 3)
    normal_gal = intersection_gal / np.linalg.norm(
        intersection_gal, axis=-1, keepdims=True
    )
    normal_gal[~valid] = np.nan

    gal2icrs = ICRS2GAL.T
    out = np.empty_like(normal_gal)
    for i, t in enumerate(times):
        icrs2mcmf = _moon_icrs2mcmf(t)
        out[i] = (icrs2mcmf @ (gal2icrs @ normal_gal[i].T)).T
    return out
