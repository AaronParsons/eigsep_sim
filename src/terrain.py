"""
Terrain models for radio observations.

Provides abstract Terrain base class and concrete implementations:
- HorizonTerrain: HEALPix-based horizon distance map
- LunarDisk: Lunar occultation for orbital observers
- DEMTerrain: Digital elevation model (optional, requires eigsep_terrain)

All terrains provide a consistent interface: mask() for visibility and
emission() for thermal contribution.
"""

import numpy as np
import healpy
from abc import ABC, abstractmethod

from .const import R_MOON, GM_MOON


class Terrain(ABC):
    """Abstract terrain model providing visibility mask and thermal emission.

    Subclasses implement specific terrain types (horizon maps, lunar disk, DEM).
    """

    @abstractmethod
    def mask(self, crds_top):
        """Compute terrain visibility mask.

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix) or (npix, 3)
            Topocentric unit vectors pointing toward sky pixels.

        Returns
        -------
        ndarray, shape (npix,)
            Boolean mask: True = sky-visible, False = terrain-blocked.
        """

    @abstractmethod
    def emission(self, crds_top, freqs_hz):
        """Compute effective thermal emission from terrain.

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix) or (npix, 3)
            Topocentric unit vectors.
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].

        Returns
        -------
        ndarray, shape (npix, nfreq)
            Effective temperature [K] of terrain at each pixel/frequency.
        """


class NullTerrain(Terrain):
    """No-op terrain: all sky visible, no thermal emission."""

    def mask(self, crds_top):
        """All pixels visible."""
        if crds_top.shape[0] == 3:
            return np.ones(crds_top.shape[1], dtype=bool)
        else:
            return np.ones(crds_top.shape[0], dtype=bool)

    def emission(self, crds_top, freqs_hz):
        """Zero thermal emission."""
        if crds_top.shape[0] == 3:
            npix = crds_top.shape[1]
        else:
            npix = crds_top.shape[0]
        return np.zeros((npix, len(freqs_hz)), dtype=np.float32)


class HorizonTerrain(Terrain):
    """HEALPix-stored horizon distance map.

    Stores terrain as a HEALPix map where each pixel contains either:
    - NaN: no obstruction (open sky)
    - float value: horizon distance at that pixel [meters]

    Parameters
    ----------
    nside : int
        HEALPix resolution of the horizon map.
    horizon_map : ndarray, shape (npix,)
        Horizon distance per pixel [m]. NaN = no obstruction.
    T_terrain : float, optional
        Uniform terrain temperature [K] (default 300 K).
    reflectivity : ndarray, shape (npix,) or (npix, nfreq), optional
        Terrain reflectivity per pixel. If provided, used to scale emission.
        If None, terrain pixels are treated as thermal emitters only (no reflection).
    nside_sky : int, optional
        HEALPix resolution of sky pixels for interpolation. If None, assume
        horizon_map and sky use the same nside.
    """

    def __init__(self, nside, horizon_map, T_terrain=300.0, reflectivity=None,
                 nside_sky=None):
        self.nside = int(nside)
        self.npix = healpy.nside2npix(self.nside)
        self.horizon_map = np.asarray(horizon_map, dtype=np.float32)
        self.T_terrain = float(T_terrain)
        self.reflectivity = reflectivity
        self.nside_sky = nside_sky if nside_sky is not None else nside

        if self.horizon_map.shape[0] != self.npix:
            raise ValueError(f"horizon_map shape {self.horizon_map.shape} "
                           f"inconsistent with nside={nside} (npix={self.npix})")

    def mask(self, crds_top):
        """Compute visibility from horizon distance map.

        A pixel is blocked if any point on the ray from observer to pixel
        intersects the terrain. For efficiency, uses a simple check: if the
        horizon distance at the pixel is non-NaN and positive, terrain blocks it.

        Parameters
        ----------
        crds_top : ndarray
            Topocentric unit vectors, shape (3, npix) or (npix, 3).

        Returns
        -------
        ndarray, shape (npix,)
            Boolean mask.
        """
        # Normalize input
        if crds_top.shape[0] == 3:
            # (3, npix) format
            npix_sky = crds_top.shape[1]
        else:
            # (npix, 3) format
            npix_sky = crds_top.shape[0]
            crds_top = crds_top.T  # (3, npix)

        # If sky resolution differs from terrain, interpolate
        if npix_sky == self.npix:
            # Same resolution: direct lookup
            mask = np.isnan(self.horizon_map)
        else:
            # Different resolution: interpolate horizon map to sky pixels
            theta, phi = healpy.vec2ang(crds_top, lonlat=False)
            horizon_interp = healpy.get_interp_val(self.horizon_map, theta, phi,
                                                   lonlat=False)
            mask = np.isnan(horizon_interp)

        return mask.astype(bool)

    def emission(self, crds_top, freqs_hz):
        """Return terrain temperature for blocked pixels.

        Parameters
        ----------
        crds_top : ndarray
            Topocentric unit vectors.
        freqs_hz : ndarray
            Frequencies [Hz].

        Returns
        -------
        ndarray, shape (npix, nfreq)
            Terrain temperature (same for all frequencies if T_terrain is scalar).
        """
        # Normalize input
        if crds_top.shape[0] == 3:
            npix_sky = crds_top.shape[1]
        else:
            npix_sky = crds_top.shape[0]

        nfreq = len(freqs_hz)
        emission = np.zeros((npix_sky, nfreq), dtype=np.float32)

        # Blocked pixels (where horizon_map is not NaN) get terrain temperature
        mask = ~self.mask(crds_top)  # True = blocked
        emission[mask] = self.T_terrain

        return emission

    def set_temperature(self, T):
        """Set uniform terrain temperature.

        Parameters
        ----------
        T : float or ndarray
            Temperature [K]. If float, sets uniform temperature.
            If ndarray, should have shape (npix,).
        """
        if np.ndim(T) == 0:
            self.T_terrain = float(T)
        else:
            # Per-pixel temperature not yet implemented
            raise NotImplementedError("Per-pixel temperature not yet supported")


class LunarDisk(Terrain):
    """Lunar occultation mask for an orbiting observer.

    For a spacecraft at position pos_gal (galactic frame), computes which
    sky directions are blocked by the lunar disk.

    Parameters
    ----------
    nside : int
        HEALPix resolution.
    moon_radius_m : float, optional
        Lunar radius [m] (default 1,737,400 m).
    T_regolith : float, optional
        Lunar surface temperature [K] (default 300 K).
    """

    def __init__(self, nside, moon_radius_m=R_MOON, T_regolith=300.0):
        self.nside = int(nside)
        self.npix = healpy.nside2npix(self.nside)
        self.moon_radius = float(moon_radius_m)
        self.T_regolith = float(T_regolith)
        self.spacecraft_pos_gal = None  # Set via update()

    def update(self, spacecraft_position_gal):
        """Update spacecraft position (galactic frame, meters).

        Parameters
        ----------
        spacecraft_position_gal : ndarray, shape (3,)
            Spacecraft position [m] in galactic Cartesian coordinates.
        """
        self.spacecraft_pos_gal = np.asarray(spacecraft_position_gal,
                                            dtype=np.float64)

    def mask(self, crds_top):
        """Compute lunar occultation mask.

        A pixel is blocked if the ray from spacecraft toward that sky direction
        intersects the lunar sphere.

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix) or (npix, 3)
            Sky directions in galactic frame.

        Returns
        -------
        ndarray, shape (npix,)
            Boolean mask: True = sky visible, False = lunar disk blocks.
        """
        if self.spacecraft_pos_gal is None:
            # No spacecraft position set: all sky visible
            if crds_top.shape[0] == 3:
                return np.ones(crds_top.shape[1], dtype=bool)
            else:
                return np.ones(crds_top.shape[0], dtype=bool)

        # Normalize input
        if crds_top.shape[0] == 3:
            # (3, npix) format
            sky_dirs = crds_top  # (3, npix)
            npix = crds_top.shape[1]
        else:
            # (npix, 3) format
            sky_dirs = crds_top.T  # (3, npix)
            npix = crds_top.shape[0]

        # Lunar position (at origin in galactic frame)
        moon_pos = np.array([0.0, 0.0, 0.0])

        # For each sky pixel, compute ray-sphere intersection
        # Ray from spacecraft toward sky pixel: P(t) = pos_sc + t * dir
        # Sphere: |P - moon_pos|^2 = R^2
        # Solve: |pos_sc + t*dir|^2 = R^2

        # Vector from moon to spacecraft
        sc_to_moon = self.spacecraft_pos_gal - moon_pos  # (3,)
        d = sc_to_moon / np.linalg.norm(sc_to_moon)  # normalized

        # For each direction, check if ray intersects sphere
        visible = np.ones(npix, dtype=bool)

        for i in range(npix):
            ray_dir = sky_dirs[:, i] / np.linalg.norm(sky_dirs[:, i])

            # Quadratic coefficients for ray-sphere intersection
            # |P(t)|^2 = |sc + t*dir|^2 = R^2
            a = np.dot(ray_dir, ray_dir)  # should be 1
            b = 2 * np.dot(ray_dir, sc_to_moon)
            c = np.dot(sc_to_moon, sc_to_moon) - self.moon_radius**2

            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                # Ray intersects sphere; check if in front of spacecraft
                t1 = (-b - np.sqrt(discriminant)) / (2*a)
                t2 = (-b + np.sqrt(discriminant)) / (2*a)
                # Pixel is blocked if sphere is in front (t > 0)
                if t1 > 1e-6 or (t2 > 1e-6 and t1 < 1e-6):
                    visible[i] = False

        return visible

    def emission(self, crds_top, freqs_hz):
        """Return regolith temperature for lunar disk.

        Parameters
        ----------
        crds_top : ndarray
            Sky directions.
        freqs_hz : ndarray
            Frequencies [Hz].

        Returns
        -------
        ndarray, shape (npix, nfreq)
            Lunar surface temperature for blocked pixels, zero elsewhere.
        """
        # Normalize input
        if crds_top.shape[0] == 3:
            npix = crds_top.shape[1]
        else:
            npix = crds_top.shape[0]

        nfreq = len(freqs_hz)
        emission = np.zeros((npix, nfreq), dtype=np.float32)

        # Blocked pixels get lunar regolith temperature
        mask = ~self.mask(crds_top)  # True = blocked
        emission[mask] = self.T_regolith

        return emission

    def set_temperature(self, T):
        """Set lunar surface temperature.

        Parameters
        ----------
        T : float or ndarray
            Temperature [K]. If float, sets uniform temperature.
        """
        if np.ndim(T) == 0:
            self.T_regolith = float(T)
        else:
            raise NotImplementedError("Per-pixel temperature not yet supported")


class DEMTerrain(Terrain):
    """DEM-backed terrain using eigsep_terrain.DEM.

    Wraps a digital elevation model and uses ray tracing to compute horizon
    distances and terrain masks.

    This class is optional and requires eigsep_terrain to be installed.

    Parameters
    ----------
    dem : eigsep_terrain.dem.DEM
        DEM object with ray_trace() method.
    observer : Observer
        Observer object (e.g., EarthSurface) with location info.
    T_terrain : float, optional
        Uniform terrain temperature [K] (default 300 K).
    nside_beam : int, optional
        HEALPix resolution for ray tracing (default 8).
    """

    def __init__(self, dem, observer, T_terrain=300.0, nside_beam=8):
        try:
            from eigsep_terrain.dem import DEM
        except ImportError:
            raise ImportError("eigsep_terrain required for DEMTerrain; "
                            "install via `pip install eigsep_terrain`")

        self.dem = dem
        self.observer = observer
        self.T_terrain = float(T_terrain)
        self.nside_beam = int(nside_beam)
        self._horizon_map = None

    def _compute_horizon(self):
        """Compute horizon distance map via DEM ray tracing."""
        if self._horizon_map is not None:
            return self._horizon_map

        # Use DEM's ray_trace method to compute horizon distances
        # This is a placeholder; actual implementation depends on DEM API
        raise NotImplementedError("DEMTerrain ray_trace integration pending")

    def mask(self, crds_top):
        """Compute visibility from DEM via ray tracing."""
        horizon = self._compute_horizon()
        # Use horizon map to determine visibility
        # (similar to HorizonTerrain logic)
        raise NotImplementedError("DEMTerrain mask() pending")

    def emission(self, crds_top, freqs_hz):
        """Return terrain temperature from DEM."""
        raise NotImplementedError("DEMTerrain emission() pending")
