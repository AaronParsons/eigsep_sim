"""
Terrain models for radio observations.

Provides abstract Terrain base class and concrete implementations:
- HorizonTerrain: HEALPix-based horizon distance map
- LunarDisk: Lunar occultation for orbital observers
- DEMTerrain: Digital elevation model (optional, requires eigsep_terrain)

All terrains provide a consistent interface: mask() for visibility and
emission() for thermal contribution.
"""

import os
import numpy as np
import healpy
from abc import ABC, abstractmethod

from .const import R_MOON, GM_MOON

HORIZON_MODELS_NPZ = os.path.join(
    os.path.dirname(__file__), "data", "horizon_models_v000.npz"
)


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

    def unresolved_emission(self, freqs_hz):
        """Emission spectrum for terrain-blocked pixels omitted by sky_mask.

        Return ``None`` when omitted blocked pixels cannot be represented by a
        single spectrum. ForwardModel then requires full geometry for exact
        terrain emission.
        """
        return None


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

    def unresolved_emission(self, freqs_hz):
        """No terrain emission for omitted pixels."""
        return np.zeros(len(freqs_hz), dtype=np.float32)


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
                 nside_sky=None, center=None, height=None, metadata=None):
        self.nside = int(nside)
        self.npix = healpy.nside2npix(self.nside)
        self.horizon_map = np.asarray(horizon_map, dtype=np.float32)
        self.T_terrain = float(T_terrain)
        self.reflectivity = reflectivity
        self.nside_sky = nside_sky if nside_sky is not None else nside
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.height = None if height is None else float(height)
        self.metadata = {} if metadata is None else dict(metadata)

        if self.horizon_map.shape[0] != self.npix:
            raise ValueError(f"horizon_map shape {self.horizon_map.shape} "
                           f"inconsistent with nside={nside} (npix={self.npix})")

    @classmethod
    def from_file(cls, path, index=None, height=None, T_terrain=300.0,
                  reflectivity=None, nside_sky=None):
        """Load a precomputed HEALPix horizon model from an NPZ file.

        The packaged Marjum file stores:
        - ``r``: horizon distance maps, shape ``(n_models, npix)``
        - ``nside``: HEALPix nside for each map
        - ``heights``: antenna heights above the modeled center, metres
        - ``centers``: DEM/grid center coordinates for each height slice

        Finite ``r`` values are terrain-blocked directions; ``NaN`` values are
        open sky. Select one model by explicit ``index`` or nearest ``height``.
        If neither is given, the first model is used.
        """
        npz = np.load(path, allow_pickle=False)
        if "r" not in npz or "nside" not in npz:
            raise ValueError("horizon model NPZ must contain 'r' and 'nside'")

        maps = np.asarray(npz["r"], dtype=np.float32)
        if maps.ndim == 1:
            maps = maps[None, :]
        n_models = maps.shape[0]

        if index is not None and height is not None:
            raise ValueError("Specify either index or height, not both")
        if height is not None:
            if "heights" not in npz:
                raise ValueError("Cannot select by height: NPZ has no 'heights'")
            heights = np.asarray(npz["heights"], dtype=float)
            index = int(np.argmin(np.abs(heights - float(height))))
        elif index is None:
            index = 0

        index = int(index)
        if not 0 <= index < n_models:
            raise IndexError(f"index {index} out of range [0, {n_models})")

        center = npz["centers"][index] if "centers" in npz else None
        selected_height = npz["heights"][index] if "heights" in npz else None
        metadata = {
            "path": str(path),
            "index": index,
            "available_heights": npz["heights"] if "heights" in npz else None,
            "available_centers": npz["centers"] if "centers" in npz else None,
        }
        return cls(
            int(npz["nside"]),
            maps[index],
            T_terrain=T_terrain,
            reflectivity=reflectivity,
            nside_sky=nside_sky,
            center=center,
            height=selected_height,
            metadata=metadata,
        )

    @classmethod
    def from_packaged_model(cls, index=None, height=None, **kwargs):
        """Load the packaged Marjum horizon model.

        Parameters are forwarded to :meth:`from_file`; use ``height=...`` to
        select the nearest antenna-height slice.
        """
        return cls.from_file(
            HORIZON_MODELS_NPZ, index=index, height=height, **kwargs
        )

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

        # Always use the supplied coordinates. Direct array lookup is only valid
        # for unrotated native HEALPix pixel centers; using it for rotated sky
        # directions scrambles otherwise contiguous terrain regions. Treat the
        # horizon map as a categorical visibility mask and use nearest-neighbor
        # lookup instead of interpolating NaNs.
        pix = healpy.vec2pix(
            self.nside, crds_top[0], crds_top[1], crds_top[2]
        )
        mask = np.isnan(self.horizon_map[pix])

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

    def unresolved_emission(self, freqs_hz):
        """Uniform terrain temperature for terrain-blocked omitted pixels."""
        return np.full(len(freqs_hz), self.T_terrain, dtype=np.float32)

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
        if crds_top.shape[0] == 3:
            sky_dirs = np.asarray(crds_top, dtype=np.float64)
        else:
            sky_dirs = np.asarray(crds_top, dtype=np.float64).T

        npix = sky_dirs.shape[1]
        if self.spacecraft_pos_gal is None:
            return np.ones(npix, dtype=bool)

        sky_dirs = sky_dirs / np.linalg.norm(sky_dirs, axis=0, keepdims=True)
        distance = float(np.linalg.norm(self.spacecraft_pos_gal))
        moon_to_spacecraft = self.spacecraft_pos_gal / distance
        limb_dot = -np.sqrt(max(0.0, 1.0 - (self.moon_radius / distance) ** 2))
        return (moon_to_spacecraft @ sky_dirs) > limb_dot

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

    def unresolved_emission(self, freqs_hz):
        """Uniform regolith temperature for lunar-disk omitted pixels."""
        return np.full(len(freqs_hz), self.T_regolith, dtype=np.float32)

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
