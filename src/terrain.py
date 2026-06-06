"""
Terrain models for radio observations.

Provides abstract Terrain base class and concrete implementations:
- HorizonTerrain: HEALPix-based horizon distance map

All terrains provide a consistent interface: mask() for visibility and
emission() for thermal contribution.
"""

import os
import numpy as np
import healpy
from abc import ABC, abstractmethod

HORIZON_MODELS_NPZ = os.path.join(
    os.path.dirname(__file__), "data", "horizon_models_v000.npz"
)


class Terrain(ABC):
    """Abstract terrain model providing visibility mask and thermal emission.

    Subclasses implement specific terrain types (horizon maps, DEM).
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

    def __init__(
        self,
        nside,
        horizon_map,
        T_terrain=300.0,
        reflectivity=None,
        nside_sky=None,
        center=None,
        height=None,
        metadata=None,
    ):
        self.nside = int(nside)
        self.npix = healpy.nside2npix(self.nside)
        self.horizon_map = np.asarray(horizon_map, dtype=np.float32)
        self.T_terrain = float(T_terrain)
        self.reflectivity = reflectivity
        self.nside_sky = nside_sky if nside_sky is not None else nside
        self.center = (
            None if center is None else np.asarray(center, dtype=np.float32)
        )
        self.height = None if height is None else float(height)
        self.metadata = {} if metadata is None else dict(metadata)

        if self.horizon_map.shape[0] != self.npix:
            raise ValueError(
                f"horizon_map shape {self.horizon_map.shape} "
                f"inconsistent with nside={nside} (npix={self.npix})"
            )

    @classmethod
    def from_file(
        cls,
        path,
        index=None,
        height=None,
        T_terrain=300.0,
        reflectivity=None,
        nside_sky=None,
    ):
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
                raise ValueError(
                    "Cannot select by height: NPZ has no 'heights'"
                )
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
        pix = healpy.vec2pix(self.nside, crds_top[0], crds_top[1], crds_top[2])
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
            raise NotImplementedError(
                "Per-pixel temperature not yet supported"
            )
