"""Compatibility classes for uniform lunar occultation emission.

Lunar occultation geometry is owned by :class:`eigsep_sim.observer.LunarOrbit`
and consumed by :class:`eigsep_sim.simulate.ForwardModel`.  These classes remain
as light-weight compatibility helpers for older code that used
``ForwardModel(..., surface_model=UniformLunarSurface(...))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .const import R_MOON


@dataclass(frozen=True)
class LunarSurfaceGeometry:
    """Deprecated container for legacy lunar-surface intersection results."""

    blocked: np.ndarray
    intercepts_gal_m: np.ndarray
    normals_gal: np.ndarray

    @property
    def sky_mask(self):
        """Visibility mask with ``True`` for unobstructed sky rays."""
        return ~self.blocked


class LunarSurfaceModel:
    """Deprecated base class for lunar occultation emission compatibility."""

    def __init__(self, moon_radius_m=R_MOON):
        self.moon_radius_m = float(moon_radius_m)

    def prepare_geometry(self, spacecraft_positions_gal_m, sky_dirs_gal):
        """Intersect spacecraft rays with a spherical Moon.

        New code should use ``LunarOrbit.above_horizon*`` for visibility. This
        method remains for callers that inspect legacy geometry diagnostics.
        """
        positions = np.asarray(spacecraft_positions_gal_m, dtype=float)
        if positions.ndim == 1:
            positions = positions[None, :]
        directions = np.asarray(sky_dirs_gal, dtype=float)
        if directions.shape[0] != 3:
            directions = directions.T
        directions = directions / np.linalg.norm(
            directions, axis=0, keepdims=True
        )
        dot = positions @ directions
        radius_sq = np.einsum("ti,ti->t", positions, positions)
        disc = dot**2 - (radius_sq[:, None] - self.moon_radius_m**2)
        blocked = (dot < 0.0) & (disc >= 0.0)
        near_t = -dot - np.sqrt(np.maximum(disc, 0.0))
        blocked &= near_t >= 0.0
        intercepts = (
            positions[:, None, :]
            + near_t[:, :, None] * directions.T[None, :, :]
        )
        intercepts = np.where(blocked[:, :, None], intercepts, np.nan)
        normals = intercepts / self.moon_radius_m
        return LunarSurfaceGeometry(blocked, intercepts, normals)

    def thermal_emission(self, geometry, freqs_hz):
        """Return lunar thermal brightness for legacy geometry."""
        raise NotImplementedError

    def unresolved_emission(self, freqs_hz):
        """Return the omitted-pixel spectrum, or ``None`` if inexact."""
        return None


class UniformLunarSurface(LunarSurfaceModel):
    """Uniform-temperature lunar occultation emission compatibility helper."""

    def __init__(self, T_regolith_K=300.0, moon_radius_m=R_MOON):
        super().__init__(moon_radius_m=moon_radius_m)
        self.T_regolith_K = float(T_regolith_K)

    def thermal_emission(self, geometry, freqs_hz):
        emission = np.zeros(
            geometry.blocked.shape + (len(freqs_hz),), dtype=np.float32
        )
        emission[geometry.blocked] = self.T_regolith_K
        return emission

    def unresolved_emission(self, freqs_hz):
        """Uniform disk spectrum for exact reduced ``sky_mask`` geometry."""
        return np.full(len(freqs_hz), self.T_regolith_K, dtype=np.float32)
