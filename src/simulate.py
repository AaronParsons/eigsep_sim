"""
JAX-based forward model for global 21cm radiometry.

Provides ForwardModel: composed-object simulator with coefficient-based
beam and sky (via spectral basis decomposition) and optional terrain.

Architecture:
- Model objects (Beam, Sky, Observer, Terrain) are immutable numpy-based descriptors
- JAX conversion happens at the ForwardModel boundary
- Simulates antenna temperature given basis coefficients
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import healpy

from .healpix import interpolate_map, float_dtype
from .beam import Beam
from .sky import Sky
from .observer import Observer
from .terrain import Terrain


# ─────────────────────────────────────────────────────────────────────────────
# JAX beam-integration kernels (reused from sim_jax)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0, 5))
def _beam_sum(beam_nside, beam_map, sky_masked, crds_top, rot_ms,
              npix_sky=None):
    """
    Beam-weighted sum over the gridded (HEALPix) sky.

    The denominator (beam solid angle) is rotation-invariant for a full-sphere
    sky and is precomputed once, halving the number of reductions per
    orientation vs. computing it inside the scan loop.

    Parameters
    ----------
    beam_nside : int  (static)
    beam_map   : (npix_beam, nfreq)
    sky_masked : (npix_sky,  nfreq)  sky already multiplied by horizon+terrain mask
    crds_top   : (3, npix_sky)       topocentric pixel unit vectors
    rot_ms     : (n_orient, 3, 3)    topocentric → beam-frame rotations
    npix_sky   : int or None  (static)
        Number of sky pixels.  If None, inferred from sky_masked.shape[0].
        Used to scale the beam sum to match the sky-pixel sampling density.

    Returns
    -------
    num : (n_orient, nfreq)   beam-weighted sky sum
    den : (n_orient, nfreq)   beam solid-angle (repeated for each orientation)
    """
    npix_beam = beam_map.shape[0]
    if npix_sky is None:
        npix_sky = sky_masked.shape[0]
    den_row = jnp.sum(beam_map, axis=0) * (npix_sky / npix_beam)

    def body(_, R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ crds_top))
        return None, jnp.sum(wgt * sky_masked, axis=0)

    _, num = jax.lax.scan(body, None, rot_ms)
    den = jnp.broadcast_to(den_row[None, :], num.shape)
    return num, den


# ─────────────────────────────────────────────────────────────────────────────
# ForwardModel
# ─────────────────────────────────────────────────────────────────────────────

class ForwardModel:
    """
    JAX-based forward model for global 21cm radiometry.

    Composes Observer, Beam, Sky, and optional Terrain objects into a single
    forward simulator. Coefficients (sky_coeffs, beam_coeffs) are parameters
    passed at simulation time, not stored in the model.

    All JAX conversion happens at the boundary (in simulate()); model objects
    remain pure numpy throughout.

    Parameters
    ----------
    observer : Observer
        Observer instance (EarthSurface, LunarSurface, or LunarOrbit).
    beam : Beam
        Beam instance with HEALPix coefficients and basis.
    sky : Sky
        Sky descriptor with basis (no stored coefficients).
    terrain : Terrain, optional
        Terrain model for horizon/occultation. If None, use observer.above_horizon().
    """

    def __init__(self, observer: Observer, beam: Beam, sky: Sky,
                 terrain: Terrain | None = None):
        """
        Initialize ForwardModel from composed objects.

        Parameters
        ----------
        observer : Observer
            Observer instance.
        beam : Beam
            Beam with HEALPix coefficients (n_dipoles, npix_beam, nmodes_beam).
        sky : Sky
            Sky descriptor (no state, pure metadata).
        terrain : Terrain, optional
            Optional terrain model.
        """
        self.observer = observer
        self.beam = beam
        self.sky = sky
        self.terrain = terrain

        # Cache static galactic coordinates
        self._crds_gal = np.array(
            healpy.pix2vec(sky.nside, np.arange(sky.npix)), dtype=np.float32
        )  # (3, npix_sky)

        # Static JAX arrays (created on demand)
        self._beam_basis_A_jax = None
        self._sky_basis_A_jax = None
        self._crds_gal_jax = None

    def _ensure_jax_arrays(self):
        """Convert and cache basis matrices as JAX arrays."""
        if self._beam_basis_A_jax is None:
            self._beam_basis_A_jax = jnp.asarray(self.beam.basis.A, dtype=float_dtype)
        if self._sky_basis_A_jax is None:
            self._sky_basis_A_jax = jnp.asarray(self.sky.basis.A, dtype=float_dtype)
        if self._crds_gal_jax is None:
            self._crds_gal_jax = jnp.asarray(self._crds_gal, dtype=float_dtype)

    def precompute_geometry(self, times):
        """
        Precompute rotation matrices and terrain masks for a list of times.

        This caches geometry that depends on observer position/orientation
        but not on sky/beam coefficients, enabling reuse across multiple
        model evaluations at the same times.

        Parameters
        ----------
        times : list of Time or array-like
            Observation epochs.

        Returns
        -------
        geom : dict
            Cached geometry with keys:
            - 'rot_gal2top': list of (3, 3) rotation matrices
            - 'crds_top': list of (3, npix_sky) topocentric coordinates
            - 'masks': list of (npix_sky,) float32 visibility masks (1=sky, 0=blocked)
        """
        from astropy.time import Time

        times = [Time(t) if not isinstance(t, Time) else t for t in times]
        geom = {'rot_gal2top': [], 'crds_top': [], 'masks': []}

        for t in times:
            self.observer.set_time(t)

            # Galactic → topocentric rotation
            R = self.observer.rot_gal2top().astype(np.float32)
            geom['rot_gal2top'].append(R)

            # Topocentric coordinates for all galactic pixels
            crds_top = R @ self._crds_gal
            geom['crds_top'].append(crds_top)

            # Visibility mask (1=sky, 0=blocked)
            mask = self._compute_mask(crds_top)
            geom['masks'].append(mask)

        return geom

    def _compute_mask(self, crds_top):
        """
        Compute visibility mask (1.0=visible, 0.0=blocked) at topocentric coordinates.

        Combines observer.above_horizon() with optional terrain masking.

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix_sky)

        Returns
        -------
        mask : ndarray, shape (npix_sky,), float32
        """
        # Geometric horizon / occultation
        geo_mask = self.observer.above_horizon(self.sky.nside).astype(np.float32)

        if self.terrain is not None:
            terrain_mask = self.terrain.mask(crds_top).astype(np.float32)
            return geo_mask * terrain_mask
        return geo_mask

    def simulate(self, sky_coeffs, beam_coeffs, times=None, geom=None,
                 T_gnd=300.0):
        """
        Simulate antenna temperature given basis coefficients.

        Either times or geom must be provided:
        - times: list of Time objects → precompute geometry on the fly
        - geom: pre-computed geometry dict from precompute_geometry()

        Parameters
        ----------
        sky_coeffs : ndarray, shape (npix_sky, nmodes_sky)
            Sky basis coefficients.
        beam_coeffs : ndarray, shape (n_dipoles, npix_beam, nmodes_beam)
            Beam basis coefficients per dipole.
        times : list of Time, optional
            Observation times. Ignored if geom is provided.
        geom : dict, optional
            Pre-computed geometry (from precompute_geometry()).
            If None, computed on the fly from times.
        T_gnd : float
            Ground temperature [K] for blocked pixels.

        Returns
        -------
        antenna_temp : ndarray, shape (ntimes, n_dipoles, nfreq)
            Antenna temperature [K] at each time and dipole orientation.
        """
        self._ensure_jax_arrays()

        # Compute geometry if not provided
        if geom is None:
            if times is None:
                raise ValueError("Either times or geom must be provided")
            geom = self.precompute_geometry(times)

        ntimes = len(geom['rot_gal2top'])
        n_dipoles = beam_coeffs.shape[0]
        nfreq = len(self.sky.freqs_hz)

        antenna_temp = np.zeros((ntimes, n_dipoles, nfreq), dtype=np.float32)

        # Convert coefficients to JAX
        sky_coeffs_jax = jnp.asarray(sky_coeffs, dtype=float_dtype)
        beam_coeffs_jax = jnp.asarray(beam_coeffs, dtype=float_dtype)

        # Reconstruct sky: (npix_sky, nfreq)
        sky_recon_jax = jnp.matmul(sky_coeffs_jax, self._sky_basis_A_jax.T)

        for ti in range(ntimes):
            # Apply mask and ground temperature
            mask = geom['masks'][ti]
            crds_top = geom['crds_top'][ti]
            sky_masked = sky_recon_jax * mask[:, None] + T_gnd * (1.0 - mask[:, None])
            sky_masked_jax = jnp.asarray(sky_masked, dtype=float_dtype)
            crds_top_jax = jnp.asarray(crds_top, dtype=float_dtype)

            # Process each dipole
            for di in range(n_dipoles):
                # Reconstruct beam for this dipole: (npix_beam, nfreq)
                beam_recon_jax = jnp.matmul(beam_coeffs_jax[di], self._beam_basis_A_jax.T)

                # Compute antenna temperature via beam-weighted sum
                num, den = _beam_sum(
                    self.beam.nside,
                    beam_recon_jax,
                    sky_masked_jax,
                    crds_top_jax,
                    jnp.eye(3, dtype=float_dtype)[None, :, :],  # Identity rotation
                    npix_sky=self.sky.npix
                )
                antenna_temp[ti, di, :] = np.asarray(num[0] / den[0])

        return antenna_temp


class SourceCatalog:
    """
    Placeholder for source catalog.

    In full implementations, would handle:
    - Fixed extragalactic/stellar sources (static galactic coordinates)
    - Solar-system bodies (ephemeris updates per timestep)
    - Point source interpolation into beam
    """
    pass
