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
from .const import DTYPE_R_NPY, DTYPE_R_JAX
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
            healpy.pix2vec(sky.nside, np.arange(sky.npix)), dtype=DTYPE_R_NPY
        )  # (3, npix_sky)

        # Static JAX arrays (created on demand)
        self._beam_basis_A_jax = None
        self._sky_basis_A_jax = None
        self._crds_gal_jax = None

    def _ensure_jax_arrays(self):
        """Convert and cache basis matrices as JAX arrays; build JIT-compiled sim kernel."""
        if self._beam_basis_A_jax is None:
            self._beam_basis_A_jax = jnp.asarray(self.beam.basis.A, dtype=DTYPE_R_JAX)
        if self._sky_basis_A_jax is None:
            self._sky_basis_A_jax = jnp.asarray(self.sky.basis.A, dtype=DTYPE_R_JAX)
        if self._crds_gal_jax is None:
            self._crds_gal_jax = jnp.asarray(self._crds_gal, dtype=DTYPE_R_JAX)
        if not hasattr(self, '_sim_jit'):
            self._sim_jit = self._build_sim_fn()

    def _build_sim_fn(self):
        """Build and JIT-compile the inner simulation kernel.

        Returns a function ``sim(sky_coeffs, beam_coeffs, masks, crds_top, T_gnd)``
        that is fully JAX-traceable: basis matrices, nside, and pixel counts are
        captured as compile-time constants so the XLA kernel is compiled once and
        reused across every call with the same array shapes.
        """
        A_sky = self._sky_basis_A_jax        # (nfreq, nmodes_sky)
        A_beam = self._beam_basis_A_jax      # (nfreq, nmodes_beam)
        beam_nside = self.beam.nside         # static int
        npix_beam = self.beam.npix           # static int
        npix_sky = self.sky.npix             # static int
        scale = float(npix_sky) / float(npix_beam)

        @jax.jit
        def _sim(sky_coeffs, beam_coeffs, masks, crds_top, T_gnd):
            """
            sky_coeffs  : (npix_sky, nmodes_sky)
            beam_coeffs : (n_dipoles, npix_beam, nmodes_beam)
            masks       : (ntimes, npix_sky)       float32  1=visible 0=blocked
            crds_top    : (ntimes, 3, npix_sky)    float32  topocentric unit vecs
            T_gnd       : scalar [K]
            Returns     : (ntimes, n_dipoles, nfreq)
            """
            sky_recon = sky_coeffs @ A_sky.T          # (npix_sky, nfreq)
            beam_recon_all = beam_coeffs @ A_beam.T   # (n_dipoles, npix_beam, nfreq)

            def one_time(_, args):
                mask, crds = args              # (npix_sky,), (3, npix_sky)
                sky_masked = sky_recon * mask[:, None] + T_gnd * (1.0 - mask[:, None])

                def one_dipole(beam_recon):   # (npix_beam, nfreq)
                    wgt = interpolate_map(beam_nside, beam_recon, *crds)  # (npix_sky, nfreq)
                    num = jnp.sum(wgt * sky_masked, axis=0)               # (nfreq,)
                    den = jnp.sum(beam_recon, axis=0) * scale             # (nfreq,)
                    return num / den

                return None, jax.vmap(one_dipole)(beam_recon_all)  # (n_dipoles, nfreq)

            _, antenna_temp = jax.lax.scan(one_time, None, (masks, crds_top))
            return antenna_temp  # (ntimes, n_dipoles, nfreq)

        return _sim

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

        # Stacked JAX arrays for JIT-compiled simulation kernel
        geom['masks_jax'] = jnp.stack(
            [jnp.asarray(m, dtype=DTYPE_R_JAX) for m in geom['masks']]
        )  # (ntimes, npix_sky)
        geom['crds_top_jax'] = jnp.stack(
            [jnp.asarray(c, dtype=DTYPE_R_JAX) for c in geom['crds_top']]
        )  # (ntimes, 3, npix_sky)
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
        antenna_temp : jnp.ndarray, shape (ntimes, n_dipoles, nfreq)
            Antenna temperature [K] at each time and dipole orientation.
        """
        self._ensure_jax_arrays()

        # Compute geometry if not provided
        if geom is None:
            if times is None:
                raise ValueError("Either times or geom must be provided")
            geom = self.precompute_geometry(times)

        # Convert coefficients to JAX
        sky_coeffs_jax = jnp.asarray(sky_coeffs, dtype=DTYPE_R_JAX)
        beam_coeffs_jax = jnp.asarray(beam_coeffs, dtype=DTYPE_R_JAX)

        # Get stacked geometry (cached by precompute_geometry, or build lazily)
        masks_jax = geom.get('masks_jax')
        crds_top_jax = geom.get('crds_top_jax')
        if masks_jax is None:
            masks_jax = jnp.stack(
                [jnp.asarray(m, dtype=DTYPE_R_JAX) for m in geom['masks']]
            )
        if crds_top_jax is None:
            crds_top_jax = jnp.stack(
                [jnp.asarray(c, dtype=DTYPE_R_JAX) for c in geom['crds_top']]
            )

        # Single JIT-compiled call: scan over time, vmap over dipoles
        return self._sim_jit(
            sky_coeffs_jax, beam_coeffs_jax, masks_jax, crds_top_jax,
            jnp.asarray(T_gnd, dtype=DTYPE_R_JAX)
        )


class SourceCatalog:
    """
    Placeholder for source catalog.

    In full implementations, would handle:
    - Fixed extragalactic/stellar sources (static galactic coordinates)
    - Solar-system bodies (ephemeris updates per timestep)
    - Point source interpolation into beam
    """
    pass
