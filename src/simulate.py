"""
JAX-based forward model for global 21cm radiometry.

Provides ForwardModel: composed-object simulator with coefficient-based
beam and sky (via spectral basis decomposition) and optional terrain.

Architecture:
- Model objects (Beam, Sky, Observer, Terrain) are immutable numpy-based descriptors
- JAX conversion happens at the ForwardModel boundary
- Simulates antenna temperature given basis coefficients

Performance notes:
- precompute_geometry() computes beam-pixel indices and bilinear weights once per
  time step (O(npix_sky) trig per step).  simulate() reuses these, eliminating all
  transcendental arithmetic from the JIT hot path — only vectorised gather+sum
  remains, giving a ~50× speedup for the steady-state kernel.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import healjax
import healpy

from .healpix import float_dtype
from .const import DTYPE_R_NPY, DTYPE_R_JAX
from .beam import Beam
from .sky import Sky
from .observer import Observer
from .terrain import Terrain


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

    def _ensure_jax_arrays(self):
        """Convert and cache basis matrices as JAX arrays; build JIT-compiled sim kernel."""
        if self._beam_basis_A_jax is None:
            self._beam_basis_A_jax = jnp.asarray(self.beam.basis.A, dtype=DTYPE_R_JAX)
        if self._sky_basis_A_jax is None:
            self._sky_basis_A_jax = jnp.asarray(self.sky.basis.A, dtype=DTYPE_R_JAX)
        if not hasattr(self, '_sim_jit'):
            self._sim_jit = self._build_sim_fn()

    def _build_sim_fn(self):
        """Build and JIT-compile the inner simulation kernel.

        Returns a function ``sim(sky_coeffs, beam_coeffs, masks, px_all, wgts_all, T_gnd)``
        that is fully JAX-traceable.  Beam-pixel indices (px_all) and bilinear
        interpolation weights (wgts_all) are passed as precomputed arrays from
        precompute_geometry(), eliminating all transcendental arithmetic from the
        XLA hot path.
        """
        A_sky = self._sky_basis_A_jax        # (nfreq, nmodes_sky)
        A_beam = self._beam_basis_A_jax      # (nfreq, nmodes_beam)
        npix_beam = self.beam.npix           # static int
        npix_sky = self.sky.npix             # static int
        scale = float(npix_sky) / float(npix_beam)

        @jax.jit
        def _sim(sky_coeffs, beam_coeffs, masks, px_all, wgts_all, T_gnd):
            """
            sky_coeffs  : (npix_sky, nmodes_sky)
            beam_coeffs : (n_dipoles, npix_beam, nmodes_beam)
            masks       : (ntimes, npix_sky)           float32  1=visible 0=blocked
            px_all      : (ntimes, 4, npix_sky)        int32    beam pixel indices
            wgts_all    : (ntimes, 4, npix_sky)        float32  bilinear weights
            T_gnd       : scalar [K]
            Returns     : (ntimes, n_dipoles, nfreq)
            """
            sky_recon = sky_coeffs @ A_sky.T          # (npix_sky, nfreq)
            beam_recon_all = beam_coeffs @ A_beam.T   # (n_dipoles, npix_beam, nfreq)
            # Denominator (beam solid angle) is rotation-invariant: compute once
            den_all = jnp.sum(beam_recon_all, axis=1) * scale  # (n_dipoles, nfreq)

            def one_time(_, args):
                mask, px, wgts = args   # (npix_sky,), (4,npix_sky), (4,npix_sky)
                sky_masked = sky_recon * mask[:, None] + T_gnd * (1.0 - mask[:, None])

                def one_dipole(beam_recon_d, den_d):  # (npix_beam, nfreq), (nfreq,)
                    # Gather beam at sky-pixel directions via precomputed weights
                    beam_at_sky = jnp.sum(
                        beam_recon_d[px] * wgts[:, :, None], axis=0
                    )                                  # (npix_sky, nfreq)
                    num = jnp.sum(beam_at_sky * sky_masked, axis=0)  # (nfreq,)
                    return num / den_d

                return None, jax.vmap(one_dipole)(beam_recon_all, den_all)

            _, antenna_temp = jax.lax.scan(one_time, None, (masks, px_all, wgts_all))
            return antenna_temp  # (ntimes, n_dipoles, nfreq)

        return _sim

    def precompute_geometry(self, times):
        """
        Precompute rotation matrices, terrain masks, and beam interpolation
        weights for a list of observation times.

        The expensive coordinate-conversion and bilinear-interpolation-weight
        computation (O(npix_sky) trig per time step) is done here once rather
        than inside the JIT-compiled simulate() kernel, which only needs
        vectorised gather+sum operations.

        Parameters
        ----------
        times : list of Time or array-like
            Observation epochs.

        Returns
        -------
        geom : dict
            Cached geometry:
            - 'rot_gal2top': list of (3, 3) rotation matrices
            - 'crds_top': list of (3, npix_sky) topocentric coordinates
            - 'masks': list of (npix_sky,) float32 visibility masks
            - 'masks_jax': (ntimes, npix_sky) stacked JAX array
            - 'px_jax': (ntimes, 4, npix_sky) int32 — beam pixel indices
            - 'wgts_jax': (ntimes, 4, npix_sky) float32 — bilinear weights
        """
        from astropy.time import Time

        times = [Time(t) if not isinstance(t, Time) else t for t in times]
        geom = {'rot_gal2top': [], 'crds_top': [], 'masks': []}
        ntimes = len(times)
        npix_sky = self.sky.npix

        # Batch-compute rotation matrices when the observer supports it.
        # EarthSurface uses a vectorised astropy call (61× faster than looping).
        # Other observers fall back to the default per-step loop.
        if hasattr(self.observer, 'rot_gal2top_stack'):
            R_all = self.observer.rot_gal2top_stack(times)   # (ntimes, 3, 3)
        else:
            R_all = None

        for i, t in enumerate(times):
            self.observer.set_time(t)

            R = R_all[i] if R_all is not None else self.observer.rot_gal2top().astype(np.float32)
            geom['rot_gal2top'].append(R)

            crds_top = R @ self._crds_gal
            geom['crds_top'].append(crds_top)

            # For observers with rot_gal2top_stack (surface types), "above
            # horizon" is crds_top[2] > 0; skip the redundant rot_gal2top()
            # call that observer.above_horizon() would make internally.
            # LunarOrbit (no rot_gal2top_stack) still delegates to above_horizon.
            if R_all is not None:
                geo_mask = (crds_top[2] > 0).astype(np.float32)
                if self.terrain is not None:
                    terrain_mask = self.terrain.mask(crds_top).astype(np.float32)
                    geom['masks'].append(geo_mask * terrain_mask)
                else:
                    geom['masks'].append(geo_mask)
            else:
                geom['masks'].append(self._compute_mask(crds_top))

        # Vectorise beam-interpolation weight computation over all times at once.
        # healpy's C implementation processes ntimes*npix_sky pixels in a single
        # call, ~600× faster than looping with per-step JAX dispatch overhead.
        # crds_top is (3, npix_sky); stack → (ntimes, 3, npix_sky), transpose
        # → (ntimes, npix_sky, 3), reshape → (ntimes*npix_sky, 3) so each row
        # is one [x,y,z] direction vector.
        crds_flat = (np.stack(geom['crds_top'])   # (ntimes, 3, npix_sky)
                       .transpose(0, 2, 1)         # (ntimes, npix_sky, 3)
                       .reshape(ntimes * npix_sky, 3))
        th_all, ph_all = healpy.vec2ang(crds_flat)        # (ntimes*npix_sky,) each
        px_flat, wgts_flat = healpy.get_interp_weights(
            self.beam.nside, th_all, ph_all, nest=False
        )  # (4, ntimes*npix_sky) each
        # Reshape to (ntimes, 4, npix_sky)
        px_all   = px_flat.reshape(4, ntimes, npix_sky).transpose(1, 0, 2).astype(np.int32)
        wgts_all = wgts_flat.reshape(4, ntimes, npix_sky).transpose(1, 0, 2).astype(np.float32)

        geom['masks_jax'] = jnp.stack(
            [jnp.asarray(m, dtype=DTYPE_R_JAX) for m in geom['masks']]
        )  # (ntimes, npix_sky)
        geom['px_jax']   = jnp.asarray(px_all,   dtype=jnp.int32)    # (ntimes, 4, npix_sky)
        geom['wgts_jax'] = jnp.asarray(wgts_all, dtype=DTYPE_R_JAX)  # (ntimes, 4, npix_sky)
        return geom

    def _compute_mask(self, crds_top):
        """
        Compute visibility mask (1.0=visible, 0.0=blocked) at topocentric coordinates.

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix_sky)

        Returns
        -------
        mask : ndarray, shape (npix_sky,), float32
        """
        geo_mask = self.observer.above_horizon(self.sky.nside).astype(np.float32)

        if self.terrain is not None:
            terrain_mask = self.terrain.mask(crds_top).astype(np.float32)
            return geo_mask * terrain_mask
        return geo_mask

    def simulate(self, sky_coeffs, beam_coeffs, times=None, geom=None,
                 T_gnd=300.0):
        """
        Simulate antenna temperature given basis coefficients.

        Either times or geom must be provided.

        Parameters
        ----------
        sky_coeffs : ndarray, shape (npix_sky, nmodes_sky)
        beam_coeffs : ndarray, shape (n_dipoles, npix_beam, nmodes_beam)
        times : list of Time, optional
            Ignored if geom is provided.
        geom : dict, optional
            Pre-computed geometry from precompute_geometry().
        T_gnd : float
            Ground temperature [K] for blocked pixels.

        Returns
        -------
        antenna_temp : jnp.ndarray, shape (ntimes, n_dipoles, nfreq)
        """
        self._ensure_jax_arrays()

        if geom is None:
            if times is None:
                raise ValueError("Either times or geom must be provided")
            geom = self.precompute_geometry(times)

        sky_coeffs_jax = jnp.asarray(sky_coeffs, dtype=DTYPE_R_JAX)
        beam_coeffs_jax = jnp.asarray(beam_coeffs, dtype=DTYPE_R_JAX)

        masks_jax = geom.get('masks_jax')
        px_jax = geom.get('px_jax')
        wgts_jax = geom.get('wgts_jax')

        if masks_jax is None:
            masks_jax = jnp.stack(
                [jnp.asarray(m, dtype=DTYPE_R_JAX) for m in geom['masks']]
            )

        if px_jax is None or wgts_jax is None:
            # Fallback for geom dicts built before this optimisation was added
            px_list, wgts_list = [], []
            for crds in geom['crds_top']:
                th, ph = healjax.vec2ang(crds[0], crds[1], crds[2])
                px, wgts = healjax.get_interp_weights(th, ph, self.beam.nside)
                px_list.append(np.array(px, dtype=np.int32))
                wgts_list.append(np.array(wgts, dtype=np.float32))
            px_jax = jnp.stack([jnp.asarray(p, dtype=jnp.int32) for p in px_list])
            wgts_jax = jnp.stack([jnp.asarray(w, dtype=DTYPE_R_JAX) for w in wgts_list])
            geom['px_jax'] = px_jax
            geom['wgts_jax'] = wgts_jax

        return self._sim_jit(
            sky_coeffs_jax, beam_coeffs_jax, masks_jax, px_jax, wgts_jax,
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
