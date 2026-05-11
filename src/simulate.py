"""
JAX-based forward model for global 21cm radiometry.

Provides ForwardModel: composed-object simulator with coefficient-based
beam and sky (via spectral basis decomposition) and optional terrain.

Architecture:
- Model objects (Beam, Sky, Observer, Terrain) are immutable numpy-based descriptors
- JAX conversion happens at the ForwardModel boundary
- Simulates antenna temperature given basis coefficients

Performance notes:
- precompute_geometry() stores rotation matrices (rots_jax, body_rots_jax) and
  terrain masks (terrain_masks_jax) as small JAX arrays (~100 KB for 1296 steps).
  simulate() passes these plus crds_gal_jax to a JIT-compiled scan kernel that
  performs gal→top and top→body rotations, healjax interpolation, and sky
  integration entirely inside XLA — fusing all coordinate arithmetic into one pass.
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
                 terrain: Terrain | None = None, transmitters=None):
        self.observer = observer
        self.beam = beam
        self.sky = sky
        self.terrain = terrain

        # Transmitters: list of (direction_topo, freqs_tx_hz, power_K) tuples.
        # direction_topo : (3,) unit vector in topocentric frame
        # freqs_tx_hz    : (n_tx_freqs,) specific emitting frequencies [Hz]
        # power_K        : scalar or (n_tx_freqs,) equivalent temperature [K]
        #
        # Store as:
        #   _tx_dirs       : (n_sources, 3) topocentric unit vectors
        #   _tx_T_internal : (n_sources, nfreq) full-band power, scaled by
        #                    sky.npix/beam.npix so the kernel denominator
        #                    (sum over beam pixels) cancels correctly.
        nfreq = len(beam.freqs_hz)
        scale = float(sky.npix) / float(beam.npix)
        if transmitters:
            dirs, T_internals = [], []
            for direction, freqs_tx, power_K in transmitters:
                d = np.asarray(direction, dtype=np.float32)
                d = d / np.linalg.norm(d)
                dirs.append(d)
                T_full = np.zeros(nfreq, dtype=np.float32)
                pwr = np.broadcast_to(np.asarray(power_K, dtype=np.float32),
                                      np.asarray(freqs_tx).shape)
                # Match each transmitter frequency to the nearest simulation bin
                for f_tx, p in zip(freqs_tx, pwr):
                    idx = int(np.argmin(np.abs(beam.freqs_hz - f_tx)))
                    T_full[idx] += float(p)
                T_internals.append(T_full * scale)
            self._tx_dirs = np.stack(dirs)                   # (n_sources, 3)
            self._tx_T_internal = np.stack(T_internals)      # (n_sources, nfreq)
        else:
            self._tx_dirs = np.zeros((0, 3), dtype=np.float32)
            self._tx_T_internal = np.zeros((0, nfreq), dtype=np.float32)

        # Cache static galactic coordinates
        self._crds_gal = np.array(
            healpy.pix2vec(sky.nside, np.arange(sky.npix)), dtype=DTYPE_R_NPY
        )  # (3, npix_sky)

        # Static JAX arrays (created on demand)
        self._beam_basis_A_jax = None
        self._sky_basis_A_jax = None

    def build_sky_mask(self, rots=None, times=None):
        """Return a bool (npix_sky,) mask: True for pixels ever above the horizon.

        Pass the result as ``sky_mask`` to ``precompute_geometry`` to exclude
        pixels that never contribute, reducing coordinate arrays and kernel work
        proportionally.  Gains are largest for long integrations at a fixed
        sky orientation; for a full az/alt scan nearly all pixels are visible.

        Parameters
        ----------
        rots : list of (3, 3) ndarray, optional
            Pre-computed galactic-to-topocentric rotation matrices.
        times : list of Time, optional
            Observation epochs (queries the observer ephemeris).
        """
        if rots is not None:
            R_arr = np.stack([np.asarray(r, dtype=np.float32) for r in rots])
            crds_top_arr = R_arr @ self._crds_gal          # (ntimes, 3, npix_sky)
            return np.any(crds_top_arr[:, 2, :] > 0, axis=0)  # (npix_sky,)
        elif times is not None:
            from astropy.time import Time
            mask = np.zeros(self.sky.npix, dtype=bool)
            for t in times:
                self.observer.set_time(Time(t) if not isinstance(t, Time) else t)
                R = self.observer.rot_gal2top().astype(np.float32)
                mask |= (R @ self._crds_gal)[2] > 0
            return mask
        else:
            raise ValueError("Either rots or times must be provided")

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

        Returns a function ``sim(sky_coeffs, beam_coeffs, terrain_masks, rots,
        body_rots, T_gnd, tx_crds_all, crds_gal)`` that is fully JAX-traceable.
        Rotation matmuls (gal→top, top→body) and healjax interpolation run inside
        jax.lax.scan so that coordinate arithmetic, beam gather, and sky integration
        are fused by XLA.

        Transmitter temperatures (tx_T_jax, (n_sources, nfreq)) are closed over as
        compile-time constants; tx_crds_all carries per-step body-frame directions.
        When n_sources=0 the arrays are empty and the TX sum contributes zero.
        """
        A_sky = self._sky_basis_A_jax        # (nfreq, nmodes_sky)
        A_beam = self._beam_basis_A_jax      # (nfreq, nmodes_beam)
        npix_beam = self.beam.npix           # static int
        npix_sky = self.sky.npix             # static int
        nside_beam = self.beam.nside         # static int — required by healjax
        scale = float(npix_sky) / float(npix_beam)
        tx_T_jax = jnp.asarray(self._tx_T_internal, dtype=DTYPE_R_JAX)  # (n_src, nfreq)

        @jax.jit
        def _sim(sky_coeffs, beam_coeffs, terrain_masks, rots, body_rots,
                 T_gnd, tx_crds_all, crds_gal):
            """
            sky_coeffs    : (npix_vis, nmodes_sky)
            beam_coeffs   : (n_dipoles, npix_beam, nmodes_beam)
            terrain_masks : (ntimes, npix_vis)        float32  terrain factor (1=clear)
            rots          : (ntimes, 3, 3)            float32  gal→top rotation matrices
            body_rots     : (ntimes, 3, 3)            float32  top→body rotation matrices
            T_gnd         : scalar [K]
            tx_crds_all   : (ntimes, n_sources, 3)   float32  TX body-frame directions
            crds_gal      : (3, npix_vis)             float32  galactic unit vectors
            Returns       : (ntimes, n_dipoles, nfreq)
            """
            sky_recon = sky_coeffs @ A_sky.T          # (npix_vis, nfreq)
            beam_recon_all = beam_coeffs @ A_beam.T   # (n_dipoles, npix_beam, nfreq)
            den_all = jnp.sum(beam_recon_all, axis=1) * scale  # (n_dipoles, nfreq)

            def one_time(_, args):
                terrain_mask, R, br, tx_c = args
                # R: (3,3)  br: (3,3)  terrain_mask: (npix_vis,)  tx_c: (n_sources,3)
                crds_top = R @ crds_gal                              # (3, npix_vis)
                geo_mask = (crds_top[2] > 0).astype(DTYPE_R_JAX)   # (npix_vis,)
                mask = geo_mask * terrain_mask                       # (npix_vis,)
                crds_body = br @ crds_top                           # (3, npix_vis)

                th, ph = healjax.vec2ang(crds_body[0], crds_body[1], crds_body[2])
                px, wgts = healjax.get_interp_weights(th, ph, nside_beam)
                # px: (4, npix_vis)   wgts: (4, npix_vis)

                sky_masked = sky_recon * mask[:, None] + T_gnd * (1.0 - mask[:, None])

                def one_dipole(beam_recon_d, den_d):  # (npix_beam, nfreq), (nfreq,)
                    # Accumulate 4 bilinear neighbors without materialising (4,npix,nfreq)
                    beam_at_sky = jax.lax.fori_loop(
                        0, 4,
                        lambda k, acc: acc + beam_recon_d[px[k]] * wgts[k, :, None],
                        jnp.zeros_like(sky_recon),
                    )                                  # (npix_vis, nfreq)
                    num = jnp.sum(beam_at_sky * sky_masked, axis=0)  # (nfreq,)

                    # TX: interpolate beam at each source direction
                    th_tx, ph_tx = healjax.vec2ang(tx_c[:, 0], tx_c[:, 1], tx_c[:, 2])
                    tx_px, tx_wgts = healjax.get_interp_weights(th_tx, ph_tx, nside_beam)
                    # tx_px: (4, n_sources)   tx_wgts: (4, n_sources)
                    beam_at_tx = jax.lax.fori_loop(
                        0, 4,
                        lambda k, acc: acc + beam_recon_d[tx_px[k]] * tx_wgts[k, :, None],
                        jnp.zeros_like(tx_T_jax),
                    )                                  # (n_sources, nfreq)
                    num = num + jnp.sum(beam_at_tx * tx_T_jax, axis=0)  # (nfreq,)
                    return num / den_d

                return None, jax.vmap(one_dipole)(beam_recon_all, den_all)

            _, antenna_temp = jax.lax.scan(
                one_time, None, (terrain_masks, rots, body_rots, tx_crds_all)
            )
            return antenna_temp  # (ntimes, n_dipoles, nfreq)

        return _sim

    def precompute_geometry(self, times=None, rots=None, body_rots=None,
                            sky_mask=None):
        """
        Precompute rotation matrices and terrain masks for a list of observation
        times or pre-computed rotation matrices.

        Parameters
        ----------
        times : list of Time or array-like, optional
            Observation epochs.  Mutually exclusive with ``rots``.
        rots : list of (3, 3) ndarray, optional
            Pre-computed galactic-to-topocentric rotation matrices, one per
            step.  When provided ``times`` is ignored and the observer
            ephemeris is not queried — useful for non-temporal scanning (e.g.
            az/alt sweeps at a fixed epoch computed once outside this call).
        body_rots : list of (3, 3) ndarray, optional
            Per-step topocentric-to-body rotation matrices.  When provided,
            sky coordinates are rotated from topocentric into the antenna body
            frame before beam pixel lookup.  If None, body frame coincides
            with the topocentric frame (the default).
        sky_mask : ndarray of bool, shape (npix_sky,), optional
            Static pixel-reduction mask from ``build_sky_mask()``.  Only
            masked-in pixels are included in the geometry arrays, reducing
            coordinate and kernel sizes proportionally.  ``simulate()``
            automatically gathers the corresponding sky coefficients via the
            ``sky_indices_jax`` entry stored in the returned geom dict.

        Returns
        -------
        geom : dict
            Cached geometry (JAX arrays ready for the kernel):
            - 'rots_jax': (ntimes, 3, 3) float32 — gal→top rotation matrices
            - 'body_rots_jax': (ntimes, 3, 3) float32 — top→body rotations (identity if body_rots=None)
            - 'terrain_masks_jax': (ntimes, npix_vis) float32 — terrain factor (1=clear, computed
              in-kernel from crds_top[2]>0); all ones when no terrain
            - 'crds_gal_jax': (3, npix_vis) float32 — galactic unit vectors (filtered by sky_mask)
            - 'tx_crds_jax': (ntimes, n_sources, 3) float32 — TX body-frame directions
            - 'sky_indices_jax': (npix_vis,) int32 — only present when sky_mask given
            Times path also retains: 'rot_gal2top', 'crds_top', 'masks' (numpy arrays)
        """
        from astropy.time import Time

        # Optional static pixel reduction: restrict to ever-visible pixels.
        if sky_mask is not None:
            sky_mask_np = np.asarray(sky_mask, dtype=bool)
            sky_indices = np.where(sky_mask_np)[0].astype(np.int32)
            crds_gal = self._crds_gal[:, sky_mask_np]  # (3, npix_vis)
        else:
            sky_indices = None
            crds_gal = self._crds_gal                  # (3, npix_sky)

        npix_vis = crds_gal.shape[1]
        geom = {}

        if rots is not None:
            # Rots path: skip observer ephemeris and gal→top matmul.
            # The kernel performs R @ crds_gal per step inside jax.lax.scan.
            R_arr = np.stack([np.asarray(r, dtype=np.float32) for r in rots])
            ntimes = R_arr.shape[0]
            geom['rot_gal2top'] = R_arr

            if self.terrain is not None:
                # Compute crds_top only to evaluate the terrain mask.
                crds_top_arr = R_arr @ crds_gal      # (ntimes, 3, npix_vis)
                terrain_masks = np.stack([
                    self.terrain.mask(crds_top_arr[i]).astype(np.float32)
                    for i in range(ntimes)
                ])                                   # (ntimes, npix_vis)
            else:
                terrain_masks = np.ones((ntimes, npix_vis), dtype=np.float32)

        elif times is not None:
            times = [Time(t) if not isinstance(t, Time) else t for t in times]
            ntimes = len(times)
            rot_list, crds_list, masks_list, terrain_list = [], [], [], []

            # Batch-compute rotation matrices when the observer supports it.
            # EarthSurface uses a vectorised astropy call (61× faster than looping).
            if hasattr(self.observer, 'rot_gal2top_stack'):
                R_all = self.observer.rot_gal2top_stack(times)   # (ntimes, 3, 3)
            else:
                R_all = None

            for i, t in enumerate(times):
                self.observer.set_time(t)
                R = R_all[i] if R_all is not None else self.observer.rot_gal2top().astype(np.float32)
                rot_list.append(R)
                crds_top = R @ crds_gal              # (3, npix_vis)
                crds_list.append(crds_top)

                geo_mask = (crds_top[2] > 0).astype(np.float32)
                if self.terrain is not None:
                    t_mask = self.terrain.mask(crds_top).astype(np.float32)
                    masks_list.append(geo_mask * t_mask)
                    terrain_list.append(t_mask)
                else:
                    masks_list.append(geo_mask)
                    terrain_list.append(np.ones(npix_vis, dtype=np.float32))

            R_arr = np.stack(rot_list)
            geom.update({
                'rot_gal2top': R_arr,
                'crds_top':    np.stack(crds_list),
                'masks':       np.stack(masks_list).astype(np.float32),
            })
            terrain_masks = np.stack(terrain_list).astype(np.float32)
        else:
            raise ValueError("Either times or rots must be provided")

        # Build body_rots array: identity if not provided.
        if body_rots is not None:
            body_rots_arr = np.stack(
                [np.asarray(br, dtype=np.float32) for br in body_rots]
            )                                        # (ntimes, 3, 3)
        else:
            body_rots_arr = np.broadcast_to(
                np.eye(3, dtype=np.float32), (ntimes, 3, 3)
            ).copy()

        # Transmitters are topocentric; only body_rots applies (gal→top does NOT).
        n_sources = self._tx_dirs.shape[0]
        if n_sources > 0:
            tx_body = (body_rots_arr @ self._tx_dirs.T).transpose(0, 2, 1)
        else:
            tx_body = np.zeros((ntimes, 0, 3), dtype=np.float32)

        geom['rots_jax']          = jnp.asarray(R_arr,         dtype=DTYPE_R_JAX)  # (ntimes, 3, 3)
        geom['body_rots_jax']     = jnp.asarray(body_rots_arr, dtype=DTYPE_R_JAX)  # (ntimes, 3, 3)
        geom['terrain_masks_jax'] = jnp.asarray(terrain_masks, dtype=DTYPE_R_JAX)  # (ntimes, npix_vis)
        geom['crds_gal_jax']      = jnp.asarray(crds_gal,      dtype=DTYPE_R_JAX)  # (3, npix_vis)
        geom['tx_crds_jax']       = jnp.asarray(tx_body,       dtype=DTYPE_R_JAX)  # (ntimes, n_src, 3)
        if sky_indices is not None:
            geom['sky_indices_jax'] = jnp.asarray(sky_indices, dtype=jnp.int32)
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

        sky_indices_jax = geom.get('sky_indices_jax')
        if sky_indices_jax is not None:
            sky_coeffs_jax = sky_coeffs_jax[sky_indices_jax]  # (npix_vis, nmodes_sky)

        return self._sim_jit(
            sky_coeffs_jax,
            beam_coeffs_jax,
            geom['terrain_masks_jax'],
            geom['rots_jax'],
            geom['body_rots_jax'],
            jnp.asarray(T_gnd, dtype=DTYPE_R_JAX),
            geom['tx_crds_jax'],
            geom['crds_gal_jax'],
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
