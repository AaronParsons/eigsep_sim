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
  terrain masks/emission and rotations as JAX arrays. simulate() passes these
  plus crds_gal_jax to a JIT-compiled scan kernel that performs gal→top and
  top→body rotations, healjax interpolation, and sky integration entirely
  inside XLA — fusing all coordinate arithmetic into one pass.
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
        Terrain model for explicit sky blocking. If None, no surface horizon
        is applied; lunar orbit occultation is handled by the observer.
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
        """Return a bool (npix_sky,) mask: True for pixels ever visible.

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
        if rots is None and times is None:
            raise ValueError("Either rots or times must be provided")

        if self.terrain is None and not getattr(self.observer, "occludes_sky", False):
            return np.ones(self.sky.npix, dtype=bool)

        if rots is not None:
            R_arr = np.stack([np.asarray(r, dtype=np.float32) for r in rots])
            ntimes = R_arr.shape[0]
            crds_top_arr = R_arr @ self._crds_gal      # (ntimes, 3, npix_sky)
            masks = np.ones((ntimes, self.sky.npix), dtype=bool)
            if getattr(self.observer, "occludes_sky", False):
                masks &= self.observer.above_horizon(self.sky.nside)[None, :]
            if self.terrain is not None:
                masks &= np.stack([
                    self.terrain.mask(crds_top_arr[i])
                    for i in range(ntimes)
                ])
            return np.any(masks, axis=0)
        elif times is not None:
            from astropy.time import Time
            times = [Time(t) if not isinstance(t, Time) else t for t in times]
            mask = np.zeros(self.sky.npix, dtype=bool)
            if hasattr(self.observer, 'rot_gal2top_stack'):
                R_all = self.observer.rot_gal2top_stack(times)
            else:
                R_all = None
            if (getattr(self.observer, "occludes_sky", False)
                    and hasattr(self.observer, "above_horizon_stack")):
                obs_mask_all = self.observer.above_horizon_stack(
                    times, self.sky.nside
                )
            else:
                obs_mask_all = None
            for i, t in enumerate(times):
                if R_all is None or (
                    getattr(self.observer, "occludes_sky", False)
                    and obs_mask_all is None
                ):
                    self.observer.set_time(t)
                if obs_mask_all is not None:
                    step_mask = obs_mask_all[i].copy()
                elif getattr(self.observer, "occludes_sky", False):
                    step_mask = self.observer.above_horizon(self.sky.nside)
                else:
                    step_mask = np.ones(self.sky.npix, dtype=bool)
                if self.terrain is not None:
                    R = (
                        R_all[i]
                        if R_all is not None
                        else self.observer.rot_gal2top().astype(np.float32)
                    )
                    step_mask &= self.terrain.mask(R @ self._crds_gal)
                mask |= step_mask
            self.observer.set_time(times[-1])
            return mask

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

        Returns a function ``sim(sky_coeffs, beam_coeffs, terrain_masks,
        terrain_emissions, default_emission_masks, unresolved_emission,
        unresolved_default_emission, rots, body_rots, T_gnd, tx_crds_all,
        crds_gal)`` that is fully JAX-traceable.
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
        def _sim(sky_coeffs, beam_coeffs, terrain_masks, terrain_emissions,
                 default_emission_masks, unresolved_emission,
                 unresolved_default_emission, rots, body_rots, T_gnd,
                 tx_crds_all, crds_gal):
            """
            sky_coeffs    : (npix_vis, nmodes_sky)
            beam_coeffs   : (n_dipoles, npix_beam, nmodes_beam)
            terrain_masks : (ntimes, npix_vis)        float32  visibility factor (1=sky)
            terrain_emissions : (ntimes, npix_vis, nfreq) float32  blocked brightness [K]
            default_emission_masks : (ntimes, npix_vis) float32  T_gnd fallback factor
            unresolved_emission : (nfreq,) float32  omitted-pixel brightness
            unresolved_default_emission : (nfreq,) float32  omitted-pixel T_gnd factor
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
                terrain_mask, terrain_emit, default_emit_mask, R, br, tx_c = args
                # R: (3,3)  br: (3,3)  terrain_mask: (npix_vis,)  tx_c: (n_sources,3)
                crds_top = R @ crds_gal                              # (3, npix_vis)
                mask = terrain_mask                                # (npix_vis,)
                crds_body = br @ crds_top                           # (3, npix_vis)

                th, ph = healjax.vec2ang(crds_body[0], crds_body[1], crds_body[2])
                px, wgts = healjax.get_interp_weights(th, ph, nside_beam)
                # px: (4, npix_vis)   wgts: (4, npix_vis)

                # TX interpolation locations are independent of dipole.
                th_tx, ph_tx = healjax.vec2ang(tx_c[:, 0], tx_c[:, 1], tx_c[:, 2])
                tx_px, tx_wgts = healjax.get_interp_weights(
                    th_tx, ph_tx, nside_beam
                )
                # tx_px: (4, n_sources)   tx_wgts: (4, n_sources)

                def one_dipole(beam_recon_d, den_d):  # (npix_beam, nfreq), (nfreq,)
                    # Accumulate 4 bilinear neighbors without materialising (4,npix,nfreq)
                    beam_at_sky = jax.lax.fori_loop(
                        0, 4,
                        lambda k, acc: acc + beam_recon_d[px[k]] * wgts[k, :, None],
                        jnp.zeros_like(sky_recon),
                    )                                  # (npix_vis, nfreq)
                    sky_num = jnp.sum(
                        beam_at_sky * sky_recon * mask[:, None], axis=0
                    )
                    terrain_num = jnp.sum(beam_at_sky * terrain_emit, axis=0)
                    sampled_weight = jnp.sum(beam_at_sky, axis=0)
                    default_weight = jnp.sum(
                        beam_at_sky * default_emit_mask[:, None], axis=0
                    )
                    # Pixels omitted by sky_mask use the terrain-provided
                    # unresolved spectrum. Sampled observer-only occultation
                    # still uses the scalar T_gnd fallback.
                    unresolved_weight = den_d - sampled_weight
                    num = (
                        sky_num
                        + terrain_num
                        + T_gnd * default_weight
                        + unresolved_weight * (
                            unresolved_emission
                            + T_gnd * unresolved_default_emission
                        )
                    )

                    # TX: interpolate beam at each source direction
                    beam_at_tx = jax.lax.fori_loop(
                        0, 4,
                        lambda k, acc: acc + beam_recon_d[tx_px[k]] * tx_wgts[k, :, None],
                        jnp.zeros_like(tx_T_jax),
                    )                                  # (n_sources, nfreq)
                    num = num + jnp.sum(beam_at_tx * tx_T_jax, axis=0)  # (nfreq,)
                    return num / den_d

                return None, jax.vmap(one_dipole)(beam_recon_all, den_all)

            _, antenna_temp = jax.lax.scan(
                one_time, None,
                (
                    terrain_masks,
                    terrain_emissions,
                    default_emission_masks,
                    rots,
                    body_rots,
                    tx_crds_all,
                ),
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
            - 'body_rots_jax': (ntimes, 3, 3) float32 — top→body rotations
            - 'terrain_masks_jax': (ntimes, npix_vis) float32 — visibility factor (1=sky)
            - 'terrain_emissions_jax': (ntimes, npix_vis, nfreq) float32 — blocked brightness
            - 'default_emission_masks_jax': (ntimes, npix_vis) float32 — T_gnd fallback factor
            - 'unresolved_emission_jax': (nfreq,) float32 — omitted-pixel brightness
            - 'unresolved_default_emission_jax': (nfreq,) float32 — omitted-pixel T_gnd factor
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
        if self.terrain is not None:
            unresolved_emission = self.terrain.unresolved_emission(
                self.beam.freqs_hz
            )
            unresolved_default_emission = np.zeros(
                len(self.beam.freqs_hz), dtype=np.float32
            )
            if unresolved_emission is None:
                if sky_mask is not None and npix_vis != self.sky.npix:
                    raise ValueError(
                        "sky_mask with terrain requires terrain.unresolved_emission() "
                        "or full geometry for exact omitted-pixel emission"
                    )
                unresolved_emission = np.zeros(
                    len(self.beam.freqs_hz), dtype=np.float32
                )
            else:
                unresolved_emission = np.asarray(
                    unresolved_emission, dtype=np.float32
                )
        else:
            unresolved_emission = np.zeros(len(self.beam.freqs_hz), dtype=np.float32)
            unresolved_default_emission = np.ones(
                len(self.beam.freqs_hz), dtype=np.float32
            )

        geom = {}

        if rots is not None:
            # Rots path: skip observer ephemeris and gal→top matmul.
            # The kernel performs R @ crds_gal per step inside jax.lax.scan.
            R_arr = np.stack([np.asarray(r, dtype=np.float32) for r in rots])
            ntimes = R_arr.shape[0]
            geom['rot_gal2top'] = R_arr

            crds_top_arr = R_arr @ crds_gal      # (ntimes, 3, npix_vis)
            if getattr(self.observer, "occludes_sky", False):
                obs_mask_full = self.observer.above_horizon(
                    self.sky.nside
                ).astype(np.float32)
                if sky_indices is not None:
                    obs_mask = obs_mask_full[sky_indices]
                else:
                    obs_mask = obs_mask_full
                obs_masks = np.broadcast_to(
                    obs_mask, (ntimes, npix_vis)
                ).astype(np.float32)
            else:
                obs_masks = np.ones((ntimes, npix_vis), dtype=np.float32)

            if self.terrain is not None:
                terrain_masks_list = []
                terrain_emissions_list = []
                default_emission_masks_list = []
                for i in range(ntimes):
                    t_mask = self.terrain.mask(crds_top_arr[i]).astype(np.float32)
                    terrain_masks_list.append(
                        (obs_masks[i] * t_mask).astype(np.float32)
                    )
                    t_emit = self.terrain.emission(
                        crds_top_arr[i], self.beam.freqs_hz
                    ).astype(np.float32)
                    terrain_emissions_list.append(t_emit)
                    default_emission_masks_list.append(1.0 - obs_masks[i])
                terrain_masks = np.stack(terrain_masks_list)
                terrain_emissions = np.stack(terrain_emissions_list)
                default_emission_masks = np.stack(default_emission_masks_list)
            else:
                terrain_masks = obs_masks
                terrain_emissions = np.zeros(
                    (ntimes, npix_vis, len(self.beam.freqs_hz)), dtype=np.float32
                )
                default_emission_masks = 1.0 - obs_masks

        elif times is not None:
            times = [Time(t) if not isinstance(t, Time) else t for t in times]
            ntimes = len(times)
            rot_list, crds_list = [], []
            masks_list, emissions_list, default_emission_masks_list = [], [], []

            # Batch-compute rotation matrices when the observer supports it.
            # EarthSurface uses a vectorised astropy call (61× faster than looping).
            if hasattr(self.observer, 'rot_gal2top_stack'):
                R_all = self.observer.rot_gal2top_stack(times)   # (ntimes, 3, 3)
            else:
                R_all = None

            if (getattr(self.observer, "occludes_sky", False)
                    and hasattr(self.observer, "above_horizon_stack")):
                obs_mask_all = self.observer.above_horizon_stack(
                    times, self.sky.nside
                ).astype(np.float32)
                if sky_indices is not None:
                    obs_mask_all = obs_mask_all[:, sky_indices]
            else:
                obs_mask_all = None

            needs_step_time = R_all is None or (
                getattr(self.observer, "occludes_sky", False)
                and obs_mask_all is None
            )

            for i, t in enumerate(times):
                if needs_step_time:
                    self.observer.set_time(t)
                R = (
                    R_all[i]
                    if R_all is not None
                    else self.observer.rot_gal2top().astype(np.float32)
                )
                rot_list.append(R)
                crds_top = R @ crds_gal              # (3, npix_vis)
                crds_list.append(crds_top)

                if obs_mask_all is not None:
                    obs_mask = obs_mask_all[i]
                elif getattr(self.observer, "occludes_sky", False):
                    obs_mask_full = self.observer.above_horizon(
                        self.sky.nside
                    ).astype(np.float32)
                    if sky_indices is not None:
                        obs_mask = obs_mask_full[sky_indices]
                    else:
                        obs_mask = obs_mask_full
                else:
                    obs_mask = np.ones(npix_vis, dtype=np.float32)
                if self.terrain is not None:
                    t_mask = self.terrain.mask(crds_top).astype(np.float32)
                    masks_list.append(obs_mask * t_mask)
                    t_emit = self.terrain.emission(
                        crds_top, self.beam.freqs_hz
                    ).astype(np.float32)
                    emissions_list.append(t_emit)
                    default_emission_masks_list.append(1.0 - obs_mask)
                else:
                    masks_list.append(obs_mask)
                    emissions_list.append(
                        np.zeros(
                            (npix_vis, len(self.beam.freqs_hz)), dtype=np.float32
                        )
                    )
                    default_emission_masks_list.append(1.0 - obs_mask)

            R_arr = np.stack(rot_list)
            geom.update({
                'rot_gal2top': R_arr,
                'crds_top':    np.stack(crds_list),
                'masks':       np.stack(masks_list).astype(np.float32),
            })
            terrain_masks = np.stack(masks_list).astype(np.float32)
            terrain_emissions = np.stack(emissions_list).astype(np.float32)
            default_emission_masks = np.stack(
                default_emission_masks_list
            ).astype(np.float32)
            self.observer.set_time(times[-1])
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

        geom['rots_jax'] = jnp.asarray(R_arr, dtype=DTYPE_R_JAX)
        geom['body_rots_jax'] = jnp.asarray(body_rots_arr, dtype=DTYPE_R_JAX)
        geom['terrain_masks_jax'] = jnp.asarray(
            terrain_masks, dtype=DTYPE_R_JAX
        )
        geom['terrain_emissions_jax'] = jnp.asarray(
            terrain_emissions, dtype=DTYPE_R_JAX
        )
        geom['default_emission_masks_jax'] = jnp.asarray(
            default_emission_masks, dtype=DTYPE_R_JAX
        )
        geom['unresolved_emission_jax'] = jnp.asarray(
            unresolved_emission, dtype=DTYPE_R_JAX
        )
        geom['unresolved_default_emission_jax'] = jnp.asarray(
            unresolved_default_emission, dtype=DTYPE_R_JAX
        )
        geom['crds_gal_jax'] = jnp.asarray(crds_gal, dtype=DTYPE_R_JAX)
        geom['tx_crds_jax'] = jnp.asarray(tx_body, dtype=DTYPE_R_JAX)
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
        if getattr(self.observer, "occludes_sky", False):
            obs_mask = self.observer.above_horizon(self.sky.nside).astype(np.float32)
        else:
            obs_mask = np.ones(self.sky.npix, dtype=np.float32)

        if self.terrain is not None:
            terrain_mask = self.terrain.mask(crds_top).astype(np.float32)
            return obs_mask * terrain_mask
        return obs_mask

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
            geom['terrain_emissions_jax'],
            geom['default_emission_masks_jax'],
            geom['unresolved_emission_jax'],
            geom['unresolved_default_emission_jax'],
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
