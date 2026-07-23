"""
JAX-based forward model for global 21cm radiometry.

Provides ForwardModel: composed-object simulator with coefficient-based
beam and sky (via spectral basis decomposition) and optional terrain.

Architecture:
- Model objects (Beam, Sky, Observer, Terrain) are immutable numpy-based descriptors
- JAX conversion happens at the ForwardModel boundary
- Simulates antenna temperature given basis coefficients

Performance notes:
- precompute_geometry() stores terrain masks/emission plus body-frame HEALPix
  interpolation pixels and weights as JAX arrays. simulate() passes these to a
  JIT-compiled scan kernel so repeated optimization steps only reconstruct the
  sky and beam maps, gather cached beam samples, and integrate the sky.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
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

    def __init__(
        self,
        observer: Observer,
        beam: Beam,
        sky: Sky,
        terrain: Terrain | None = None,
        transmitters=None,
    ):
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
        #   _tx_T_internal : (n_sources, nfreq) full-band power [K].
        #                    The kernel divides by the total beam integral
        #                    (sampled_weight), so no pre-scaling is needed.
        nfreq = len(beam.freqs_hz)
        if transmitters:
            dirs, T_internals = [], []
            for direction, freqs_tx, power_K in transmitters:
                d = np.asarray(direction, dtype=np.float32)
                d = d / np.linalg.norm(d)
                dirs.append(d)
                T_full = np.zeros(nfreq, dtype=np.float32)
                pwr = np.broadcast_to(
                    np.asarray(power_K, dtype=np.float32),
                    np.asarray(freqs_tx).shape,
                )
                # Match each transmitter frequency to the nearest simulation bin
                for f_tx, p in zip(freqs_tx, pwr):
                    idx = int(np.argmin(np.abs(beam.freqs_hz - f_tx)))
                    T_full[idx] += float(p)
                T_internals.append(T_full)
            self._tx_dirs = np.stack(dirs)  # (n_sources, 3)
            self._tx_T_internal = np.stack(T_internals)  # (n_sources, nfreq)
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

        if self.terrain is None and not getattr(
            self.observer, "occludes_sky", False
        ):
            return np.ones(self.sky.npix, dtype=bool)

        if rots is not None:
            R_arr = np.stack([np.asarray(r, dtype=np.float32) for r in rots])
            ntimes = R_arr.shape[0]
            crds_top_arr = R_arr @ self._crds_gal  # (ntimes, 3, npix_sky)
            masks = np.ones((ntimes, self.sky.npix), dtype=bool)
            if getattr(self.observer, "occludes_sky", False):
                masks &= self.observer.above_horizon(self.sky.nside)[None, :]
            if self.terrain is not None:
                masks &= np.stack(
                    [self.terrain.mask(crds_top_arr[i]) for i in range(ntimes)]
                )
            return np.any(masks, axis=0)
        elif times is not None:
            from astropy.time import Time

            times = [Time(t) if not isinstance(t, Time) else t for t in times]
            mask = np.zeros(self.sky.npix, dtype=bool)
            if hasattr(self.observer, "rot_gal2top_stack"):
                R_all = self.observer.rot_gal2top_stack(times)
            else:
                R_all = None
            if getattr(self.observer, "occludes_sky", False) and hasattr(
                self.observer, "above_horizon_stack"
            ):
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

    def _observer_occultation_emission(self):
        """Return observer-provided emission for occulted rays, if any."""
        if not getattr(self.observer, "occludes_sky", False):
            return None
        if not hasattr(self.observer, "occultation_emission"):
            return None
        emission = self.observer.occultation_emission(self.beam.freqs_hz)
        if emission is None:
            return None
        return np.asarray(emission, dtype=np.float32)

    def _ensure_jax_arrays(self):
        """Convert and cache basis matrices as JAX arrays; build JIT-compiled sim kernel."""
        if self._beam_basis_A_jax is None:
            self._beam_basis_A_jax = jnp.asarray(
                self.beam.basis.A, dtype=DTYPE_R_JAX
            )
        if self._sky_basis_A_jax is None:
            self._sky_basis_A_jax = jnp.asarray(
                self.sky.basis.A, dtype=DTYPE_R_JAX
            )
        if not hasattr(self, "_sim_jit"):
            self._sim_jit = self._build_sim_fn()
        if not hasattr(self, "_adjoint_jit"):
            self._adjoint_jit = self._build_adjoint_fn()

    def _build_sim_fn(self):
        """Build and JIT-compile the inner simulation kernel.

        Returns a function ``sim(sky_coeffs, beam_coeffs, terrain_masks,
        terrain_emissions, default_emission_masks, unresolved_emission,
        unresolved_default_emission, unresolved_beam_weights, beam_px,
        beam_wgts, T_gnd, tx_px, tx_wgts)`` that is fully JAX-traceable.
        Geometry-dependent rotations and HEALPix interpolation are cached by
        ``precompute_geometry()`` so repeated optimizer evaluations only gather
        beam samples and integrate the sky.

        Transmitter temperatures (tx_T_jax, (n_sources, nfreq)) are closed over as
        compile-time constants. When n_sources=0 the arrays are empty and the TX
        sum contributes zero.
        """
        A_sky = self._sky_basis_A_jax  # (nfreq, nmodes_sky)
        A_beam = self._beam_basis_A_jax  # (nfreq, nmodes_beam)
        tx_T_jax = jnp.asarray(
            self._tx_T_internal, dtype=DTYPE_R_JAX
        )  # (n_src, nfreq)

        @jax.jit
        def _sim(
            sky_coeffs,
            beam_recon_all,
            terrain_masks,
            terrain_emissions,
            default_emission_masks,
            unresolved_emission,
            unresolved_default_emission,
            unresolved_beam_weights_all,
            beam_px_all,
            beam_wgts_all,
            T_gnd,
            T_iso,
            tx_px_all,
            tx_wgts_all,
        ):
            """
            sky_coeffs    : (npix_vis, nmodes_sky)
            beam_coeffs   : (n_dipoles, npix_beam, nmodes_beam)
            terrain_masks : (ntimes, npix_vis)        float32  visibility factor (1=sky)
            terrain_emissions : (ntimes, npix_vis, nfreq) float32  blocked brightness [K]
            default_emission_masks : (ntimes, npix_vis) float32  T_gnd fallback factor
            unresolved_emission : (nfreq,) float32  omitted-pixel brightness
            unresolved_default_emission : (nfreq,) float32  omitted-pixel T_gnd factor
            unresolved_beam_weights_all : (ntimes, npix_beam) float32 omitted-pixel beam sampling accumulator
            beam_px_all   : (ntimes, 4, npix_vis)     int32    cached beam pixels
            beam_wgts_all : (ntimes, 4, npix_vis)     float32  cached beam weights
            T_gnd         : scalar [K]
            T_iso         : (nfreq,) float32  isotropic additive sky component (e.g. T21) [K].
                            Bypasses the sky basis so its spectral shape is preserved
                            exactly.  Contributes T_iso * f_vis to the antenna temperature
                            where f_vis = sum_p(B_p*mask_p)/sum_p(B_p).
            tx_px_all     : (ntimes, 4, n_sources)    int32    cached TX pixels
            tx_wgts_all   : (ntimes, 4, n_sources)    float32  cached TX weights
            Returns       : (ntimes, n_dipoles, nfreq)
            """
            sky_recon = sky_coeffs @ A_sky.T  # (npix_vis, nfreq)
            # beam_recon_all is passed in already reconstructed
            # (n_dipoles, npix_beam, nfreq).  The coefficient reconstruction
            # beam_coeffs @ A_beam.T is done by the caller so a parametric beam
            # map (e.g. dipole_beam_maps_jax) can be substituted unchanged.

            def one_time(_, args):
                (
                    terrain_mask,
                    terrain_emit,
                    default_emit_mask,
                    unresolved_beam_weights,
                    px,
                    wgts,
                    tx_px,
                    tx_wgts,
                ) = args
                mask = terrain_mask  # (npix_vis,)

                def one_dipole(beam_recon_d):
                    # Accumulate 4 bilinear neighbors without materialising (4,npix,nfreq)
                    beam_at_sky = jax.lax.fori_loop(
                        0,
                        4,
                        lambda k, acc: acc
                        + beam_recon_d[px[k]] * wgts[k, :, None],
                        jnp.zeros_like(sky_recon),
                    )  # (npix_vis, nfreq)
                    sky_num = jnp.sum(
                        beam_at_sky * sky_recon * mask[:, None], axis=0
                    )
                    # Visible-pixel beam integral — used for T_iso injection.
                    # Kept separate from sky_num so T_iso bypasses the basis.
                    visible_weight = jnp.sum(beam_at_sky * mask[:, None], axis=0)
                    sky_num = sky_num + T_iso * visible_weight
                    terrain_num = jnp.sum(beam_at_sky * terrain_emit, axis=0)
                    sampled_weight = jnp.sum(beam_at_sky, axis=0)
                    default_weight = jnp.sum(
                        beam_at_sky * default_emit_mask[:, None], axis=0
                    )
                    # Pixels omitted by sky_mask use the terrain-provided
                    # unresolved spectrum. Sampled observer-only occultation
                    # still uses the scalar T_gnd fallback.
                    unresolved_weight = unresolved_beam_weights @ beam_recon_d
                    num = (
                        sky_num
                        + terrain_num
                        + T_gnd * default_weight
                        + unresolved_weight
                        * (
                            unresolved_emission
                            + T_gnd * unresolved_default_emission
                        )
                    )

                    # TX: interpolate beam at each source direction
                    beam_at_tx = jax.lax.fori_loop(
                        0,
                        4,
                        lambda k, acc: acc
                        + beam_recon_d[tx_px[k]] * tx_wgts[k, :, None],
                        jnp.zeros_like(tx_T_jax),
                    )  # (n_sources, nfreq)
                    num = num + jnp.sum(
                        beam_at_tx * tx_T_jax, axis=0
                    )  # (nfreq,)
                    # Normalise by total beam integral → output in Kelvin.
                    # unresolved_weight covers sky_mask-omitted pixels; zero
                    # in the full-sky case.
                    denom = sampled_weight + unresolved_weight
                    return num / denom

                return None, jax.vmap(one_dipole)(beam_recon_all)

            _, antenna_temp = jax.lax.scan(
                one_time,
                None,
                (
                    terrain_masks,
                    terrain_emissions,
                    default_emission_masks,
                    unresolved_beam_weights_all,
                    beam_px_all,
                    beam_wgts_all,
                    tx_px_all,
                    tx_wgts_all,
                ),
            )
            return antenna_temp  # (ntimes, n_dipoles, nfreq)

        return _sim

    def _build_adjoint_fn(self):
        """Build a JIT adjoint/diagonal Gauss-Newton accumulation kernel."""
        A_sky = self._sky_basis_A_jax
        A_beam = self._beam_basis_A_jax
        tx_T_jax = jnp.asarray(self._tx_T_internal, dtype=DTYPE_R_JAX)
        n_sky_modes = A_sky.shape[1]
        n_beam_pix = self.beam.npix
        n_beam_modes = A_beam.shape[1]

        @jax.jit
        def _adjoint(
            sky_coeffs,
            beam_coeffs,
            weighted_resid,
            inv_noise_var,
            terrain_masks,
            terrain_emissions,
            default_emission_masks,
            unresolved_emission,
            unresolved_default_emission,
            unresolved_beam_weights_all,
            beam_px_all,
            beam_wgts_all,
            T_gnd,
            tx_px_all,
            tx_wgts_all,
        ):
            sky_recon = sky_coeffs @ A_sky.T
            beam_recon_all = beam_coeffs @ A_beam.T
            sky_num0 = jnp.zeros(
                (sky_coeffs.shape[0], n_sky_modes), dtype=sky_coeffs.dtype
            )
            sky_den0 = jnp.zeros_like(sky_num0)
            beam_num0 = jnp.zeros_like(beam_coeffs)
            beam_den0 = jnp.zeros_like(beam_coeffs)

            def one_time(carry, args):
                sky_num, sky_den, beam_num, beam_den = carry
                (
                    wr_t,
                    w_t,
                    terrain_mask,
                    terrain_emit,
                    default_emit_mask,
                    unresolved_beam_weights,
                    px,
                    wgts,
                    tx_px,
                    tx_wgts,
                ) = args
                mask = terrain_mask

                def one_dipole(d, dip_carry):
                    sky_num_d, sky_den_d, beam_num_d, beam_den_d = dip_carry
                    beam_recon_d = beam_recon_all[d]
                    wr_d = wr_t[d]
                    w_d = w_t[d]

                    beam_at_sky = jax.lax.fori_loop(
                        0,
                        4,
                        lambda k, acc: acc
                        + beam_recon_d[px[k]] * wgts[k, :, None],
                        jnp.zeros_like(sky_recon),
                    )
                    # Mirror the forward-model normalisation (GN approx:
                    # denominator gradient w.r.t. beam is ignored).
                    sampled_weight_d = jnp.sum(beam_at_sky, axis=0)  # (nfreq,)
                    unresolved_weight_d = unresolved_beam_weights @ beam_recon_d
                    denom_d = sampled_weight_d + unresolved_weight_d  # (nfreq,)

                    sky_jac_f = beam_at_sky * mask[:, None] / denom_d[None, :]
                    sky_num_d = sky_num_d - jnp.einsum(
                        "pf,fm->pm", sky_jac_f * wr_d[None, :], A_sky
                    )
                    sky_den_d = sky_den_d + jnp.einsum(
                        "pf,fm->pm", (sky_jac_f**2) * w_d[None, :], A_sky**2
                    )

                    beam_rhs_pix_f = (
                        sky_recon * mask[:, None] + terrain_emit
                        + T_gnd * default_emit_mask[:, None]
                    ) / denom_d[None, :]
                    pix_num = jnp.zeros(
                        (n_beam_pix, wr_d.shape[0]), dtype=beam_coeffs.dtype
                    )
                    pix_den = jnp.zeros_like(pix_num)

                    def scatter_sky(k, vals):
                        num_k, den_k = vals
                        jac = beam_rhs_pix_f * wgts[k, :, None]
                        num_k = num_k.at[px[k]].add(-jac * wr_d[None, :])
                        den_k = den_k.at[px[k]].add((jac**2) * w_d[None, :])
                        return num_k, den_k

                    pix_num, pix_den = jax.lax.fori_loop(
                        0, 4, scatter_sky, (pix_num, pix_den)
                    )

                    unresolved_spec = (
                        unresolved_emission
                        + T_gnd * unresolved_default_emission
                    )
                    unres_jac = (
                        unresolved_beam_weights[:, None]
                        * unresolved_spec[None, :]
                        / denom_d[None, :]
                    )
                    pix_num = pix_num - unres_jac * wr_d[None, :]
                    pix_den = pix_den + (unres_jac**2) * w_d[None, :]

                    tx_num = jnp.zeros_like(pix_num)
                    tx_den = jnp.zeros_like(pix_den)

                    def scatter_tx(k, vals):
                        num_k, den_k = vals
                        jac = tx_T_jax * tx_wgts[k, :, None] / denom_d[None, :]
                        num_k = num_k.at[tx_px[k]].add(-jac * wr_d[None, :])
                        den_k = den_k.at[tx_px[k]].add((jac**2) * w_d[None, :])
                        return num_k, den_k

                    tx_num, tx_den = jax.lax.fori_loop(
                        0, 4, scatter_tx, (tx_num, tx_den)
                    )
                    pix_num = pix_num + tx_num
                    pix_den = pix_den + tx_den

                    beam_num_d = beam_num_d.at[d].add(pix_num @ A_beam)
                    beam_den_d = beam_den_d.at[d].add(pix_den @ (A_beam**2))
                    return sky_num_d, sky_den_d, beam_num_d, beam_den_d

                return (
                    jax.lax.fori_loop(
                        0,
                        beam_coeffs.shape[0],
                        one_dipole,
                        (sky_num, sky_den, beam_num, beam_den),
                    ),
                    None,
                )

            (sky_num, sky_den, beam_num, beam_den), _ = jax.lax.scan(
                one_time,
                (sky_num0, sky_den0, beam_num0, beam_den0),
                (
                    weighted_resid,
                    inv_noise_var,
                    terrain_masks,
                    terrain_emissions,
                    default_emission_masks,
                    unresolved_beam_weights_all,
                    beam_px_all,
                    beam_wgts_all,
                    tx_px_all,
                    tx_wgts_all,
                ),
            )
            return sky_num, sky_den, beam_num, beam_den

        return _adjoint

    def accumulate_sky_beam_adjoint(
        self,
        sky_coeffs,
        beam_coeffs,
        residual,
        inv_noise_var,
        geom,
        T_gnd=300.0,
    ):
        """Accumulate adjoint numerators and diagonal GN denominators.

        The returned numerators are negative data-gradient accumulations
        ``-J.T @ W @ residual``. Denominators are deterministic diagonal
        Gauss-Newton terms ``diag(J.T @ W @ J)`` for use as a Jacobi/LM
        preconditioned dirty-map update.
        """
        self._ensure_jax_arrays()
        sky_coeffs_jax = jnp.asarray(sky_coeffs, dtype=DTYPE_R_JAX)
        beam_coeffs_jax = jnp.asarray(beam_coeffs, dtype=DTYPE_R_JAX)
        sky_indices_jax = geom.get("sky_indices_jax")
        if sky_indices_jax is not None:
            sky_coeffs_vis = sky_coeffs_jax[sky_indices_jax]
        else:
            sky_coeffs_vis = sky_coeffs_jax

        target_shape = (
            int(geom["terrain_masks_jax"].shape[0]),
            int(beam_coeffs_jax.shape[0]),
            int(self._sky_basis_A_jax.shape[0]),
        )
        residual = self._match_observation_shape(
            residual, target_shape, name="residual"
        )
        inv_noise_var = self._match_observation_shape(
            inv_noise_var, target_shape, name="inv_noise_var"
        )
        residual_jax = jnp.asarray(residual, dtype=DTYPE_R_JAX)
        inv_noise_var_jax = jnp.asarray(inv_noise_var, dtype=DTYPE_R_JAX)
        weighted_resid = residual_jax * inv_noise_var_jax
        sky_num, sky_den, beam_num, beam_den = self._adjoint_jit(
            sky_coeffs_vis,
            beam_coeffs_jax,
            weighted_resid,
            inv_noise_var_jax,
            geom["terrain_masks_jax"],
            geom["terrain_emissions_jax"],
            geom["default_emission_masks_jax"],
            geom["unresolved_emission_jax"],
            geom["unresolved_default_emission_jax"],
            geom["unresolved_beam_weights_jax"],
            geom["beam_px_jax"],
            geom["beam_wgts_jax"],
            jnp.asarray(T_gnd, dtype=DTYPE_R_JAX),
            geom["tx_px_jax"],
            geom["tx_wgts_jax"],
        )

        if sky_indices_jax is not None:
            sky_num_full = (
                jnp.zeros_like(sky_coeffs_jax).at[sky_indices_jax].set(sky_num)
            )
            sky_den_full = (
                jnp.zeros_like(sky_coeffs_jax).at[sky_indices_jax].set(sky_den)
            )
            sky_num, sky_den = sky_num_full, sky_den_full

        return {
            "sky_num": sky_num,
            "sky_den": sky_den,
            "beam_num": beam_num,
            "beam_den": beam_den,
        }

    def precompute_geometry(
        self, times=None, rots=None, body_rots=None, sky_mask=None
    ):
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
            Lunar surface models repeat the observer's current orbital phase;
            use ``times=`` for dynamic lunar surface geometry.
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
            - 'unresolved_beam_weights_jax': (ntimes, npix_beam) float32 — omitted-pixel beam sampling accumulator
            - 'crds_gal_jax': (3, npix_vis) float32 — galactic unit vectors (filtered by sky_mask)
            - 'tx_crds_jax': (ntimes, n_sources, 3) float32 — TX body-frame directions
            - 'beam_px_jax': (ntimes, 4, npix_vis) int32 — cached beam interpolation pixels
            - 'beam_wgts_jax': (ntimes, 4, npix_vis) float32 — cached beam interpolation weights
            - 'tx_px_jax': (ntimes, 4, n_sources) int32 — cached TX interpolation pixels
            - 'tx_wgts_jax': (ntimes, 4, n_sources) float32 — cached TX interpolation weights
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
            crds_gal = self._crds_gal  # (3, npix_sky)

        npix_vis = crds_gal.shape[1]
        observer_occultation_emission = self._observer_occultation_emission()
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
                        "sky_mask with terrain emission requires "
                        "unresolved_emission() or full geometry for exact "
                        "omitted-pixel emission"
                    )
                unresolved_emission = np.zeros(
                    len(self.beam.freqs_hz), dtype=np.float32
                )
            else:
                unresolved_emission = np.asarray(
                    unresolved_emission, dtype=np.float32
                )
        elif observer_occultation_emission is not None:
            unresolved_emission = observer_occultation_emission
            unresolved_default_emission = np.zeros(
                len(self.beam.freqs_hz), dtype=np.float32
            )
        else:
            unresolved_emission = np.zeros(
                len(self.beam.freqs_hz), dtype=np.float32
            )
            unresolved_default_emission = np.ones(
                len(self.beam.freqs_hz), dtype=np.float32
            )

        geom = {}

        if rots is not None:
            # Rots path: skip observer ephemeris and gal→top matmul.
            # The kernel performs R @ crds_gal per step inside jax.lax.scan.
            R_arr = np.stack([np.asarray(r, dtype=np.float32) for r in rots])
            ntimes = R_arr.shape[0]
            geom["rot_gal2top"] = R_arr

            crds_top_arr = R_arr @ crds_gal  # (ntimes, 3, npix_vis)
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
                    t_mask = self.terrain.mask(crds_top_arr[i]).astype(
                        np.float32
                    )
                    terrain_masks_list.append(
                        (obs_masks[i] * t_mask).astype(np.float32)
                    )
                    t_emit = self.terrain.emission(
                        crds_top_arr[i], self.beam.freqs_hz
                    ).astype(np.float32)
                    if observer_occultation_emission is not None:
                        t_emit = t_emit + (
                            (1.0 - obs_masks[i])[:, None]
                            * observer_occultation_emission[None, :]
                        )
                        default_mask = np.zeros(npix_vis, dtype=np.float32)
                    else:
                        default_mask = 1.0 - obs_masks[i]
                    terrain_emissions_list.append(t_emit)
                    default_emission_masks_list.append(default_mask)
                terrain_masks = np.stack(terrain_masks_list)
                terrain_emissions = np.stack(terrain_emissions_list)
                default_emission_masks = np.stack(default_emission_masks_list)
            else:
                terrain_masks = obs_masks
                if observer_occultation_emission is not None:
                    terrain_emissions = (
                        (1.0 - obs_masks)[:, :, None]
                        * observer_occultation_emission[None, None, :]
                    ).astype(np.float32)
                    default_emission_masks = np.zeros_like(obs_masks)
                else:
                    terrain_emissions = np.zeros(
                        (ntimes, npix_vis, len(self.beam.freqs_hz)),
                        dtype=np.float32,
                    )
                    default_emission_masks = 1.0 - obs_masks

        elif times is not None:
            times = [Time(t) if not isinstance(t, Time) else t for t in times]
            ntimes = len(times)
            rot_list, crds_list = [], []
            masks_list, emissions_list, default_emission_masks_list = (
                [],
                [],
                [],
            )

            # Batch-compute rotation matrices when the observer supports it.
            # EarthSurface uses a vectorised astropy call (61× faster than looping).
            if hasattr(self.observer, "rot_gal2top_stack"):
                R_all = self.observer.rot_gal2top_stack(
                    times
                )  # (ntimes, 3, 3)
            else:
                R_all = None

            if getattr(self.observer, "occludes_sky", False) and hasattr(
                self.observer, "above_horizon_stack"
            ):
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
                crds_top = R @ crds_gal  # (3, npix_vis)
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
                    if observer_occultation_emission is not None:
                        t_emit = t_emit + (
                            (1.0 - obs_mask)[:, None]
                            * observer_occultation_emission[None, :]
                        )
                        default_mask = np.zeros(npix_vis, dtype=np.float32)
                    else:
                        default_mask = 1.0 - obs_mask
                    emissions_list.append(t_emit)
                    default_emission_masks_list.append(default_mask)
                else:
                    masks_list.append(obs_mask)
                    if observer_occultation_emission is not None:
                        emissions_list.append(
                            (
                                (1.0 - obs_mask)[:, None]
                                * observer_occultation_emission[None, :]
                            ).astype(np.float32)
                        )
                        default_emission_masks_list.append(
                            np.zeros(npix_vis, dtype=np.float32)
                        )
                    else:
                        emissions_list.append(
                            np.zeros(
                                (npix_vis, len(self.beam.freqs_hz)),
                                dtype=np.float32,
                            )
                        )
                        default_emission_masks_list.append(1.0 - obs_mask)

            R_arr = np.stack(rot_list)
            geom.update(
                {
                    "rot_gal2top": R_arr,
                    "crds_top": np.stack(crds_list),
                    "masks": np.stack(masks_list).astype(np.float32),
                }
            )
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
            )  # (ntimes, 3, 3)
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

        # Beam interpolation depends only on geometry, so cache it once. During
        # calibration this avoids repeating rotations, vec2ang, and HEALPix
        # interpolation for every loss, gradient, and Hessian-vector product.
        crds_body = body_rots_arr @ (R_arr @ crds_gal)
        beam_th = np.arccos(np.clip(crds_body[:, 2], -1.0, 1.0))
        beam_ph = np.mod(
            np.arctan2(crds_body[:, 1], crds_body[:, 0]), 2.0 * np.pi
        )
        beam_px, beam_wgts = healpy.get_interp_weights(
            self.beam.nside, beam_th, beam_ph
        )
        if sky_mask is not None:
            omitted_crds_gal = self._crds_gal[:, ~sky_mask_np]
            omitted_crds_body = body_rots_arr @ (R_arr @ omitted_crds_gal)
            omitted_th = np.arccos(np.clip(omitted_crds_body[:, 2], -1.0, 1.0))
            omitted_ph = np.mod(
                np.arctan2(omitted_crds_body[:, 1], omitted_crds_body[:, 0]),
                2.0 * np.pi,
            )
            omitted_px, omitted_wgts = healpy.get_interp_weights(
                self.beam.nside, omitted_th, omitted_ph
            )
            unresolved_beam_weights = np.zeros(
                (ntimes, self.beam.npix), dtype=np.float32
            )
            for time_index in range(ntimes):
                for neighbor_index in range(4):
                    np.add.at(
                        unresolved_beam_weights[time_index],
                        omitted_px[neighbor_index, time_index],
                        omitted_wgts[neighbor_index, time_index],
                    )
        else:
            unresolved_beam_weights = np.zeros(
                (ntimes, self.beam.npix), dtype=np.float32
            )

        if n_sources > 0:
            tx_th = np.arccos(np.clip(tx_body[:, :, 2], -1.0, 1.0))
            tx_ph = np.mod(
                np.arctan2(tx_body[:, :, 1], tx_body[:, :, 0]), 2.0 * np.pi
            )
            tx_px, tx_wgts = healpy.get_interp_weights(
                self.beam.nside, tx_th, tx_ph
            )
        else:
            tx_px = np.empty((4, ntimes, 0), dtype=np.int32)
            tx_wgts = np.empty((4, ntimes, 0), dtype=np.float32)

        geom["rots_jax"] = jnp.asarray(R_arr, dtype=DTYPE_R_JAX)
        geom["body_rots_jax"] = jnp.asarray(body_rots_arr, dtype=DTYPE_R_JAX)
        geom["beam_px_jax"] = jnp.asarray(
            beam_px.transpose(1, 0, 2), dtype=jnp.int32
        )
        geom["beam_wgts_jax"] = jnp.asarray(
            beam_wgts.transpose(1, 0, 2), dtype=DTYPE_R_JAX
        )
        geom["tx_px_jax"] = jnp.asarray(
            tx_px.transpose(1, 0, 2), dtype=jnp.int32
        )
        geom["tx_wgts_jax"] = jnp.asarray(
            tx_wgts.transpose(1, 0, 2), dtype=DTYPE_R_JAX
        )
        geom["terrain_masks_jax"] = jnp.asarray(
            terrain_masks, dtype=DTYPE_R_JAX
        )
        geom["terrain_emissions_jax"] = jnp.asarray(
            terrain_emissions, dtype=DTYPE_R_JAX
        )
        geom["default_emission_masks_jax"] = jnp.asarray(
            default_emission_masks, dtype=DTYPE_R_JAX
        )
        geom["unresolved_emission_jax"] = jnp.asarray(
            unresolved_emission, dtype=DTYPE_R_JAX
        )
        geom["unresolved_default_emission_jax"] = jnp.asarray(
            unresolved_default_emission, dtype=DTYPE_R_JAX
        )
        geom["unresolved_beam_weights_jax"] = jnp.asarray(
            unresolved_beam_weights, dtype=DTYPE_R_JAX
        )
        geom["crds_gal_jax"] = jnp.asarray(crds_gal, dtype=DTYPE_R_JAX)
        geom["tx_crds_jax"] = jnp.asarray(tx_body, dtype=DTYPE_R_JAX)
        if sky_indices is not None:
            geom["sky_indices_jax"] = jnp.asarray(sky_indices, dtype=jnp.int32)
        return geom

    def sample_beam_weights(self, geom, freq_index, beam_coeffs=None):
        """Sample beam weights for linear recovery diagnostics."""
        from .recovery import sample_beam_weights

        return sample_beam_weights(self, geom, freq_index, beam_coeffs)

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
            obs_mask = self.observer.above_horizon(self.sky.nside).astype(
                np.float32
            )
        else:
            obs_mask = np.ones(self.sky.npix, dtype=np.float32)

        if self.terrain is not None:
            terrain_mask = self.terrain.mask(crds_top).astype(np.float32)
            return obs_mask * terrain_mask
        return obs_mask

    @staticmethod
    def _match_observation_shape(array, target_shape, name="array"):
        """Return observations with explicit (ntimes, ndipoles, nfreq) axes."""
        arr = np.asarray(array, dtype=DTYPE_R_NPY)
        if arr.shape == target_shape:
            return arr
        ntimes, ndipoles, nfreq = target_shape
        if arr.shape == (ntimes, nfreq):
            return np.broadcast_to(arr[:, None, :], target_shape).astype(
                DTYPE_R_NPY
            )
        if arr.shape == (ntimes * ndipoles, nfreq):
            return arr.reshape(target_shape)
        if arr.shape == (nfreq,):
            return np.broadcast_to(arr, target_shape).astype(DTYPE_R_NPY)
        raise ValueError(
            f"{name} has shape {arr.shape}; expected {target_shape}, "
            f"({ntimes}, {nfreq}) broadcast over dipoles, "
            f"or ({ntimes * ndipoles}, {nfreq}) flattened observations"
        )

    def simulate(
        self, sky_coeffs, beam_coeffs=None, times=None, geom=None, T_gnd=300.0,
        T_iso=None, beam_maps=None,
    ):
        """
        Simulate antenna temperature given basis coefficients.

        Either times or geom must be provided.

        Parameters
        ----------
        sky_coeffs : ndarray, shape (npix_sky, nmodes_sky)
        beam_coeffs : ndarray, shape (n_dipoles, npix_beam, nmodes_beam), optional
            Beam spectral-basis coefficients; reconstructed to maps via
            ``beam_coeffs @ basis.A.T``.  Provide this OR ``beam_maps``.
        times : list of Time, optional
            Ignored if geom is provided.
        geom : dict, optional
            Pre-computed geometry from precompute_geometry().
        beam_maps : ndarray, shape (n_dipoles, npix_beam, nfreq), optional
            Pre-reconstructed beam power maps, used directly in place of
            ``beam_coeffs @ basis.A.T``.  Lets a parametric beam (e.g.
            :func:`eigsep_sim.beam.dipole_beam_maps_jax`, differentiable in
            dipole length/orientation) drive the forward model.  Takes
            precedence over ``beam_coeffs`` when both are given.
        T_gnd : float
            Ground temperature [K] for blocked pixels.
        T_iso : ndarray, shape (nfreq,), optional
            Isotropic additive sky component (e.g. cosmological T21 signal) [K].
            Bypasses the sky basis so its full spectral shape enters the data
            unchanged.  Contributes ``T_iso * f_vis`` to the antenna temperature
            where ``f_vis = sum_p(B_p*mask_p) / sum_p(B_p)`` accounts for lunar
            occultation.  Defaults to zero.

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
        if beam_maps is not None:
            beam_recon_all = jnp.asarray(beam_maps, dtype=DTYPE_R_JAX)
        elif beam_coeffs is not None:
            beam_recon_all = (
                jnp.asarray(beam_coeffs, dtype=DTYPE_R_JAX)
                @ self._beam_basis_A_jax.T
            )
        else:
            raise ValueError("provide either beam_coeffs or beam_maps")

        sky_indices_jax = geom.get("sky_indices_jax")
        if sky_indices_jax is not None:
            sky_coeffs_jax = sky_coeffs_jax[
                sky_indices_jax
            ]  # (npix_vis, nmodes_sky)

        nfreq = len(self.beam.freqs_hz)
        T_iso_jax = (
            jnp.zeros(nfreq, dtype=DTYPE_R_JAX)
            if T_iso is None
            else jnp.asarray(T_iso, dtype=DTYPE_R_JAX)
        )

        return self._sim_jit(
            sky_coeffs_jax,
            beam_recon_all,
            geom["terrain_masks_jax"],
            geom["terrain_emissions_jax"],
            geom["default_emission_masks_jax"],
            geom["unresolved_emission_jax"],
            geom["unresolved_default_emission_jax"],
            geom["unresolved_beam_weights_jax"],
            geom["beam_px_jax"],
            geom["beam_wgts_jax"],
            jnp.asarray(T_gnd, dtype=DTYPE_R_JAX),
            T_iso_jax,
            geom["tx_px_jax"],
            geom["tx_wgts_jax"],
        )


class StackedForwardModel:
    """Concatenate multiple ForwardModel instances with shared sky and beam.

    This is useful for multi-spacecraft campaigns where each spacecraft has its
    own observer and geometry, but all observations constrain the same sky and
    beam coefficients. ``geom`` passed to :meth:`simulate` must be a list of
    geometry dicts in the same order as ``forward_models``.
    """

    def __init__(self, forward_models):
        self.forward_models = list(forward_models)
        if len(self.forward_models) == 0:
            raise ValueError("forward_models must be non-empty")
        self.sky = self.forward_models[0].sky
        self.beam = self.forward_models[0].beam
        for fwd in self.forward_models[1:]:
            if fwd.sky is not self.sky or fwd.beam is not self.beam:
                raise ValueError(
                    "all stacked ForwardModel instances must share sky and beam"
                )

    def simulate(
        self, sky_coeffs, beam_coeffs=None, times=None, geom=None, **kwargs
    ):
        """Simulate all models and concatenate along the time axis.

        ``beam_maps=`` may be passed via ``kwargs`` in place of ``beam_coeffs``
        to drive all stacked models with a parametric beam.
        """
        if geom is None:
            if times is None:
                raise ValueError("Either times or geom must be provided")
            geom = [
                fwd.precompute_geometry(times=times)
                for fwd in self.forward_models
            ]
        if len(geom) != len(self.forward_models):
            raise ValueError(
                "geom must contain one geometry dict per stacked ForwardModel"
            )
        outputs = [
            fwd.simulate(sky_coeffs, beam_coeffs, geom=g, **kwargs)
            for fwd, g in zip(self.forward_models, geom)
        ]
        return jnp.concatenate(outputs, axis=0)


class SourceCatalog:
    """
    Placeholder for source catalog.

    In full implementations, would handle:
    - Fixed extragalactic/stellar sources (static galactic coordinates)
    - Solar-system bodies (ephemeris updates per timestep)
    - Point source interpolation into beam
    """

    pass
