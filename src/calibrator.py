"""
Top-level calibrator for joint sky/beam estimation.

Provides Calibrator: optimizes sky and beam basis coefficients jointly using
Anderson-accelerated fixed-point iteration with JAX autodiff gradients.

Architecture:
- ForwardModel: forward simulator (sky_coeffs, beam_coeffs) → antenna_temp
- Calibrator: wrapper with loss function, Anderson Acceleration, step solvers
- Params dict: {'sky_coeffs': ..., 'beam_coeffs': ...}
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict

from .const import DTYPE_R_NPY, DTYPE_R_JAX
from .simulate import ForwardModel

_BEAM_HARMONIC_Q_CACHE = {}


class AndersonAccelerator:
    """
    Type-II Anderson Acceleration for fixed-point iteration.

    Maintains a history of iterates and applies optimal linear combination
    to accelerate convergence.

    Parameters
    ----------
    m : int, optional
        History depth (default 5). Higher values can improve convergence
        at the cost of more memory.
    tol : float, optional
        Threshold for numerical rank deficiency (default 1e-10).
    """

    def __init__(self, m: int = 5, tol: float = 1e-10):
        self.m = max(1, int(m))
        self.tol = float(tol)
        self.x_history = []
        self.fx_diff_history = []

    def reset(self):
        """Clear history."""
        self.x_history = []
        self.fx_diff_history = []

    def apply(self, x_new: np.ndarray, fx_new: np.ndarray) -> np.ndarray:
        """
        Apply Anderson Acceleration to accelerate fixed-point iteration.

        The iteration is x_{n+1} = g(x_n), and this method computes an
        accelerated estimate given x_new and f(x_new) = g(x_new) - x_new.

        Parameters
        ----------
        x_new : ndarray
            Current iterate x_n.
        fx_new : ndarray
            Residual f(x_n) = g(x_n) - x_new.

        Returns
        -------
        x_acc : ndarray
            Anderson-accelerated iterate.
        """
        # Flatten for easier manipulation
        x_shape = x_new.shape
        x = x_new.ravel().astype(np.float64)
        fx = fx_new.ravel().astype(np.float64)

        self.x_history.append(x.copy())
        self.fx_diff_history.append(fx.copy())

        # Keep only the last m iterates
        if len(self.x_history) > self.m:
            self.x_history.pop(0)
            self.fx_diff_history.pop(0)

        # The unaccelerated fixed-point update is g(x) = x + f(x).
        fixed_point = x + fx
        if len(self.x_history) < 2:
            return fixed_point.reshape(x_shape).astype(x_new.dtype)

        # Type-II Anderson acceleration:
        #   g(x_k) - (ΔX_k + ΔF_k) γ,
        # where γ minimizes ||f_k - ΔF_k γ||₂.
        k = len(self.x_history) - 1
        x_diffs = np.column_stack(
            [self.x_history[i + 1] - self.x_history[i] for i in range(k)]
        )
        fx_diffs = np.column_stack(
            [
                self.fx_diff_history[i + 1] - self.fx_diff_history[i]
                for i in range(k)
            ]
        )
        gram = fx_diffs.T @ fx_diffs
        rhs = fx_diffs.T @ fx
        try:
            gamma = np.linalg.solve(gram + self.tol * np.eye(k), rhs)
        except np.linalg.LinAlgError:
            return fixed_point.reshape(x_shape).astype(x_new.dtype)

        x_acc = fixed_point - (x_diffs + fx_diffs) @ gamma
        return x_acc.reshape(x_shape).astype(x_new.dtype)


class Calibrator:
    """
    Joint sky/beam calibrator using Anderson-accelerated fixed-point iteration.

    Optimizes sky and beam basis coefficients jointly to minimize the
    difference between model predictions and observed data. Uses alternating
    optimization (sky step, beam step) with Anderson Acceleration for
    convergence acceleration.

    Parameters
    ----------
    fwd : ForwardModel
        Forward model instance.
    data : ndarray, shape (ntimes, n_dipoles, nfreq)
        Observed antenna temperature [K].
    inv_noise_var : ndarray, optional
        Inverse noise variance weights. If None, uses uniform weighting.
    m_anderson : int, optional
        Anderson Acceleration history depth (default 5).
    lam_beam : float, optional
        Beam regularization strength (default 0.01).
    lam_sky : float, optional
        Sky regularization strength (default 0.0).
    lam_beam_harmonic : float, optional
        Spherical-harmonic beam-shape regularization strength. Penalizes
        high-ell structure in reconstructed beam maps relative to the nominal
        beam. Disabled by default.
    """

    def __init__(
        self,
        fwd: ForwardModel,
        data: np.ndarray,
        inv_noise_var: Optional[np.ndarray] = None,
        m_anderson: int = 5,
        lam_beam: float = 0.01,
        lam_sky: float = 0.0,
        lam_beam_harmonic: float = 1e5,
        beam_harmonic_lmin: int = 4,
        beam_harmonic_lmax: Optional[int] = None,
        beam_harmonic_power: float = 1.0,
    ):
        """
        Initialize Calibrator.

        Parameters
        ----------
        fwd : ForwardModel
            Forward model.
        data : ndarray, shape (ntimes, n_dipoles, nfreq)
            Observed data.
        inv_noise_var : ndarray, optional
            Inverse noise variance weights.
        m_anderson : int, optional
            Anderson history depth.
        lam_beam : float, optional
            Beam regularization.
        lam_sky : float, optional
            Sky regularization.
        lam_beam_harmonic : float, optional
            Spherical-harmonic beam-shape regularization.
        beam_harmonic_lmin : int, optional
            Lowest spherical-harmonic ell to penalize.
        beam_harmonic_lmax : int, optional
            Maximum ell used to build the harmonic penalty operator.
        beam_harmonic_power : float, optional
            Power-law exponent applied to ell(ell+1) weights.
        """
        self.fwd = fwd
        self._data = np.asarray(data, dtype=DTYPE_R_NPY)
        self._inv_noise_var = (
            np.asarray(inv_noise_var, dtype=DTYPE_R_NPY)
            if inv_noise_var is not None
            else np.ones_like(data, dtype=DTYPE_R_NPY)
        )
        self._lam_beam = float(lam_beam)
        self._lam_sky = float(lam_sky)
        self._lam_beam_harmonic = float(lam_beam_harmonic)
        self._beam_harmonic_lmin = int(beam_harmonic_lmin)
        self._beam_harmonic_lmax = (
            None if beam_harmonic_lmax is None else int(beam_harmonic_lmax)
        )
        self._beam_harmonic_power = float(beam_harmonic_power)
        self._beam_harmonic_q = None
        self._beam_harmonic_q_jax = None
        self._beam_harmonic_gram = None
        self._beam_harmonic_gram_jax = None
        self._beam_harmonic_penalty_jit = None
        self._beam_harmonic_diag = None
        self._observation_cache = {}

        # Anderson accelerator
        self._aa = AndersonAccelerator(m=m_anderson)

        # Cache initial nominal beam coefficients for regularization
        self._beam_nom = None

        # Precomputed geometry (cached from init_params or fit)
        self._geom = None

    def _matched_observations(self, pred_shape):
        """Return cached observations matched to a simulator output shape."""
        pred_shape = tuple(int(dim) for dim in pred_shape)
        cached = self._observation_cache.get(pred_shape)
        if cached is not None:
            return cached

        nfreq = pred_shape[-1]
        data = self.fwd._match_observation_shape(
            self._data, pred_shape, name="data"
        )
        inv_noise_var = self.fwd._match_observation_shape(
            self._inv_noise_var, pred_shape, name="inv_noise_var"
        )
        cached = {
            "data": data,
            "inv_noise_var": inv_noise_var,
            "data_flat_jax": jnp.reshape(
                jnp.asarray(data, dtype=DTYPE_R_JAX), (-1, nfreq)
            ),
            "inv_noise_var_flat_jax": jnp.reshape(
                jnp.asarray(inv_noise_var, dtype=DTYPE_R_JAX), (-1, nfreq)
            ),
        }
        self._observation_cache[pred_shape] = cached
        return cached

    def _resolve_geom(
        self, times=None, rots=None, body_rots=None, geom=None, sky_mask=None
    ):
        """Compute and cache geometry from whichever source is provided."""
        if geom is not None:
            self._geom = geom
        elif rots is not None:
            self._geom = self.fwd.precompute_geometry(
                rots=rots, body_rots=body_rots, sky_mask=sky_mask
            )
        elif times is not None:
            self._geom = self.fwd.precompute_geometry(
                times=times, sky_mask=sky_mask
            )

    def init_params(
        self, times=None, rots=None, body_rots=None, geom=None, sky_mask=None
    ) -> Dict[str, np.ndarray]:
        """
        Initialize parameters with nominal beam and zero sky coefficients.

        Also precomputes and caches geometry when observation data is provided.

        Parameters
        ----------
        times : list of Time, optional
        rots : list of (3, 3) ndarray, optional
            Pre-computed gal→top rotation matrices (mutually exclusive with times).
        body_rots : list of (3, 3) ndarray, optional
            Per-step top→body rotations (used with rots or times).
        geom : dict, optional
            Pre-computed geometry dict from ForwardModel.precompute_geometry().
            Takes priority over times/rots when provided.
        sky_mask : ndarray of bool, optional
            Pixel-reduction mask from ForwardModel.build_sky_mask().

        Returns
        -------
        params : dict with keys 'sky_coeffs' and 'beam_coeffs'.
        """
        sky_npix = self.fwd.sky.npix
        sky_nmodes = self.fwd.sky.nmodes
        beam_coeffs = self.fwd.beam.coeffs.astype(DTYPE_R_NPY)

        params = {
            "sky_coeffs": np.zeros((sky_npix, sky_nmodes), dtype=DTYPE_R_NPY),
            "beam_coeffs": beam_coeffs.copy(),
        }
        self._beam_nom = beam_coeffs.copy()

        self._resolve_geom(
            times=times,
            rots=rots,
            body_rots=body_rots,
            geom=geom,
            sky_mask=sky_mask,
        )
        return params

    def _loss(self, params: Dict[str, np.ndarray]) -> float:
        """
        Compute loss = sum of squared weighted residuals + regularization.

        Parameters
        ----------
        params : dict
            Parameters {'sky_coeffs', 'beam_coeffs'}.

        Returns
        -------
        loss : float
        """
        import jax.numpy as jnp

        # Forward simulation (returns JAX array)
        pred = self.fwd.simulate(
            params["sky_coeffs"], params["beam_coeffs"], geom=self._geom
        )

        # Reshape pred and data to (ntimes*n_dipoles, nfreq) for consistent loss computation
        obs = self._matched_observations(pred.shape)
        pred_flat = jnp.reshape(pred, (-1, pred.shape[-1]))
        data_flat = obs["data_flat_jax"]
        inv_noise_var_flat = obs["inv_noise_var_flat_jax"]
        # Data residual
        resid = pred_flat - data_flat
        loss = jnp.mean(inv_noise_var_flat * resid**2)

        # Beam regularization (ridge toward nominal)
        if self._lam_beam > 0 and self._beam_nom is not None:
            beam_nom_jax = jnp.asarray(self._beam_nom)
            beam_diff = params["beam_coeffs"] - beam_nom_jax
            loss = loss + self._lam_beam * jnp.mean(beam_diff**2)

        # Harmonic beam-shape regularization. This damps high-ell spatial
        # structure in reconstructed beam maps at each frequency.
        if self._lam_beam_harmonic > 0:
            loss = loss + self._lam_beam_harmonic * (
                self._beam_harmonic_penalty_jax(params["beam_coeffs"])
            )

        # Sky regularization (ridge toward zero)
        if self._lam_sky > 0:
            loss = loss + self._lam_sky * jnp.mean(params["sky_coeffs"] ** 2)

        return loss

    def data_loss(self, params: Dict[str, np.ndarray]) -> float:
        """Return the unregularized weighted residual loss."""
        import jax.numpy as jnp

        pred = self.fwd.simulate(
            params["sky_coeffs"], params["beam_coeffs"], geom=self._geom
        )
        obs = self._matched_observations(pred.shape)
        pred_flat = jnp.reshape(pred, (-1, pred.shape[-1]))
        data_flat = obs["data_flat_jax"]
        inv_noise_var_flat = obs["inv_noise_var_flat_jax"]
        resid = pred_flat - data_flat
        return float(jnp.mean(inv_noise_var_flat * resid**2))

    def _ensure_linear_ops(self):
        """Build cached linear-operator solvers for the conditional problems.

        The forward model is bilinear: with the beam fixed, predictions are
        an affine function of sky_coeffs through a dense operator
        ``G[d, t, p, f]`` (beam sampled at sky pixels times the visibility
        mask); with the sky fixed they are linear in beam_coeffs through a
        scatter-built operator ``W[t, q, f]``.  Precomputing these operators
        once per conditional solve makes each CG iteration two einsums
        instead of a jvp-of-grad pass through the full simulation kernel
        (~50x faster per iteration on CPU).

        Returns None (and the callers fall back to the autodiff path) when
        the geometry is not a plain dict (e.g. StackedForwardModel) or when
        transmitters are present (their contribution is not built into W).
        """
        if getattr(self, "_linops_geom", None) is self._geom:
            return getattr(self, "_linops", None)
        self._linops_geom = self._geom
        self._linops = None
        if not isinstance(self._geom, dict):
            return None
        if getattr(self.fwd, "_tx_dirs", np.zeros((0, 3))).shape[0] > 0:
            return None

        self.fwd._ensure_jax_arrays()
        geom = self._geom
        A_sky = self.fwd._sky_basis_A_jax  # (F, M)
        A_beam = self.fwd._beam_basis_A_jax  # (F, K)
        px = geom["beam_px_jax"]  # (T, 4, P)
        wg = geom["beam_wgts_jax"]  # (T, 4, P)
        mask = geom["terrain_masks_jax"]  # (T, P)
        emit = geom["terrain_emissions_jax"]  # (T, P, F)
        dmask = geom["default_emission_masks_jax"]  # (T, P)
        ub = geom["unresolved_beam_weights_jax"]  # (T, Q)
        ue = geom["unresolved_emission_jax"]  # (F,)
        ude = geom["unresolved_default_emission_jax"]  # (F,)
        sky_idx = geom.get("sky_indices_jax")
        n_beam_pix = int(self.fwd.beam.npix)
        t_gnd = jnp.asarray(300.0, dtype=DTYPE_R_JAX)

        @jax.jit
        def build_g(beam_coeffs):
            # Layout (D, F, T, P): CG contractions over p become contiguous
            # batched GEMVs (batch dims d, f) instead of strided gathers.
            # Divides by denom = sampled_weight + unresolved_weight so that
            # fwd_op(sky) returns Kelvin, matching the output of simulate().
            beam_recon = beam_coeffs @ A_beam.T  # (D, Q, F)

            def per_dipole(brd):
                b_at = (brd[px] * wg[..., None]).sum(axis=1)  # (T, P, F)
                sampled_w = b_at.sum(axis=1)          # (T, F)
                denom = sampled_w + ub @ brd           # (T, F)
                b_at_norm = b_at / denom[:, None, :]   # (T, P, F)
                return (b_at_norm * mask[..., None]).transpose(2, 0, 1)  # (F,T,P)

            return jax.vmap(per_dipole)(beam_recon)  # (D, F, T, P)

        @jax.jit
        def build_denom(beam_coeffs):
            """Per-dipole beam integral denom[d,t,f] = sampled_weight + unresolved_weight."""
            beam_recon = beam_coeffs @ A_beam.T  # (D, Q, F)

            def per_dipole(brd):
                b_at = (brd[px] * wg[..., None]).sum(axis=1)  # (T, P, F)
                return b_at.sum(axis=1) + ub @ brd  # (T, F)

            return jax.vmap(per_dipole)(beam_recon)  # (D, T, F)

        @jax.jit
        def build_w(sky_coeffs_vis):
            sky_recon = sky_coeffs_vis @ A_sky.T  # (P, F)
            s_eff = (
                sky_recon[None] * mask[..., None]
                + emit
                + t_gnd * dmask[..., None]
            )  # (T, P, F)
            ntimes = px.shape[0]
            nfreq = s_eff.shape[-1]
            flat_idx = (
                jnp.arange(ntimes)[:, None, None] * n_beam_pix + px
            ).reshape(-1)
            vals = (wg[..., None] * s_eff[:, None, :, :]).reshape(-1, nfreq)
            w_op = (
                jnp.zeros((ntimes * n_beam_pix, nfreq), dtype=DTYPE_R_JAX)
                .at[flat_idx]
                .add(vals)
                .reshape(ntimes, n_beam_pix, nfreq)
            )
            w_op = w_op + ub[..., None] * (ue + t_gnd * ude)[None, None, :]
            return w_op.transpose(2, 0, 1)  # (F, T, Q)

        @jax.jit
        def sky_cg_solve(g_op, sky_vis, data_eff, inv_var, lam_abs, n_iter):
            # g_op: (D, F, T, P); data_eff/inv_var passed as (D, F, T).
            # n_iter is a dynamic integer — lax.while_loop compiles the body
            # once regardless of iteration count (see beam_cg_solve for rationale).
            n_data = data_eff.size

            def fwd_op(v):  # (P, M) -> (D, F, T)
                vp = (v @ A_sky.T).T  # (F, P)
                return jnp.einsum("dftp,fp->dft", g_op, vp)

            def adj_op(u):  # (D, F, T) -> (P, M)
                return jnp.einsum("dftp,dft->fp", g_op, u).T @ A_sky

            resid = fwd_op(sky_vis) - data_eff
            b = -(2.0 / n_data) * adj_op(inv_var * resid)
            if self._lam_sky > 0:
                n_sky = sky_vis.size
                b = b - (2.0 * self._lam_sky / n_sky) * sky_vis

            def hvp(v):
                h = (2.0 / n_data) * adj_op(inv_var * fwd_op(v))
                if self._lam_sky > 0:
                    h = h + (2.0 * self._lam_sky / sky_vis.size) * v
                return h + lam_abs * v

            # Diagonal preconditioner: H_diag[p,m] = (2/N) sum_{d,f,t}
            # A_sky[f,m]^2 * g_op[d,f,t,p]^2 * inv_var[d,f,t].
            # With all-sky illumination (wide tumbling) the sky normal matrix
            # spans ~npix*nmodes ~ 12k dimensions and plain CG stalls; the
            # diagonal preconditioner approximately equalises per-pixel/mode
            # curvature and reduces the effective condition number by orders
            # of magnitude.
            g_sq_wt = jnp.einsum("dftp,dft->fp", g_op ** 2, inv_var)
            M_diag = (2.0 / n_data) * jnp.einsum("fm,fp->pm", A_sky ** 2, g_sq_wt)
            M_inv = 1.0 / jnp.maximum(M_diag + lam_abs, 1e-30)

            # Preconditioned CG via lax.while_loop.
            b_dot = jnp.dot(b.ravel(), b.ravel())
            x0 = jnp.zeros_like(b)
            r0 = b
            z0 = r0 * M_inv
            p0 = z0
            rz0 = jnp.dot(r0.ravel(), z0.ravel())

            def cg_cond(state):
                i, _x, r, _z, _p, _rz = state
                rr = jnp.dot(r.ravel(), r.ravel())
                return (i < n_iter) & (rr > 1e-6 * b_dot)

            def cg_body(state):
                i, x, r, z, p, rz = state
                Ap = hvp(p)
                pAp = jnp.dot(p.ravel(), Ap.ravel())
                alpha = jnp.where(pAp > 0, rz / pAp, 0.0)
                x_new = x + alpha * p
                r_new = r - alpha * Ap
                z_new = r_new * M_inv
                rz_new = jnp.dot(r_new.ravel(), z_new.ravel())
                beta = jnp.where(rz > 0, rz_new / rz, 0.0)
                p_new = z_new + beta * p
                return i + 1, x_new, r_new, z_new, p_new, rz_new

            _, delta, _, _, _, _ = jax.lax.while_loop(
                cg_cond, cg_body, (0, x0, r0, z0, p0, rz0)
            )
            return delta

        @jax.jit
        def beam_cg_solve(
            w_op, beam_coeffs, beam_nom, data, inv_var, lam_abs, n_iter
        ):
            # w_op: (D, F, T, Q); data/inv_var passed as (D, F, T).
            # n_iter is a dynamic integer — lax.while_loop compiles the body
            # once regardless of iteration count, avoiding static_argnums and
            # the XLA graph blow-up that unrolling causes at large n_iter.
            n_data = data.size
            n_beam = beam_coeffs.size

            def fwd_op(v):  # (D, Q, K) -> (D, F, T)
                vr = (v @ A_beam.T).transpose(0, 2, 1)  # (D, F, Q)
                return jnp.einsum("dftq,dfq->dft", w_op, vr)

            def adj_op(u):  # (D, F, T) -> (D, Q, K)
                y = jnp.einsum("dftq,dft->dfq", w_op, u)
                return y.transpose(0, 2, 1) @ A_beam

            resid = fwd_op(beam_coeffs) - data
            b = -(2.0 / n_data) * adj_op(inv_var * resid)
            if self._lam_beam > 0:
                b = b - (2.0 * self._lam_beam / n_beam) * (
                    beam_coeffs - beam_nom
                )

            def hvp(v):
                h = (2.0 / n_data) * adj_op(inv_var * fwd_op(v))
                if self._lam_beam > 0:
                    h = h + (2.0 * self._lam_beam / n_beam) * v
                return h + lam_abs * v

            # CG via lax.while_loop so n_iter is a dynamic runtime value.
            b_dot = jnp.dot(b.ravel(), b.ravel())
            x0 = jnp.zeros_like(b)
            r0 = b  # residual at x=0 is just b
            p0 = b
            rr0 = b_dot

            def cg_cond(state):
                i, _x, _r, _p, rr = state
                return (i < n_iter) & (rr > 1e-6 * b_dot)

            def cg_body(state):
                i, x, r, p, rr = state
                Ap = hvp(p)
                pAp = jnp.dot(p.ravel(), Ap.ravel())
                alpha = jnp.where(pAp > 0, rr / pAp, 0.0)
                x_new = x + alpha * p
                r_new = r - alpha * Ap
                rr_new = jnp.dot(r_new.ravel(), r_new.ravel())
                beta = jnp.where(rr > 0, rr_new / rr, 0.0)
                p_new = r_new + beta * p
                return i + 1, x_new, r_new, p_new, rr_new

            _, delta, _, _, _ = jax.lax.while_loop(
                cg_cond, cg_body, (0, x0, r0, p0, rr0)
            )
            return delta

        @jax.jit
        def build_sky_normal(beam_coeffs, sky_vis, data_eff, inv_var):
            # Assemble the exact conditional sky Gauss-Newton system H x = b.
            # The model decouples across frequency in map space, so
            # H = (2/N) sum_f (a_f a_f^T) (x) B_f with the per-frequency
            # pixel Gram B_f = G_f^T W_f G_f.  Returned dense so the caller
            # can Cholesky-solve it in float64 (the system is ill-conditioned
            # from sky coverage; iterative CG stalls on it).
            g_op = build_g(beam_coeffs)  # (D, F, T, Pe)
            n_data = data_eff.size
            n_freq = A_sky.shape[0]
            n_mode = A_sky.shape[1]
            pe = g_op.shape[-1]

            resid = (
                jnp.einsum("dftp,fp->dft", g_op, (sky_vis @ A_sky.T).T)
                - data_eff
            )
            b = -(2.0 / n_data) * (
                jnp.einsum("dftp,dft->fp", g_op, inv_var * resid).T @ A_sky
            )

            gmat = jnp.transpose(g_op, (1, 0, 2, 3)).reshape(n_freq, -1, pe)
            wmat = jnp.transpose(inv_var, (1, 0, 2)).reshape(n_freq, -1)
            b_f = jnp.einsum("fnp,fnq->fpq", gmat * wmat[..., None], gmat)
            h = (2.0 / n_data) * jnp.einsum(
                "fm,fn,fpq->pmqn", A_sky, A_sky, b_f
            )
            return h.reshape(pe * n_mode, pe * n_mode), b

        # Fused operator: H[d,f,t,i] = sum_p g_op[d,f,t,p] * U_sp[p,i].
        # Building H directly inside JIT avoids materialising the full
        # (D,F,T,P) g_op array (~552 MB at nside=16, ntimes=500) as a
        # Python-visible JAX buffer.  Only constructed when sky.basis carries
        # GSM spatial eigenmodes (i.e. created via SkyBasis.from_gsm with the
        # svd_modes patch).
        sky_basis = self.fwd.sky.basis
        build_H = None
        if getattr(sky_basis, "svd_modes", None) is not None:
            svd_modes_full = np.asarray(sky_basis.svd_modes, dtype=DTYPE_R_NPY)
            _U_sp_np = (
                svd_modes_full[np.asarray(sky_idx)]
                if sky_idx is not None
                else svd_modes_full
            )  # (P, n_spatial)
            _U_sp_jax = jnp.asarray(_U_sp_np, dtype=DTYPE_R_JAX)

            @jax.jit
            def build_H(beam_coeffs):
                beam_recon = beam_coeffs @ A_beam.T  # (D, Q, F)

                def per_dipole(brd):
                    b_at = (brd[px] * wg[..., None]).sum(axis=1)  # (T, P, F)
                    sampled_w = b_at.sum(axis=1)           # (T, F)
                    denom = sampled_w + ub @ brd            # (T, F)
                    b_at_norm = b_at / denom[:, None, :]   # (T, P, F)
                    g_ftp = (b_at_norm * mask[..., None]).transpose(2, 0, 1)
                    return jnp.einsum("ftp,pi->fti", g_ftp, _U_sp_jax)  # (F, T, I)

                return jax.vmap(per_dipole)(beam_recon)  # (D, F, T, I)

        self._linops = {
            "build_g": build_g,
            "build_H": build_H,
            "build_w": build_w,
            "build_denom": build_denom,
            "sky_cg_solve": sky_cg_solve,
            "beam_cg_solve": beam_cg_solve,
            "build_sky_normal": build_sky_normal,
            "sky_idx": sky_idx,
        }
        return self._linops

    def _sky_step_direct(self, params, step_size, max_unknowns=8000):
        """Conditional sky solve by exact Cholesky of the normal equations.

        Returns the proposed sky_coeffs array, or None when the fast path is
        unavailable or the dense system (``npix_vis * nmodes``) exceeds
        ``max_unknowns`` (dense Cholesky is O(n^3); above this the
        linear-operator CG path is used instead).  Exact for the quadratic
        conditional problem, so it reaches the conditional minimum in one
        step where truncated CG stalls on the ill-conditioned (kappa ~ 1e8)
        sky-coverage curvature.
        """
        import numpy as _np
        import scipy.linalg as _sla

        ops = self._ensure_linear_ops()
        if ops is None:
            return None
        beam_jax = jnp.asarray(params["beam_coeffs"], dtype=DTYPE_R_JAX)
        sky_full = jnp.asarray(params["sky_coeffs"], dtype=DTYPE_R_JAX)
        sky_idx = ops["sky_idx"]
        sky_vis = sky_full[sky_idx] if sky_idx is not None else sky_full
        n_unknowns = int(sky_vis.shape[0] * sky_vis.shape[1])
        if n_unknowns > max_unknowns:
            return None

        const = self.fwd.simulate(
            jnp.zeros_like(sky_full), beam_jax, geom=self._geom
        )
        obs = self._matched_observations(const.shape)
        data_eff = jnp.transpose(
            jnp.asarray(obs["data"], dtype=DTYPE_R_JAX) - const, (1, 2, 0)
        )
        inv_var = jnp.transpose(
            jnp.asarray(obs["inv_noise_var"], dtype=DTYPE_R_JAX), (1, 2, 0)
        )

        h_jax, b_jax = ops["build_sky_normal"](
            beam_jax, sky_vis, data_eff, inv_var
        )
        h = _np.asarray(h_jax, dtype=_np.float64)
        h = 0.5 * (h + h.T)
        diag = _np.diag_indices_from(h)
        if self._lam_sky > 0:
            h[diag] += 2.0 * self._lam_sky / n_unknowns
        # Tiny relative ridge so the never-sampled (null-space) sky
        # directions stay finite without perturbing constrained modes.
        h[diag] += 1e-8 * _np.trace(h) / n_unknowns
        try:
            c, low = _sla.cho_factor(h, check_finite=False)
            delta = _sla.cho_solve(
                (c, low),
                _np.asarray(b_jax, dtype=_np.float64).ravel(),
                check_finite=False,
            )
        except _sla.LinAlgError:
            return None

        delta = jnp.asarray(delta.reshape(sky_vis.shape), dtype=DTYPE_R_JAX)
        sky_vis_new = sky_vis + step_size * delta
        if sky_idx is not None:
            return np.asarray(
                jnp.zeros_like(sky_full).at[sky_idx].set(sky_vis_new),
                dtype=DTYPE_R_NPY,
            )
        return np.asarray(sky_vis_new, dtype=DTYPE_R_NPY)

    def _sky_step_gsm(self, params, step_size):
        """Conditional sky solve in the GSM spatial basis.

        Compresses the per-pixel sky to the GSM spatial eigenmodes stored in
        ``sky.basis.svd_modes`` (shape ``(npix, n_spatial)``), then solves the
        resulting ``(n_spatial * n_spectral)``-dimensional normal equations
        exactly with ``np.linalg.lstsq``.  This is 10–100× faster than the
        per-pixel CG path for nside ≥ 16, while producing the exact conditional
        minimum within the GSM spatial subspace.

        Returns None when ``svd_modes`` are unavailable or the linear-op cache
        is not ready.
        """
        sky_basis = self.fwd.sky.basis
        if getattr(sky_basis, "svd_modes", None) is None:
            return None
        ops = self._ensure_linear_ops()
        if ops is None:
            return None

        U_sp_full = np.asarray(sky_basis.svd_modes, dtype=DTYPE_R_NPY)  # (npix, I)
        A_sky_np = np.asarray(self.fwd._sky_basis_A_jax)               # (F, J)
        n_spatial, n_spectral = U_sp_full.shape[1], A_sky_np.shape[1]
        n_params = n_spatial * n_spectral

        beam_jax = jnp.asarray(params["beam_coeffs"], dtype=DTYPE_R_JAX)
        sky_full = jnp.asarray(params["sky_coeffs"], dtype=DTYPE_R_JAX)
        sky_idx = ops["sky_idx"]
        sky_vis = sky_full[sky_idx] if sky_idx is not None else sky_full

        # When sky_idx restricts the beam integral to a visible subset, index
        # U_sp to match g_op's P dimension (len(sky_idx) rather than npix).
        U_sp = U_sp_full[np.asarray(sky_idx)] if sky_idx is not None else U_sp_full

        # Terrain offset: prediction at zero sky
        const = self.fwd.simulate(
            jnp.zeros_like(sky_full), beam_jax, geom=self._geom
        )
        obs = self._matched_observations(const.shape)
        data_eff = np.asarray(
            jnp.asarray(obs["data"], dtype=DTYPE_R_JAX) - const
        )  # (T, D, F)
        inv_var_np = np.asarray(obs["inv_noise_var"])  # (T, D, F)

        # H[d,f,t,i] = sum_p g_op[d,f,t,p] * U_sp[p,i]  (P = visible pixels)
        # Prefer the fused build_H (avoids materialising the full 552 MB g_op).
        if ops.get("build_H") is not None:
            H = np.asarray(ops["build_H"](beam_jax))  # (D, F, T, I)
        else:
            g_op_jax = ops["build_g"](beam_jax)       # (D, F, T, P)
            H = np.asarray(
                jnp.einsum("dftp,pi->dfti", g_op_jax, jnp.asarray(U_sp))
            )
            del g_op_jax

        # Design matrix M[(d,f,t), (i*n_spectral+j)] = A_sky[f,j] * H[d,f,t,i]
        # Rows in (D, F, T) order to match the data reshape.
        M = np.einsum("dfti,fj->dftij", H, A_sky_np).reshape(-1, n_params)

        rhs = data_eff.transpose(1, 2, 0).reshape(-1)       # (D*F*T,)
        w = np.sqrt(inv_var_np.transpose(1, 2, 0).reshape(-1))
        c, _, _, _ = np.linalg.lstsq(M * w[:, None], rhs * w, rcond=None)
        c = c.reshape(n_spatial, n_spectral)  # (n_spatial, n_spectral)

        # Reconstruct sky in per-pixel spectral space and apply step damping.
        sky_vis_new = jnp.asarray(U_sp @ c, dtype=DTYPE_R_JAX)  # (P_vis, J)
        sky_vis_stepped = sky_vis + step_size * (sky_vis_new - sky_vis)
        if sky_idx is not None:
            return np.asarray(
                jnp.zeros_like(sky_full).at[sky_idx].set(sky_vis_stepped),
                dtype=DTYPE_R_NPY,
            )
        return np.asarray(sky_vis_stepped, dtype=DTYPE_R_NPY)

    def _sky_step_linear(self, params, n_cg, lam, step_size):
        """Conditional sky solve via the precomputed linear operator.

        Returns the proposed sky_coeffs array, or None when the fast path
        is unavailable.
        """
        ops = self._ensure_linear_ops()
        if ops is None:
            return None
        # (The harmonic beam penalty does not involve the sky, so it does
        # not constrain this conditional solve.)
        beam_jax = jnp.asarray(params["beam_coeffs"], dtype=DTYPE_R_JAX)
        sky_full = jnp.asarray(params["sky_coeffs"], dtype=DTYPE_R_JAX)
        sky_idx = ops["sky_idx"]
        sky_vis = sky_full[sky_idx] if sky_idx is not None else sky_full

        # Affine offset: beam-dependent emission terms = prediction at
        # zero sky.  Solving against (data - offset) makes the problem
        # strictly linear in sky_coeffs.
        const = self.fwd.simulate(
            jnp.zeros_like(sky_full), beam_jax, geom=self._geom
        )
        obs = self._matched_observations(const.shape)
        # Solver layout is (D, F, T) to match the GEMV-friendly operator.
        data_eff = jnp.transpose(
            jnp.asarray(obs["data"], dtype=DTYPE_R_JAX) - const, (1, 2, 0)
        )
        inv_var = jnp.transpose(
            jnp.asarray(obs["inv_noise_var"], dtype=DTYPE_R_JAX), (1, 2, 0)
        )

        g_op = ops["build_g"](beam_jax)
        lam_abs = lam * 1e-6 + 1e-12
        delta = ops["sky_cg_solve"](
            g_op, sky_vis, data_eff, inv_var, lam_abs, n_cg
        )
        if sky_idx is not None:
            delta = jnp.zeros_like(sky_full).at[sky_idx].set(delta)
        return sky_full + step_size * delta

    def _beam_step_linear(self, params, n_cg, lam):
        """Conditional beam solve via the precomputed linear operator.

        Returns the proposed beam_coeffs array, or None when the fast path
        is unavailable.  The spherical-harmonic beam penalty is not built
        into the operator, so callers must fall back when it is active.
        """
        ops = self._ensure_linear_ops()
        if ops is None or self._lam_beam_harmonic > 0:
            return None
        sky_full = jnp.asarray(params["sky_coeffs"], dtype=DTYPE_R_JAX)
        sky_idx = ops["sky_idx"]
        sky_vis = sky_full[sky_idx] if sky_idx is not None else sky_full
        beam_jax = jnp.asarray(params["beam_coeffs"], dtype=DTYPE_R_JAX)
        beam_nom = jnp.asarray(
            self._beam_nom if self._beam_nom is not None else beam_jax,
            dtype=DTYPE_R_JAX,
        )

        w_op_unnorm = ops["build_w"](sky_vis)  # (F, T, Q)
        denom = ops["build_denom"](beam_jax)   # (D, T, F)
        # Normalize per-dipole so beam CG output is in Kelvin, matching simulate().
        # w_op[d,f,t,q] = w_op_unnorm[f,t,q] / denom[d,t,f]
        w_op = (
            w_op_unnorm[None] / denom.transpose(0, 2, 1)[:, :, :, None]
        )  # (D, F, T, Q)
        target_shape = (
            int(w_op.shape[2]),
            int(beam_jax.shape[0]),
            int(w_op.shape[1]),
        )
        obs = self._matched_observations(target_shape)
        # Solver layout is (D, F, T) to match the GEMV-friendly operator.
        data = jnp.transpose(
            jnp.asarray(obs["data"], dtype=DTYPE_R_JAX), (1, 2, 0)
        )
        inv_var = jnp.transpose(
            jnp.asarray(obs["inv_noise_var"], dtype=DTYPE_R_JAX), (1, 2, 0)
        )

        # Tikhonov scale relative to the data-term Hessian diagonal mean,
        # mirroring the Rademacher-probe scaling of the autodiff path.
        basis_power = jnp.mean(jnp.sum(self.fwd._beam_basis_A_jax**2, 0))
        h_scale = float(
            2.0 * jnp.mean(inv_var) * jnp.mean(w_op**2) * basis_power
        )
        lam_abs = lam * max(abs(h_scale), 1e-12) + 1e-12
        delta = ops["beam_cg_solve"](
            w_op, beam_jax, beam_nom, data, inv_var, lam_abs, n_cg
        )
        return beam_jax + delta

    def sky_step(
        self,
        params: Dict[str, np.ndarray],
        n_cg: int = 50,
        lam: float = 1e-4,
        rcond: float = 1e-6,
        step_size: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Sky coefficient update via Newton-CG step (exact for quadratic loss).

        The loss is quadratic in sky_coeffs, so a single Newton step (solved
        via Conjugate Gradient on the Hessian system) gives the exact minimizer.
        The Hessian-vector product is computed efficiently via JAX autodiff.

        Parameters
        ----------
        params : dict
            Current parameters.
        n_cg : int, optional
            Max CG iterations (default 50). Usually converges in ~10 steps.
        lam : float, optional
            Tikhonov regularization for CG (relative to gradient magnitude,
            default 1e-4).
        rcond : float, optional
            Unused; kept for API compatibility.
        step_size : float, optional
            Unused; kept for API compatibility.

        Returns
        -------
        params_new : dict
            Updated parameters with optimized sky_coeffs.
        """

        def loss_sky(sky_coeffs):
            p = {
                "sky_coeffs": sky_coeffs,
                "beam_coeffs": params["beam_coeffs"],
            }
            return self._loss(p)

        sky_jax = jnp.asarray(params["sky_coeffs"])
        loss_before = float(loss_sky(sky_jax))

        # GSM-compressed exact solve: exact conditional minimum within the GSM
        # spatial subspace (n_spatial * n_spectral ~ 16 unknowns).  10–100×
        # faster than the per-pixel CG for nside ≥ 16 and GSM-dominated sky.
        # Checked FIRST because it uses build_H (memory-safe); _sky_step_direct
        # materialises the full g_op array via build_sky_normal and is only
        # used as a fallback when GSM spatial modes are unavailable.
        sky_gsm = self._sky_step_gsm(params, step_size)
        if sky_gsm is not None:
            loss_gsm = float(loss_sky(jnp.asarray(sky_gsm)))
            params_new = params.copy()
            params_new["sky_coeffs"] = sky_gsm
            if loss_gsm < loss_before:
                return params_new
            # GSM half-step didn't improve (current per-pixel sky outfits the
            # subspace); still return the GSM result rather than falling through
            # to _sky_step_linear, which materialises the 440 MB g_op and risks
            # OOM.  The GSM subspace is the correct prior; per-pixel overfitting
            # that beats it will be cleaned up in subsequent joint iterations.
            return params_new

        # Per-pixel Cholesky fallback: only when GSM modes are unavailable.
        # Materialises g_op via build_sky_normal — avoid when build_H exists.
        sky_direct = self._sky_step_direct(params, step_size)
        if sky_direct is not None:
            loss_direct = float(loss_sky(sky_direct))
            if loss_direct < loss_before:
                params_new = params.copy()
                params_new["sky_coeffs"] = sky_direct
                return params_new

        # CG fallback: precomputed linear-operator CG (also materialises g_op).
        # Only reached when both GSM and Cholesky paths are unavailable.
        sky_fast = self._sky_step_linear(params, n_cg, lam, step_size)
        if sky_fast is not None:
            loss_fast = float(loss_sky(sky_fast))
            if loss_fast < loss_before:
                params_new = params.copy()
                params_new["sky_coeffs"] = np.asarray(
                    sky_fast, dtype=DTYPE_R_NPY
                )
                return params_new

        grad_fn = jax.grad(loss_sky)
        grad_val = grad_fn(sky_jax)

        # The sky loss is exactly quadratic in sky_coeffs, so the CG should
        # find the Newton direction without regularization.  A tiny floor
        # is kept only for numerical stability (prevents divide-by-zero in
        # degenerate subspaces).  Do NOT scale with gradient magnitude: the
        # gradient far from the minimum is O(||data||/N), while the Hessian
        # diagonal is O(||A||^2 W/N) — many orders of magnitude smaller —
        # so gradient-scaled lam_abs would completely dominate the Hessian.
        lam_abs = lam * 1e-6 + 1e-12

        # Fully JAX-native HVP: compiled as single XLA kernel by jax.scipy CG
        def hvp_flat(v):
            _, h = jax.jvp(grad_fn, (sky_jax,), (v.reshape(sky_jax.shape),))
            return h.ravel() + lam_abs * v

        b = -grad_val.ravel()
        delta, _ = jax.scipy.sparse.linalg.cg(
            hvp_flat, b, maxiter=n_cg, tol=1e-3
        )

        # Apply step_size damping before checking improvement.
        # step_size < 1 is intentional for joint sky+beam recovery: a full
        # Newton sky step brings sky to its optimal point given the current
        # beam, making the beam gradient (data term) near-zero and preventing
        # beam calibration.  A partial step leaves residual signal for the beam
        # step to act on, enabling convergence of the joint problem.
        sky_new = (sky_jax.ravel() + step_size * delta).reshape(sky_jax.shape)
        loss_new = float(loss_sky(sky_new))

        if loss_new < loss_before:
            params_new = params.copy()
            params_new["sky_coeffs"] = np.asarray(sky_new, dtype=DTYPE_R_NPY)
            return params_new

        # CG failed to improve; fall back to gradient descent with line search
        current_lr = step_size
        for _ in range(20):
            sky_new = (
                sky_jax.ravel() - current_lr * grad_val.ravel()
            ).reshape(sky_jax.shape)
            loss_new = float(loss_sky(sky_new))
            if loss_new <= loss_before:
                params_new = params.copy()
                params_new["sky_coeffs"] = np.asarray(
                    sky_new, dtype=DTYPE_R_NPY
                )
                return params_new
            current_lr *= 0.5

        return params.copy()

    def beam_cg_step(
        self,
        params: Dict[str, np.ndarray],
        n_cg: int = 50,
        lam: float = 1e-4,
        cg_tol: float = 1e-3,
    ) -> Dict[str, np.ndarray]:
        """
        Beam coefficient update via Newton-CG (mirrors sky_step for the beam).

        With sky coefficients fixed, the forward model is linear in beam
        coefficients, so the loss is quadratic. A single Newton-CG solve gives
        the conditional beam minimizer up to the CG tolerance.

        Parameters
        ----------
        params : dict
        n_cg : int, optional
            Max CG iterations per call (default 50). Lower values give a
            faster truncated Newton step that usually moves beam structure much
            more than gradient descent while avoiding full CG cost.
        lam : float, optional
            Tikhonov regularization as fraction of current loss (default 1e-4).
            Scaled by loss_before so regularization is perturbation-size-independent.

        Returns
        -------
        params_new : dict
            Updated parameters with optimized beam_coeffs.
        """

        def loss_beam(beam_coeffs):
            p = {
                "sky_coeffs": params["sky_coeffs"],
                "beam_coeffs": beam_coeffs,
            }
            return self._loss(p)

        beam_jax = jnp.asarray(params["beam_coeffs"])
        loss_before = float(loss_beam(beam_jax))

        # Fast path: precomputed linear-operator CG (skipped when the
        # harmonic beam penalty is active; that term is not built into the
        # operator).
        beam_fast = self._beam_step_linear(params, n_cg, lam)
        if beam_fast is not None:
            loss_fast = float(loss_beam(beam_fast))
            if loss_fast < loss_before:
                params_new = params.copy()
                params_new["beam_coeffs"] = np.asarray(
                    beam_fast, dtype=DTYPE_R_NPY
                )
                return params_new

        grad_fn = jax.grad(loss_beam)
        grad_val = grad_fn(beam_jax)

        # Use H-diagonal estimate as Tikhonov scale: v^T H v / ||v||^2 ≈ trace(H)/n.
        # This avoids the gradient-magnitude scaling bug: ||grad|| >> H_diag when
        # far from minimum, which makes gradient-scaled lam_abs >> H and causes CG
        # to ignore curvature (reduces to over-damped gradient descent).
        rng_key = jax.random.PRNGKey(0)
        v_probe = jax.random.rademacher(
            rng_key, beam_jax.shape, dtype=beam_jax.dtype
        )
        _, h_probe = jax.jvp(grad_fn, (beam_jax,), (v_probe,))
        h_diag_est = float(
            jnp.sum(h_probe * v_probe) / jnp.sum(v_probe * v_probe)
        )
        lam_abs = lam * max(abs(h_diag_est), 1e-12) + 1e-12

        def hvp_flat(v):
            _, h = jax.jvp(grad_fn, (beam_jax,), (v.reshape(beam_jax.shape),))
            return h.ravel() + lam_abs * v

        delta, _ = jax.scipy.sparse.linalg.cg(
            hvp_flat, -grad_val.ravel(), maxiter=n_cg, tol=cg_tol
        )

        beam_new = (beam_jax.ravel() + delta).reshape(beam_jax.shape)
        loss_new = float(loss_beam(beam_new))

        if loss_new < loss_before:
            params_new = params.copy()
            params_new["beam_coeffs"] = np.asarray(beam_new, dtype=DTYPE_R_NPY)
            return params_new

        # CG didn't improve; fall back to gradient descent with adaptive lr.
        current_lr = 0.01 / (float(jnp.max(jnp.abs(grad_val))) + 1e-30)
        for _ in range(30):
            beam_new = beam_jax - current_lr * grad_val
            loss_new = float(loss_beam(beam_new))
            if loss_new <= loss_before:
                params_new = params.copy()
                params_new["beam_coeffs"] = np.asarray(
                    beam_new, dtype=DTYPE_R_NPY
                )
                return params_new
            current_lr *= 0.5

        return params.copy()

    def joint_step(
        self, params: Dict[str, np.ndarray], n_cg: int = 100, lam: float = 1e-4
    ) -> Dict[str, np.ndarray]:
        """
        Joint sky+beam Newton-CG step with block-diagonal preconditioner.

        Treats sky_coeffs and beam_coeffs as a single flat parameter vector and
        applies one Newton step via conjugate gradient on the joint Hessian system.
        The Hessian-vector product is computed by JAX autodiff (one JVP-of-grad
        pass over the full parameter vector).

        Unlike the alternating sky_step/beam_step approach, this includes the
        off-diagonal Hessian blocks H_{sky,beam} that couple sky and beam updates.
        This breaks the trap where a converged sky absorbs beam error and makes
        beam updates appear loss-increasing.

        The sky Hessian diagonal (O(1e-5)) and beam Hessian diagonal (O(1e4))
        differ by ~9 orders of magnitude.  A single joint Rademacher probe averages
        these, yielding a lam_abs that catastrophically over-regularizes sky while
        under-regularizing beam.  Two separate probes (one per block) give
        block-appropriate regularization: lam_abs_sky ≈ lam × H_sky and
        lam_abs_beam ≈ lam × H_beam.

        Parameters
        ----------
        params : dict
            Current parameters.
        n_cg : int, optional
            Max CG iterations per Newton step (default 100).
        lam : float, optional
            Tikhonov regularization relative to per-block H-diagonal (default 1e-4).

        Returns
        -------
        params_new : dict
            Updated parameters with jointly optimized sky_coeffs and beam_coeffs.
        """
        sky_jax = jnp.asarray(params["sky_coeffs"])
        beam_jax = jnp.asarray(params["beam_coeffs"])
        sky_shape, beam_shape = sky_jax.shape, beam_jax.shape
        n_sky = sky_jax.size
        n_beam = beam_jax.size

        def pack(sky, beam):
            return jnp.concatenate([sky.ravel(), beam.ravel()])

        def unpack(theta):
            return theta[:n_sky].reshape(sky_shape), theta[n_sky:].reshape(
                beam_shape
            )

        def loss_joint(theta):
            s, b = unpack(theta)
            return self._loss({"sky_coeffs": s, "beam_coeffs": b})

        theta = pack(sky_jax, beam_jax)
        grad_fn = jax.grad(loss_joint)
        loss_before = float(loss_joint(theta))
        grad_val = grad_fn(theta)

        # Block-diagonal Rademacher probes: separate H-diagonal estimates for sky
        # and beam.  The sky and beam Hessian diagonals differ by ~9 orders of
        # magnitude so a single joint probe would give wildly inappropriate lam_abs
        # for one of the two blocks.  Two probes cost two HVP evaluations (same as
        # one probe + one fallback) but give block-appropriate regularization.
        zeros_beam = jnp.zeros(n_beam, dtype=theta.dtype)
        zeros_sky = jnp.zeros(n_sky, dtype=theta.dtype)

        rng_key = jax.random.PRNGKey(0)
        v_sky = jax.random.rademacher(rng_key, (n_sky,), dtype=theta.dtype)
        v_probe_sky = jnp.concatenate([v_sky, zeros_beam])
        _, h_probe_sky = jax.jvp(grad_fn, (theta,), (v_probe_sky,))
        h_sky_est = float(jnp.sum(h_probe_sky[:n_sky] * v_sky) / n_sky)
        lam_abs_sky = lam * max(abs(h_sky_est), 1e-12) + 1e-12

        rng_key2 = jax.random.PRNGKey(1)
        v_beam = jax.random.rademacher(rng_key2, (n_beam,), dtype=theta.dtype)
        v_probe_beam = jnp.concatenate([zeros_sky, v_beam])
        _, h_probe_beam = jax.jvp(grad_fn, (theta,), (v_probe_beam,))
        h_beam_est = float(jnp.sum(h_probe_beam[n_sky:] * v_beam) / n_beam)
        lam_abs_beam = lam * max(abs(h_beam_est), 1e-12) + 1e-12

        def hvp_flat(v):
            _, h = jax.jvp(grad_fn, (theta,), (v,))
            reg = jnp.concatenate(
                [lam_abs_sky * v[:n_sky], lam_abs_beam * v[n_sky:]]
            )
            return h + reg

        delta, _ = jax.scipy.sparse.linalg.cg(
            hvp_flat, -grad_val, maxiter=n_cg, tol=1e-3
        )

        theta_new = theta + delta
        loss_new = float(loss_joint(theta_new))

        if loss_new < loss_before:
            sky_new, beam_new = unpack(theta_new)
            return {
                "sky_coeffs": np.asarray(sky_new, dtype=DTYPE_R_NPY),
                "beam_coeffs": np.asarray(beam_new, dtype=DTYPE_R_NPY),
            }

        # Newton step didn't improve; fall back to gradient descent with line search.
        current_lr = 0.01 / (float(jnp.max(jnp.abs(grad_val))) + 1e-30)
        for _ in range(30):
            theta_new = theta - current_lr * grad_val
            loss_new = float(loss_joint(theta_new))
            if loss_new <= loss_before:
                sky_new, beam_new = unpack(theta_new)
                return {
                    "sky_coeffs": np.asarray(sky_new, dtype=DTYPE_R_NPY),
                    "beam_coeffs": np.asarray(beam_new, dtype=DTYPE_R_NPY),
                }
            current_lr *= 0.5

        return params.copy()

    def beam_step(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 0.01,
        line_search: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Optimize beam coefficients given fixed sky (JAX gradient step with line search).

        Uses JAX autodiff to compute gradient of loss w.r.t. beam coefficients,
        then applies a gradient descent step with optional line search to ensure
        loss actually decreases.

        Parameters
        ----------
        params : dict
            Current parameters.
        lr : float, optional
            Initial learning rate (default 0.01).
        line_search : bool, optional
            If True, reduce learning rate until loss decreases (default True).

        Returns
        -------
        params_new : dict
            Updated parameters with optimized beam_coeffs.
        """

        # Define loss function for beam only (keep sky fixed)
        def loss_beam(beam_coeffs):
            p = params.copy()
            p["beam_coeffs"] = beam_coeffs
            return self._loss(p)

        # Compute gradient
        grad = jax.grad(loss_beam)(params["beam_coeffs"])
        loss_before = float(loss_beam(params["beam_coeffs"]))

        # Normalize lr so the largest parameter change is lr (not lr * ||grad||).
        # Without this, a fixed lr=0.01 with gradient of O(1e8) gives steps of
        # O(1e6), which always overshoot and cause the line search to exhaust
        # all 10 halvings before reaching a useful step size.
        grad_inf = float(jnp.max(jnp.abs(grad)))
        current_lr = lr / (grad_inf + 1e-30)

        # Gradient step with line search
        for _ in range(30):  # up to 30 halvings to handle large dynamic range
            params_new = params.copy()
            params_new["beam_coeffs"] = (
                params["beam_coeffs"] - current_lr * grad
            )
            loss_new = float(loss_beam(params_new["beam_coeffs"]))

            # Accept step if loss decreased (or line search disabled)
            if not line_search or loss_new <= loss_before:
                return params_new

            # Halve learning rate and try again
            current_lr *= 0.5

        # If all attempts failed, return original params
        return params.copy()

    def _project_scale_degeneracy(
        self, params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Project the multiplicative sky/beam scale gauge in-place."""
        if self._beam_nom is None:
            return params
        beam = np.asarray(params["beam_coeffs"], dtype=DTYPE_R_NPY)
        sky = np.asarray(params["sky_coeffs"], dtype=DTYPE_R_NPY)
        nom = np.asarray(self._beam_nom, dtype=DTYPE_R_NPY)
        nom_rms = float(np.sqrt(np.mean(nom**2)))
        beam_rms = float(np.sqrt(np.mean(beam**2)))
        if nom_rms == 0.0 or beam_rms == 0.0 or not np.isfinite(beam_rms):
            return params
        scale = beam_rms / nom_rms
        out = params.copy()
        out["sky_coeffs"] = np.asarray(sky * scale, dtype=DTYPE_R_NPY)
        out["beam_coeffs"] = np.asarray(beam / scale, dtype=DTYPE_R_NPY)
        return out

    def _rms(self, value):
        value = np.asarray(value, dtype=np.float64)
        return float(np.sqrt(np.mean(value**2)))

    def _beam_roughness(self, beam_coeffs):
        beam = np.asarray(beam_coeffs, dtype=np.float64)
        if beam.shape[1] < 2:
            return 0.0
        return float(np.sqrt(np.mean(np.diff(beam, axis=1) ** 2)))

    def _ensure_beam_harmonic_regularizer(self):
        """Build the dense pixel-space high-ell harmonic penalty operator."""
        if self._beam_harmonic_q is not None:
            return self._beam_harmonic_q
        if self._lam_beam_harmonic <= 0:
            return None

        import healpy

        nside = int(self.fwd.beam.nside)
        npix = int(self.fwd.beam.npix)
        lmax_default = 3 * nside - 1
        lmax = (
            lmax_default
            if self._beam_harmonic_lmax is None
            else min(int(self._beam_harmonic_lmax), lmax_default)
        )
        lmin = max(0, int(self._beam_harmonic_lmin))
        cache_key = (
            nside,
            npix,
            lmin,
            lmax,
            float(self._beam_harmonic_power),
        )
        q = _BEAM_HARMONIC_Q_CACHE.get(cache_key)
        if q is None:
            if lmax < lmin:
                q = np.zeros((npix, npix), dtype=DTYPE_R_NPY)
            else:
                ell, _ = healpy.Alm.getlm(lmax)
                weights = np.zeros_like(ell, dtype=DTYPE_R_NPY)
                active = ell >= lmin
                if np.any(active):
                    ell_norm = max(float(lmin * (lmin + 1)), 1.0)
                    weights[active] = (
                        ell[active] * (ell[active] + 1.0) / ell_norm
                    ) ** self._beam_harmonic_power

                q = np.empty((npix, npix), dtype=DTYPE_R_NPY)
                # map2alm/alm2map accept a stack of maps. Chunking avoids one
                # HEALPix transform call per pixel while keeping memory bounded.
                chunk = max(1, min(npix, 256))
                for start in range(0, npix, chunk):
                    stop = min(start + chunk, npix)
                    unit_maps = np.zeros(
                        (stop - start, npix), dtype=DTYPE_R_NPY
                    )
                    unit_maps[
                        np.arange(stop - start), np.arange(start, stop)
                    ] = 1.0
                    alm = healpy.map2alm(
                        unit_maps,
                        lmax=lmax,
                        iter=0,
                        pol=False,
                        use_weights=False,
                    )
                    filtered = healpy.alm2map(
                        alm * weights[None, :],
                        nside,
                        lmax=lmax,
                        pol=False,
                    )
                    q[:, start:stop] = filtered.T
                q = 0.5 * (q + q.T)
                q = np.asarray(q, dtype=DTYPE_R_NPY)
            _BEAM_HARMONIC_Q_CACHE[cache_key] = q

        self._beam_harmonic_q = q
        self._beam_harmonic_q_jax = jnp.asarray(q, dtype=DTYPE_R_JAX)
        self._beam_harmonic_penalty_jit = None
        q_diag = np.clip(np.diag(q), 0.0, np.inf)
        basis_A = self._beam_basis_A_np()
        self._beam_harmonic_gram = (
            basis_A.T @ basis_A / max(float(basis_A.shape[0]), 1.0)
        ).astype(DTYPE_R_NPY)
        self._beam_harmonic_gram_jax = jnp.asarray(
            self._beam_harmonic_gram, dtype=DTYPE_R_JAX
        )

        q_jax = self._beam_harmonic_q_jax
        gram_jax = self._beam_harmonic_gram_jax

        @jax.jit
        def penalty_jit(beam_coeffs, reference_coeffs):
            diff_coeffs = beam_coeffs - reference_coeffs
            q_diff = jnp.einsum("pq,dqk->dpk", q_jax, diff_coeffs)
            grad_like = jnp.einsum("dpl,lk->dpk", q_diff, gram_jax)
            norm = max(float(beam_coeffs.shape[0] * beam_coeffs.shape[1]), 1.0)
            return jnp.sum(diff_coeffs * grad_like) / norm

        self._beam_harmonic_penalty_jit = penalty_jit
        basis_power = np.clip(np.diag(self._beam_harmonic_gram), 0.0, np.inf)
        self._beam_harmonic_diag = (
            q_diag[None, :, None] * basis_power[None, None, :]
        ).astype(DTYPE_R_NPY)
        return self._beam_harmonic_q

    def _beam_basis_A_np(self):
        return np.asarray(self.fwd.beam.basis.A, dtype=DTYPE_R_NPY)

    def _beam_maps_np(self, beam_coeffs):
        coeffs = np.asarray(beam_coeffs, dtype=DTYPE_R_NPY)
        return np.asarray(
            np.einsum("dpk,fk->dpf", coeffs, self._beam_basis_A_np()),
            dtype=DTYPE_R_NPY,
        )

    def _beam_reference_maps_np(self, beam_coeffs):
        if self._beam_nom is None:
            return np.zeros(
                (
                    *np.asarray(beam_coeffs).shape[:2],
                    self.fwd.beam.basis.nfreq,
                ),
                dtype=DTYPE_R_NPY,
            )
        return self._beam_maps_np(self._beam_nom)

    def _beam_harmonic_apply_np(self, beam_coeffs, return_penalty=False):
        q = self._ensure_beam_harmonic_regularizer()
        coeffs = np.asarray(beam_coeffs, dtype=DTYPE_R_NPY)
        if q is None:
            grad = np.zeros_like(coeffs)
            return (grad, 0.0) if return_penalty else grad
        if self._beam_nom is None:
            diff_coeffs = coeffs
        else:
            diff_coeffs = coeffs - self._beam_nom
        q_diff = np.einsum("pq,dqk->dpk", q, diff_coeffs)
        grad = np.asarray(
            np.einsum("dpl,lk->dpk", q_diff, self._beam_harmonic_gram),
            dtype=DTYPE_R_NPY,
        )
        if return_penalty:
            norm = max(float(diff_coeffs.shape[0] * diff_coeffs.shape[1]), 1.0)
            return grad, float(np.sum(diff_coeffs * grad) / norm)
        return grad

    def _beam_harmonic_penalty_jax(self, beam_coeffs):
        self._ensure_beam_harmonic_regularizer()
        beam = jnp.asarray(beam_coeffs, dtype=DTYPE_R_JAX)
        if self._beam_nom is None:
            ref = jnp.zeros_like(beam)
        else:
            ref = jnp.asarray(self._beam_nom, dtype=DTYPE_R_JAX)
        return self._beam_harmonic_penalty_jit(beam, ref)

    def _beam_harmonic_penalty(self, beam_coeffs):
        if self._lam_beam_harmonic <= 0:
            return 0.0
        _, penalty = self._beam_harmonic_apply_np(
            beam_coeffs, return_penalty=True
        )
        return penalty

    def _split_aligned_update(self, delta, reference):
        """Split ``delta`` into components parallel/perpendicular to reference."""
        delta = np.asarray(delta, dtype=DTYPE_R_NPY)
        reference = np.asarray(reference, dtype=DTYPE_R_NPY)
        denom = float(np.sum(reference * reference))
        if denom <= 0.0 or not np.isfinite(denom):
            return delta, np.zeros_like(delta), 0.0
        alpha = float(np.sum(delta * reference) / denom)
        aligned = np.asarray(alpha * reference, dtype=DTYPE_R_NPY)
        shape = np.asarray(delta - aligned, dtype=DTYPE_R_NPY)
        return shape, aligned, alpha

    def _project_joint_scale_tangent(self, sky_delta, beam_delta, params):
        """Remove the coupled sky/beam multiplicative gauge tangent."""
        sky = np.asarray(params["sky_coeffs"], dtype=DTYPE_R_NPY)
        beam = np.asarray(params["beam_coeffs"], dtype=DTYPE_R_NPY)
        sky_delta = np.asarray(sky_delta, dtype=DTYPE_R_NPY)
        beam_delta = np.asarray(beam_delta, dtype=DTYPE_R_NPY)
        denom = float(np.sum(sky * sky) + np.sum(beam * beam))
        if denom <= 0.0 or not np.isfinite(denom):
            zeros_sky = np.zeros_like(sky_delta)
            zeros_beam = np.zeros_like(beam_delta)
            return sky_delta, beam_delta, zeros_sky, zeros_beam, 0.0
        alpha = float(
            (np.sum(sky_delta * sky) - np.sum(beam_delta * beam)) / denom
        )
        sky_scale = np.asarray(alpha * sky, dtype=DTYPE_R_NPY)
        beam_scale = np.asarray(-alpha * beam, dtype=DTYPE_R_NPY)
        sky_shape = np.asarray(sky_delta - sky_scale, dtype=DTYPE_R_NPY)
        beam_shape = np.asarray(beam_delta - beam_scale, dtype=DTYPE_R_NPY)
        return sky_shape, beam_shape, sky_scale, beam_scale, alpha

    def _adaptive_fixed_point_step(
        self,
        params,
        lambda_damp=1e-2,
        step0=1.0,
        min_step=1e-4,
        allow_cg_fallback=True,
        beam_cg_niter=5,
        beam_cg_tol=1e-2,
        blocks=("joint", "sky", "beam"),
        diagnostics=True,
    ):
        pred = self.fwd.simulate(
            params["sky_coeffs"], params["beam_coeffs"], geom=self._geom
        )
        pred_np = np.asarray(pred)
        obs = self._matched_observations(pred_np.shape)
        data = obs["data"]
        inv_noise_var = obs["inv_noise_var"]
        residual = pred_np - data
        adj = self.fwd.accumulate_sky_beam_adjoint(
            params["sky_coeffs"],
            params["beam_coeffs"],
            residual,
            inv_noise_var,
            self._geom,
        )
        sky_num = np.asarray(adj["sky_num"], dtype=DTYPE_R_NPY)
        sky_den = np.asarray(adj["sky_den"], dtype=DTYPE_R_NPY)
        beam_num = np.asarray(adj["beam_num"], dtype=DTYPE_R_NPY)
        beam_den = np.asarray(adj["beam_den"], dtype=DTYPE_R_NPY)

        regularization_loss = 0.0
        if self._lam_sky > 0:
            sky_den = sky_den + self._lam_sky
            sky_num = sky_num - self._lam_sky * params["sky_coeffs"]
            regularization_loss += self._lam_sky * float(
                np.mean(params["sky_coeffs"] ** 2)
            )
        if self._lam_beam > 0 and self._beam_nom is not None:
            beam_diff = params["beam_coeffs"] - self._beam_nom
            beam_den = beam_den + self._lam_beam
            beam_num = beam_num - self._lam_beam * beam_diff
            regularization_loss += self._lam_beam * float(
                np.mean(beam_diff**2)
            )
        if self._lam_beam_harmonic > 0:
            beam_harmonic_grad, beam_harmonic_penalty = (
                self._beam_harmonic_apply_np(
                    params["beam_coeffs"], return_penalty=True
                )
            )
            beam_num = beam_num - self._lam_beam_harmonic * beam_harmonic_grad
            beam_den = beam_den + self._lam_beam_harmonic * (
                self._beam_harmonic_diag
            )
            regularization_loss += (
                self._lam_beam_harmonic * beam_harmonic_penalty
            )

        sky_floor = lambda_damp * max(float(np.max(sky_den)), 1e-30)
        beam_floor = lambda_damp * max(float(np.max(beam_den)), 1e-30)
        sky_delta = sky_num / (sky_den + sky_floor)
        beam_delta = beam_num / (beam_den + beam_floor)
        beam_shape_delta, beam_scale_delta, beam_scale_alpha = (
            self._split_aligned_update(beam_delta, params["beam_coeffs"])
        )
        (
            joint_sky_delta,
            joint_beam_delta,
            joint_sky_scale_delta,
            joint_beam_scale_delta,
            joint_scale_alpha,
        ) = self._project_joint_scale_tangent(sky_delta, beam_delta, params)

        loss_before = float(np.mean(inv_noise_var * residual**2))
        loss_before += regularization_loss
        best = params
        best_loss = loss_before
        best_type = "none"
        all_blocks = ("joint", "sky", "beam")
        blocks = tuple(blocks)
        candidate_summary = {}
        for block in all_blocks:
            candidate_summary[f"{block}_step"] = 0.0
            candidate_summary[f"{block}_loss"] = np.nan

        def make_candidate(block, step):
            candidate = params.copy()
            if block == "joint":
                candidate["sky_coeffs"] = np.asarray(
                    params["sky_coeffs"] + step * joint_sky_delta,
                    dtype=DTYPE_R_NPY,
                )
                candidate["beam_coeffs"] = np.asarray(
                    params["beam_coeffs"] + step * joint_beam_delta,
                    dtype=DTYPE_R_NPY,
                )
            elif block == "sky":
                candidate["sky_coeffs"] = np.asarray(
                    params["sky_coeffs"] + step * sky_delta,
                    dtype=DTYPE_R_NPY,
                )
            elif block == "beam":
                candidate["beam_coeffs"] = np.asarray(
                    params["beam_coeffs"] + step * beam_shape_delta,
                    dtype=DTYPE_R_NPY,
                )
            return self._project_scale_degeneracy(candidate)

        for block in blocks:
            step = float(step0)
            block_best = None
            block_best_loss = loss_before
            block_best_step = 0.0
            while step >= min_step:
                candidate = make_candidate(block, step)
                loss_candidate = float(self._loss(candidate))
                if loss_candidate <= block_best_loss:
                    block_best = candidate
                    block_best_loss = loss_candidate
                    block_best_step = step
                    break
                step *= 0.5

            candidate_summary[f"{block}_step"] = block_best_step
            candidate_summary[f"{block}_loss"] = block_best_loss
            if block_best is not None and block_best_loss < best_loss:
                best = block_best
                best_loss = block_best_loss
                best_type = f"adjoint-{block}:{block_best_step:.3g}"

        if best_type == "none" and allow_cg_fallback:
            candidate = self.sky_step(params, step_size=0.5)
            candidate = self.beam_cg_step(
                candidate, n_cg=beam_cg_niter, cg_tol=beam_cg_tol
            )
            candidate = self._project_scale_degeneracy(candidate)
            loss_candidate = float(self._loss(candidate))
            if loss_candidate <= best_loss:
                best = candidate
                best_loss = loss_candidate
                best_type = "fast-cg"

        step_info = {**candidate_summary}
        if diagnostics:
            step_info.update(
                {
                    "sky_update_rms": self._rms(sky_delta),
                    "beam_update_rms": self._rms(beam_delta),
                    "beam_shape_update_rms": self._rms(beam_shape_delta),
                    "beam_scale_update_rms": self._rms(beam_scale_delta),
                    "joint_sky_shape_update_rms": self._rms(joint_sky_delta),
                    "joint_sky_scale_update_rms": self._rms(
                        joint_sky_scale_delta
                    ),
                    "joint_beam_shape_update_rms": self._rms(joint_beam_delta),
                    "joint_beam_scale_update_rms": self._rms(
                        joint_beam_scale_delta
                    ),
                    "beam_scale_alpha": beam_scale_alpha,
                    "joint_scale_alpha": joint_scale_alpha,
                    "sky_diag_floor": sky_floor,
                    "beam_diag_floor": beam_floor,
                }
            )
        return best, best_loss, best_type, step_info

    def fit(
        self,
        params: Optional[Dict[str, np.ndarray]] = None,
        times=None,
        rots=None,
        body_rots=None,
        geom=None,
        sky_mask=None,
        max_iter: int = 30,
        tol: float = 1e-6,
        verbose: bool = True,
        solver: str = "adaptive-fixed-point",
        use_cg: bool = False,
        use_joint: bool = False,
        sky_step_size: float = 1.0,
        sky_cg_niter: int = 50,
        beam_cg_niter: int = 50,
        beam_cg_tol: float = 1e-3,
        lambda_damp: float = 1e-2,
        schedule_max_every: Optional[Dict[str, int]] = None,
        schedule_eff_alpha: float = 0.3,
        schedule_step_gain_factor: float = 2.0,
        schedule_min_step: float = 1e-4,
        schedule_lbfgs_max_every: int = 0,
        schedule_lbfgs_min_iter: int = 20,
        schedule_lbfgs_maxiter: int = 3,
        schedule_lbfgs_max_runs: int = 1,
        telemetry_level: str = "full",
    ) -> Dict:
        """Run calibration.

        ``solver`` may be ``adaptive-fixed-point`` (default),
        ``adaptive-scheduled``, ``hybrid-lbfgs``, ``fast-cg``, ``cg``,
        ``joint``, or ``alternating``. The legacy boolean
        controls remain accepted: ``use_joint=True`` selects ``joint`` and
        ``use_cg=True`` selects ``cg`` when the default solver is not explicitly
        overridden.

        A ``KeyboardInterrupt`` (e.g. Ctrl-C in a notebook) is caught and
        ends the fit cleanly at the last completed iteration; the usual
        result dict is returned with ``converged=False`` and
        ``interrupted=True`` so an interrupted run is still usable.
        """
        if telemetry_level not in ("summary", "full"):
            raise ValueError("telemetry_level must be 'summary' or 'full'")
        full_telemetry = telemetry_level == "full"

        if params is None:
            params = self.init_params(
                times=times,
                rots=rots,
                body_rots=body_rots,
                geom=geom,
                sky_mask=sky_mask,
            )
        elif self._geom is None:
            self._resolve_geom(
                times=times,
                rots=rots,
                body_rots=body_rots,
                geom=geom,
                sky_mask=sky_mask,
            )
            if self._beam_nom is None:
                self._beam_nom = np.asarray(
                    params["beam_coeffs"], dtype=DTYPE_R_NPY
                ).copy()

        if self._geom is None:
            raise ValueError(
                "Provide times, rots, or geom to specify observation geometry"
            )

        if solver == "adaptive-fixed-point":
            if use_joint:
                solver = "joint"
            elif use_cg:
                solver = "cg"
        if solver == "hybrid-lbfgs":
            first = self.fit(
                params=params,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                solver="adaptive-fixed-point",
                sky_step_size=sky_step_size,
                beam_cg_niter=beam_cg_niter,
                beam_cg_tol=beam_cg_tol,
                lambda_damp=lambda_damp,
                telemetry_level=telemetry_level,
            )
            return self.fit_lbfgs(
                first["params"], maxiter=20, history=first.get("telemetry", [])
            )

        self._aa.reset()
        losses = []
        telemetry = []
        converged = False
        previous_loss = float(self._loss(params))
        scheduler = None
        if solver == "adaptive-scheduled":
            max_every = {"sky": 5, "beam": 2, "joint": 4}
            if schedule_lbfgs_max_every and schedule_lbfgs_max_every > 0:
                max_every["lbfgs"] = int(schedule_lbfgs_max_every)
            if schedule_max_every is not None:
                max_every.update(schedule_max_every)
            priority = ["beam", "joint", "sky"]
            if max_every.get("lbfgs", 0) > 0:
                priority.append("lbfgs")
            scheduler = {
                "priority": tuple(priority),
                "max_every": max_every,
                "eff": {block: None for block in priority},
                "n_since": {block: 0 for block in priority},
                "step_gain": {block: 1.0 for block in priority},
                "n_run": {block: 0 for block in priority},
            }

        def choose_scheduled_block():
            eligible = []
            for block in scheduler["priority"]:
                if block == "lbfgs" and iteration < schedule_lbfgs_min_iter:
                    continue
                if (
                    block == "lbfgs"
                    and schedule_lbfgs_max_runs > 0
                    and scheduler["n_run"][block] >= schedule_lbfgs_max_runs
                ):
                    continue
                eligible.append(block)
            if not eligible:
                eligible = [
                    block
                    for block in scheduler["priority"]
                    if block != "lbfgs"
                ]
            overdue = {}
            for block in eligible:
                max_count = scheduler["max_every"].get(block, 0)
                if max_count and max_count > 0:
                    n_since = scheduler["n_since"][block]
                    if n_since >= max_count:
                        overdue[block] = n_since / max_count
            if overdue:
                return max(overdue, key=overdue.get), "overdue"
            for block in eligible:
                if scheduler["eff"][block] is None:
                    return block, "unmeasured"
            return (
                max(eligible, key=lambda b: scheduler["eff"][b]),
                "efficiency",
            )

        interrupted = False
        try:
            for iteration in range(max_iter):
                tic = time.perf_counter()
                beam_old = params["beam_coeffs"].copy()
                step_extra = {}

                scheduled_block = None
                schedule_reason = None
                if solver in ("adaptive-fixed-point", "adaptive-scheduled"):
                    step0 = 1.0
                    blocks = ("joint", "sky", "beam")
                    if solver == "adaptive-scheduled":
                        scheduled_block, schedule_reason = (
                            choose_scheduled_block()
                        )
                        if scheduled_block == "lbfgs":
                            lbfgs_result = self.fit_lbfgs(
                                params, maxiter=schedule_lbfgs_maxiter
                            )
                            params = lbfgs_result["params"]
                            loss = float(lbfgs_result["losses"][-1])
                            step_type = f"lbfgs:{schedule_lbfgs_maxiter}"
                            step_extra = {
                                "lbfgs_maxiter": schedule_lbfgs_maxiter,
                                "lbfgs_inner_iter": lbfgs_result.get("n_iter"),
                            }
                        else:
                            blocks = (scheduled_block,)
                            step0 = scheduler["step_gain"][scheduled_block]
                    if (
                        solver != "adaptive-scheduled"
                        or scheduled_block != "lbfgs"
                    ):
                        params, loss, step_type, step_extra = (
                            self._adaptive_fixed_point_step(
                                params,
                                lambda_damp=lambda_damp,
                                step0=step0,
                                min_step=schedule_min_step,
                                beam_cg_niter=min(beam_cg_niter, 10),
                                beam_cg_tol=beam_cg_tol,
                                blocks=blocks,
                                diagnostics=full_telemetry,
                            )
                        )
                elif solver == "joint":
                    params = self._project_scale_degeneracy(
                        self.joint_step(params)
                    )
                    loss = float(self._loss(params))
                    step_type = "joint"
                else:
                    # fast-cg truncates the beam conditional solve (10 iters
                    # is enough since the beam starts near nominal), but lets
                    # sky_cg_niter flow through unchanged.  The beam CG cap
                    # was paired with a sky CG cap of 25, but that cap only
                    # worked when few sky pixels were illuminated (equatorial
                    # attitude); with all-sky coverage the sky normal matrix
                    # spans ~npix*nmodes >> 25 and needs more iterations to
                    # converge.
                    n_cg_sky = sky_cg_niter
                    params = self.sky_step(
                        params, n_cg=n_cg_sky, step_size=sky_step_size
                    )
                    if solver in ("cg", "fast-cg"):
                        n_cg = (
                            min(beam_cg_niter, 10)
                            if solver == "fast-cg"
                            else beam_cg_niter
                        )
                        params = self.beam_cg_step(
                            params, n_cg=n_cg, cg_tol=beam_cg_tol
                        )
                        step_type = solver
                    elif solver == "alternating":
                        params = self.beam_step(params)
                        step_type = "gradient"
                    else:
                        raise ValueError(f"unknown solver: {solver}")
                    params = self._project_scale_degeneracy(params)

                    beam_new = params["beam_coeffs"]
                    beam_res = beam_new - beam_old
                    beam_acc = self._aa.apply(beam_old, beam_res)
                    params_aa = params.copy()
                    params_aa["beam_coeffs"] = np.asarray(
                        beam_acc, dtype=DTYPE_R_NPY
                    )
                    params_aa = self._project_scale_degeneracy(params_aa)
                    loss_step = float(self._loss(params))
                    if len(self._aa.x_history) < 2:
                        loss = loss_step
                    else:
                        loss_aa = float(self._loss(params_aa))
                        if loss_aa <= loss_step:
                            params = params_aa
                            loss = loss_aa
                            step_type = step_type + "+aa"
                        else:
                            loss = loss_step

                wall_time = time.perf_counter() - tic
                delta = previous_loss - loss
                if scheduler is not None:
                    eff = max(0.0, delta) / (
                        max(abs(previous_loss), 1e-30) * max(wall_time, 1e-30)
                    )
                    block = scheduled_block
                    if block is not None:
                        old_eff = scheduler["eff"][block]
                        if old_eff is None:
                            scheduler["eff"][block] = eff
                        else:
                            ema = (
                                schedule_eff_alpha * eff
                                + (1.0 - schedule_eff_alpha) * old_eff
                            )
                            scheduler["eff"][block] = min(ema, eff)
                        if block != "lbfgs":
                            accepted_step = float(
                                step_extra.get(f"{block}_step", 0.0)
                            )
                            old_gain = scheduler["step_gain"][block]
                            if accepted_step > 0.0:
                                if accepted_step >= 0.99 * old_gain:
                                    new_gain = (
                                        old_gain * schedule_step_gain_factor
                                    )
                                else:
                                    new_gain = accepted_step
                            else:
                                new_gain = old_gain / schedule_step_gain_factor
                            scheduler["step_gain"][block] = max(
                                schedule_min_step, min(1.0, float(new_gain))
                            )
                        for name in scheduler["n_since"]:
                            scheduler["n_since"][name] += 1
                        scheduler["n_since"][block] = 0
                        scheduler["n_run"][block] += 1
                losses.append(loss)
                entry = {
                    "iteration": iteration,
                    "wall_time": wall_time,
                    "loss": loss,
                    "delta_chi2": delta,
                    "delta_chi2_per_sec": delta / max(wall_time, 1e-30),
                    "step_type": step_type,
                }
                if full_telemetry:
                    entry.update(
                        {
                            "projected_sky_rms": self._rms(
                                params["sky_coeffs"]
                            ),
                            "projected_beam_rms": self._rms(
                                params["beam_coeffs"]
                            ),
                            "beam_scatter": float(
                                np.std(params["beam_coeffs"])
                            ),
                            "beam_roughness": self._beam_roughness(
                                params["beam_coeffs"]
                            ),
                            "beam_harmonic_penalty": (
                                self._beam_harmonic_penalty(
                                    params["beam_coeffs"]
                                )
                            ),
                        }
                    )
                if scheduler is not None:
                    entry.update(
                        {
                            "scheduled_block": scheduled_block,
                            "schedule_reason": schedule_reason,
                        }
                    )
                    if full_telemetry:
                        entry.update(
                            {
                                "schedule_eff_sky": scheduler["eff"].get(
                                    "sky"
                                ),
                                "schedule_eff_beam": scheduler["eff"].get(
                                    "beam"
                                ),
                                "schedule_eff_joint": scheduler["eff"].get(
                                    "joint"
                                ),
                                "schedule_eff_lbfgs": scheduler["eff"].get(
                                    "lbfgs"
                                ),
                                "schedule_n_since_sky": scheduler[
                                    "n_since"
                                ].get("sky"),
                                "schedule_n_since_beam": scheduler[
                                    "n_since"
                                ].get("beam"),
                                "schedule_n_since_joint": scheduler[
                                    "n_since"
                                ].get("joint"),
                                "schedule_n_since_lbfgs": scheduler[
                                    "n_since"
                                ].get("lbfgs"),
                                "schedule_n_run_sky": scheduler["n_run"].get(
                                    "sky"
                                ),
                                "schedule_n_run_beam": scheduler["n_run"].get(
                                    "beam"
                                ),
                                "schedule_n_run_joint": scheduler["n_run"].get(
                                    "joint"
                                ),
                                "schedule_n_run_lbfgs": scheduler["n_run"].get(
                                    "lbfgs"
                                ),
                                "schedule_step_gain_sky": scheduler[
                                    "step_gain"
                                ].get("sky"),
                                "schedule_step_gain_beam": scheduler[
                                    "step_gain"
                                ].get("beam"),
                                "schedule_step_gain_joint": scheduler[
                                    "step_gain"
                                ].get("joint"),
                            }
                        )
                entry.update(step_extra)
                telemetry.append(entry)

                if verbose:
                    if iteration == 0:
                        print(
                            f"iter {iteration:3d}: loss = {loss:.6e}  "
                            f"step = {step_type}  dt = {wall_time:.3f}s"
                        )
                    else:
                        rel = abs(losses[-2] - loss) / (
                            abs(losses[-2]) + 1e-30
                        )
                        print(
                            f"iter {iteration:3d}: loss = {loss:.6e}  "
                            f"rel_D = {rel:.2e}  step = {step_type}  "
                            f"dt = {wall_time:.3f}s "
                            f"dchi2/s = {entry['delta_chi2_per_sec']:.3e}"
                        )

                if iteration > 0:
                    rel = abs(losses[-2] - loss) / (abs(losses[-2]) + 1e-30)
                    if rel < tol:
                        if verbose:
                            print(
                                f"Converged after {iteration + 1} iterations"
                            )
                        converged = True
                        break
                previous_loss = loss
        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                print(
                    f"\nfit interrupted at iteration {iteration}; "
                    "returning best result so far."
                )

        return {
            "params": params,
            "losses": losses,
            "telemetry": telemetry,
            "converged": converged,
            "n_iter": len(losses),
            "interrupted": interrupted,
            "solver": solver,
        }

    def fit_lbfgs(
        self,
        params: Dict[str, np.ndarray],
        maxiter: int = 50,
        scales: Optional[Dict[str, np.ndarray]] = None,
        history=None,
    ) -> Dict:
        """Refine a near-minimum solution with scaled jaxopt L-BFGS."""
        try:
            import jaxopt
        except ImportError as exc:
            raise ImportError(
                "fit_lbfgs requires the jaxopt dependency"
            ) from exc

        sky0 = jnp.asarray(params["sky_coeffs"], dtype=DTYPE_R_JAX)
        beam0 = jnp.asarray(params["beam_coeffs"], dtype=DTYPE_R_JAX)
        sky_shape = sky0.shape
        beam_shape = beam0.shape
        n_sky = sky0.size
        if scales is None:
            scales = {
                "sky_coeffs": np.maximum(
                    np.sqrt(np.mean(np.asarray(sky0) ** 2)), 1.0
                ),
                "beam_coeffs": np.maximum(
                    np.sqrt(np.mean(np.asarray(beam0) ** 2)), 1.0
                ),
            }
        sky_scale = jnp.asarray(scales["sky_coeffs"], dtype=DTYPE_R_JAX)
        beam_scale = jnp.asarray(scales["beam_coeffs"], dtype=DTYPE_R_JAX)

        def pack(sky, beam):
            return jnp.concatenate(
                [(sky / sky_scale).ravel(), (beam / beam_scale).ravel()]
            )

        def unpack(theta):
            sky = theta[:n_sky].reshape(sky_shape) * sky_scale
            beam = theta[n_sky:].reshape(beam_shape) * beam_scale
            return sky, beam

        def objective(theta):
            sky, beam = unpack(theta)
            return self._loss({"sky_coeffs": sky, "beam_coeffs": beam})

        solver = jaxopt.LBFGS(fun=objective, maxiter=maxiter)
        theta0 = pack(sky0, beam0)
        result = solver.run(theta0)
        sky_new, beam_new = unpack(result.params)
        out_params = self._project_scale_degeneracy(
            {
                "sky_coeffs": np.asarray(sky_new, dtype=DTYPE_R_NPY),
                "beam_coeffs": np.asarray(beam_new, dtype=DTYPE_R_NPY),
            }
        )
        loss = float(self._loss(out_params))
        telemetry = list(history or [])
        telemetry.append(
            {
                "iteration": len(telemetry),
                "wall_time": 0.0,
                "loss": loss,
                "delta_chi2": np.nan,
                "delta_chi2_per_sec": np.nan,
                "step_type": "lbfgs",
                "projected_sky_rms": self._rms(out_params["sky_coeffs"]),
                "projected_beam_rms": self._rms(out_params["beam_coeffs"]),
                "beam_scatter": float(np.std(out_params["beam_coeffs"])),
                "beam_roughness": self._beam_roughness(
                    out_params["beam_coeffs"]
                ),
                "beam_harmonic_penalty": self._beam_harmonic_penalty(
                    out_params["beam_coeffs"]
                ),
            }
        )
        return {
            "params": out_params,
            "losses": [loss],
            "telemetry": telemetry,
            "converged": bool(getattr(result.state, "error", np.inf) < 1e-6),
            "n_iter": int(getattr(result.state, "iter_num", maxiter)),
            "solver": "lbfgs",
            "state": result.state,
        }

    def fit_hybrid(
        self,
        params: Optional[Dict[str, np.ndarray]] = None,
        max_iter: int = 30,
        lbfgs_iter: int = 20,
        **kwargs,
    ) -> Dict:
        """Run adaptive fixed-point iterations followed by L-BFGS refinement."""
        first = self.fit(
            params=params,
            max_iter=max_iter,
            solver="adaptive-fixed-point",
            **kwargs,
        )
        return self.fit_lbfgs(
            first["params"],
            maxiter=lbfgs_iter,
            history=first.get("telemetry", []),
        )
