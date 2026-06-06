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
from .linear_solver import normal_solve

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
        grad_fn = jax.grad(loss_sky)
        loss_before = float(loss_sky(sky_jax))
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
        grad_fn = jax.grad(loss_beam)
        loss_before = float(loss_beam(beam_jax))
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
                    scheduled_block, schedule_reason = choose_scheduled_block()
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
                params = self.sky_step(params, step_size=sky_step_size)
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
                                new_gain = old_gain * schedule_step_gain_factor
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
                        "projected_sky_rms": self._rms(params["sky_coeffs"]),
                        "projected_beam_rms": self._rms(params["beam_coeffs"]),
                        "beam_scatter": float(np.std(params["beam_coeffs"])),
                        "beam_roughness": self._beam_roughness(
                            params["beam_coeffs"]
                        ),
                        "beam_harmonic_penalty": self._beam_harmonic_penalty(
                            params["beam_coeffs"]
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
                            "schedule_eff_sky": scheduler["eff"].get("sky"),
                            "schedule_eff_beam": scheduler["eff"].get("beam"),
                            "schedule_eff_joint": scheduler["eff"].get(
                                "joint"
                            ),
                            "schedule_eff_lbfgs": scheduler["eff"].get(
                                "lbfgs"
                            ),
                            "schedule_n_since_sky": scheduler["n_since"].get(
                                "sky"
                            ),
                            "schedule_n_since_beam": scheduler["n_since"].get(
                                "beam"
                            ),
                            "schedule_n_since_joint": scheduler["n_since"].get(
                                "joint"
                            ),
                            "schedule_n_since_lbfgs": scheduler["n_since"].get(
                                "lbfgs"
                            ),
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
                    rel = abs(losses[-2] - loss) / (abs(losses[-2]) + 1e-30)
                    print(
                        f"iter {iteration:3d}: loss = {loss:.6e}  "
                        f"rel_D = {rel:.2e}  step = {step_type}  "
                        f"dchi2/s = {entry['delta_chi2_per_sec']:.3e}"
                    )

            if iteration > 0:
                rel = abs(losses[-2] - loss) / (abs(losses[-2]) + 1e-30)
                if rel < tol:
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    converged = True
                    break
            previous_loss = loss

        return {
            "params": params,
            "losses": losses,
            "telemetry": telemetry,
            "converged": converged,
            "n_iter": iteration + 1,
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
