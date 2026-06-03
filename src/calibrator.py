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

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict

from .const import DTYPE_R_NPY, DTYPE_R_JAX
from .simulate import ForwardModel
from .linear_solver import normal_solve


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
        x_diffs = np.column_stack([
            self.x_history[i + 1] - self.x_history[i] for i in range(k)
        ])
        fx_diffs = np.column_stack([
            self.fx_diff_history[i + 1] - self.fx_diff_history[i]
            for i in range(k)
        ])
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
    """

    def __init__(self, fwd: ForwardModel, data: np.ndarray,
                 inv_noise_var: Optional[np.ndarray] = None,
                 m_anderson: int = 5,
                 lam_beam: float = 0.01,
                 lam_sky: float = 0.0):
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
        self._data_flat_jax = jnp.reshape(
            jnp.asarray(self._data, dtype=DTYPE_R_JAX),
            (-1, self._data.shape[-1]),
        )
        self._inv_noise_var_flat_jax = jnp.reshape(
            jnp.asarray(self._inv_noise_var, dtype=DTYPE_R_JAX),
            (-1, self._inv_noise_var.shape[-1]),
        )

        # Anderson accelerator
        self._aa = AndersonAccelerator(m=m_anderson)

        # Cache initial nominal beam coefficients for regularization
        self._beam_nom = None

        # Precomputed geometry (cached from init_params or fit)
        self._geom = None

    def _resolve_geom(self, times=None, rots=None, body_rots=None,
                      geom=None, sky_mask=None):
        """Compute and cache geometry from whichever source is provided."""
        if geom is not None:
            self._geom = geom
        elif rots is not None:
            self._geom = self.fwd.precompute_geometry(
                rots=rots, body_rots=body_rots, sky_mask=sky_mask)
        elif times is not None:
            self._geom = self.fwd.precompute_geometry(
                times=times, sky_mask=sky_mask)

    def init_params(self, times=None, rots=None, body_rots=None,
                    geom=None, sky_mask=None) -> Dict[str, np.ndarray]:
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
            'sky_coeffs': np.zeros((sky_npix, sky_nmodes), dtype=DTYPE_R_NPY),
            'beam_coeffs': beam_coeffs.copy(),
        }
        self._beam_nom = beam_coeffs.copy()

        self._resolve_geom(times=times, rots=rots, body_rots=body_rots,
                           geom=geom, sky_mask=sky_mask)
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
            params['sky_coeffs'],
            params['beam_coeffs'],
            geom=self._geom
        )

        # Reshape pred and data to (ntimes*n_dipoles, nfreq) for consistent loss computation
        pred_flat = jnp.reshape(pred, (-1, pred.shape[-1]))
        # Data residual
        resid = pred_flat - self._data_flat_jax
        loss = jnp.mean(self._inv_noise_var_flat_jax * resid**2)

        # Beam regularization (ridge toward nominal)
        if self._lam_beam > 0 and self._beam_nom is not None:
            beam_nom_jax = jnp.asarray(self._beam_nom)
            beam_diff = params['beam_coeffs'] - beam_nom_jax
            loss = loss + self._lam_beam * jnp.mean(beam_diff**2)

        # Sky regularization (ridge toward zero)
        if self._lam_sky > 0:
            loss = loss + self._lam_sky * jnp.mean(params['sky_coeffs']**2)

        return loss

    def sky_step(self, params: Dict[str, np.ndarray],
                 n_cg: int = 50, lam: float = 1e-4,
                 rcond: float = 1e-6,
                 step_size: float = 1.0) -> Dict[str, np.ndarray]:
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
            p = {'sky_coeffs': sky_coeffs, 'beam_coeffs': params['beam_coeffs']}
            return self._loss(p)

        sky_jax = jnp.asarray(params['sky_coeffs'])
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
        delta, _ = jax.scipy.sparse.linalg.cg(hvp_flat, b, maxiter=n_cg, tol=1e-3)

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
            params_new['sky_coeffs'] = np.asarray(sky_new, dtype=DTYPE_R_NPY)
            return params_new

        # CG failed to improve; fall back to gradient descent with line search
        current_lr = step_size
        for _ in range(20):
            sky_new = (sky_jax.ravel() - current_lr * grad_val.ravel()).reshape(sky_jax.shape)
            loss_new = float(loss_sky(sky_new))
            if loss_new <= loss_before:
                params_new = params.copy()
                params_new['sky_coeffs'] = np.asarray(sky_new, dtype=DTYPE_R_NPY)
                return params_new
            current_lr *= 0.5

        return params.copy()

    def beam_cg_step(self, params: Dict[str, np.ndarray],
                     n_cg: int = 50, lam: float = 1e-4,
                     cg_tol: float = 1e-3) -> Dict[str, np.ndarray]:
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
            p = {'sky_coeffs': params['sky_coeffs'], 'beam_coeffs': beam_coeffs}
            return self._loss(p)

        beam_jax = jnp.asarray(params['beam_coeffs'])
        grad_fn = jax.grad(loss_beam)
        loss_before = float(loss_beam(beam_jax))
        grad_val = grad_fn(beam_jax)

        # Use H-diagonal estimate as Tikhonov scale: v^T H v / ||v||^2 ≈ trace(H)/n.
        # This avoids the gradient-magnitude scaling bug: ||grad|| >> H_diag when
        # far from minimum, which makes gradient-scaled lam_abs >> H and causes CG
        # to ignore curvature (reduces to over-damped gradient descent).
        rng_key = jax.random.PRNGKey(0)
        v_probe = jax.random.rademacher(rng_key, beam_jax.shape, dtype=beam_jax.dtype)
        _, h_probe = jax.jvp(grad_fn, (beam_jax,), (v_probe,))
        h_diag_est = float(jnp.sum(h_probe * v_probe) / jnp.sum(v_probe * v_probe))
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
            params_new['beam_coeffs'] = np.asarray(beam_new, dtype=DTYPE_R_NPY)
            return params_new

        # CG didn't improve; fall back to gradient descent with adaptive lr.
        current_lr = 0.01 / (float(jnp.max(jnp.abs(grad_val))) + 1e-30)
        for _ in range(30):
            beam_new = beam_jax - current_lr * grad_val
            loss_new = float(loss_beam(beam_new))
            if loss_new <= loss_before:
                params_new = params.copy()
                params_new['beam_coeffs'] = np.asarray(beam_new, dtype=DTYPE_R_NPY)
                return params_new
            current_lr *= 0.5

        return params.copy()

    def joint_step(self, params: Dict[str, np.ndarray],
                   n_cg: int = 100, lam: float = 1e-4) -> Dict[str, np.ndarray]:
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
        sky_jax = jnp.asarray(params['sky_coeffs'])
        beam_jax = jnp.asarray(params['beam_coeffs'])
        sky_shape, beam_shape = sky_jax.shape, beam_jax.shape
        n_sky = sky_jax.size
        n_beam = beam_jax.size

        def pack(sky, beam):
            return jnp.concatenate([sky.ravel(), beam.ravel()])

        def unpack(theta):
            return theta[:n_sky].reshape(sky_shape), theta[n_sky:].reshape(beam_shape)

        def loss_joint(theta):
            s, b = unpack(theta)
            return self._loss({'sky_coeffs': s, 'beam_coeffs': b})

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
        zeros_sky  = jnp.zeros(n_sky,  dtype=theta.dtype)

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
            reg = jnp.concatenate([lam_abs_sky * v[:n_sky], lam_abs_beam * v[n_sky:]])
            return h + reg

        delta, _ = jax.scipy.sparse.linalg.cg(
            hvp_flat, -grad_val, maxiter=n_cg, tol=1e-3
        )

        theta_new = theta + delta
        loss_new = float(loss_joint(theta_new))

        if loss_new < loss_before:
            sky_new, beam_new = unpack(theta_new)
            return {
                'sky_coeffs':  np.asarray(sky_new,  dtype=DTYPE_R_NPY),
                'beam_coeffs': np.asarray(beam_new, dtype=DTYPE_R_NPY),
            }

        # Newton step didn't improve; fall back to gradient descent with line search.
        current_lr = 0.01 / (float(jnp.max(jnp.abs(grad_val))) + 1e-30)
        for _ in range(30):
            theta_new = theta - current_lr * grad_val
            loss_new = float(loss_joint(theta_new))
            if loss_new <= loss_before:
                sky_new, beam_new = unpack(theta_new)
                return {
                    'sky_coeffs':  np.asarray(sky_new,  dtype=DTYPE_R_NPY),
                    'beam_coeffs': np.asarray(beam_new, dtype=DTYPE_R_NPY),
                }
            current_lr *= 0.5

        return params.copy()

    def beam_step(self, params: Dict[str, np.ndarray],
                  lr: float = 0.01, line_search: bool = True) -> Dict[str, np.ndarray]:
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
            p['beam_coeffs'] = beam_coeffs
            return self._loss(p)

        # Compute gradient
        grad = jax.grad(loss_beam)(params['beam_coeffs'])
        loss_before = float(loss_beam(params['beam_coeffs']))

        # Normalize lr so the largest parameter change is lr (not lr * ||grad||).
        # Without this, a fixed lr=0.01 with gradient of O(1e8) gives steps of
        # O(1e6), which always overshoot and cause the line search to exhaust
        # all 10 halvings before reaching a useful step size.
        grad_inf = float(jnp.max(jnp.abs(grad)))
        current_lr = lr / (grad_inf + 1e-30)

        # Gradient step with line search
        for _ in range(30):  # up to 30 halvings to handle large dynamic range
            params_new = params.copy()
            params_new['beam_coeffs'] = params['beam_coeffs'] - current_lr * grad
            loss_new = float(loss_beam(params_new['beam_coeffs']))

            # Accept step if loss decreased (or line search disabled)
            if not line_search or loss_new <= loss_before:
                return params_new

            # Halve learning rate and try again
            current_lr *= 0.5

        # If all attempts failed, return original params
        return params.copy()

    def fit(self, params: Optional[Dict[str, np.ndarray]] = None,
            times=None, rots=None, body_rots=None, geom=None, sky_mask=None,
            max_iter: int = 30,
            tol: float = 1e-6,
            verbose: bool = True,
            use_cg: bool = False,
            use_joint: bool = False,
            sky_step_size: float = 1.0,
            beam_cg_niter: int = 50,
            beam_cg_tol: float = 1e-3) -> Dict:
        """
        Run calibration with Anderson-accelerated alternating sky/beam iteration.

        Each iteration:
          1. sky_step  — near-exact Newton-CG solve (quadratic in sky_coeffs)
          2. beam_cg_step (use_cg=True), joint_step (use_joint=True), or beam_step
          3. Anderson Acceleration on the beam coefficients

        Parameters
        ----------
        params : dict, optional
            Initial parameters. If None, calls init_params().
        times : list of Time, optional
            Observation epochs. Mutually exclusive with rots.
        rots : list of (3, 3) ndarray, optional
            Pre-computed gal→top rotation matrices (mutually exclusive with times).
        body_rots : list of (3, 3) ndarray, optional
            Per-step top→body rotations.
        geom : dict, optional
            Pre-computed geometry from ForwardModel.precompute_geometry().
            Takes priority over times/rots when provided.
        sky_mask : ndarray of bool, optional
            Pixel-reduction mask from ForwardModel.build_sky_mask().
        max_iter : int, optional
            Maximum iterations (default 30).
        tol : float, optional
            Convergence tolerance on relative loss change (default 1e-6).
        verbose : bool, optional
            Print per-iteration progress (default True).
        use_cg : bool, optional
            If True, use beam_cg_step (Newton-CG) for the beam step. Default
            False: use beam_step (gradient descent with line search), which is
            more stable for joint sky+beam recovery.
        use_joint : bool, optional
            If True, replace sky_step + beam_step with a single joint_step that
            optimizes sky and beam simultaneously via Newton-CG with block-diagonal
            regularization.  This uses the off-diagonal Hessian coupling to avoid
            the alternating trap (sky absorbing beam error), enabling beam
            convergence even when the alternating steps stall.
            Overrides use_cg when True.
        sky_step_size : float, optional
            Damping factor for the sky Newton step in alternating mode (default
            1.0 = full step). Ignored when use_joint=True.
        beam_cg_niter : int, optional
            Inner CG iterations for the beam step when use_cg=True (default 50).
            Use small values such as 5-10 for a fast truncated Newton beam
            update.
        beam_cg_tol : float, optional
            Inner CG tolerance for the beam step when use_cg=True (default 1e-3).

        Returns
        -------
        result : dict
            - 'params': final optimized parameters
            - 'losses': loss at each iteration
            - 'converged': whether tolerance was met
            - 'n_iter': iterations completed
        """
        if params is None:
            params = self.init_params(times=times, rots=rots,
                                      body_rots=body_rots, geom=geom,
                                      sky_mask=sky_mask)
        elif self._geom is None:
            self._resolve_geom(times=times, rots=rots, body_rots=body_rots,
                               geom=geom, sky_mask=sky_mask)

        if self._geom is None:
            raise ValueError(
                "Provide times, rots, or geom to specify observation geometry"
            )

        self._aa.reset()
        losses = []
        converged = False

        for iteration in range(max_iter):
            beam_old = params['beam_coeffs'].copy()

            if use_joint:
                # Joint Newton-CG: sky + beam updated simultaneously.
                # Block-diagonal regularization prevents the sky Hessian (O(1e-5))
                # and beam Hessian (O(1e4)) from interfering with each other.
                params = self.joint_step(params)
            else:
                # Alternating: sky first (near-exact quadratic solve), then beam.
                params = self.sky_step(params, step_size=sky_step_size)
                if use_cg:
                    params = self.beam_cg_step(
                        params, n_cg=beam_cg_niter, cg_tol=beam_cg_tol
                    )
                else:
                    params = self.beam_step(params)

            # Anderson Acceleration on beam coefficients.
            # The fixed-point residual is (new_beam - old_beam); AA extrapolates
            # toward the fixed point using the history of residuals.
            beam_new = params['beam_coeffs']
            beam_res = beam_new - beam_old
            beam_acc = self._aa.apply(beam_old, beam_res)
            params_aa = params.copy()
            params_aa['beam_coeffs'] = np.asarray(beam_acc, dtype=DTYPE_R_NPY)

            # Accept AA point only if it doesn't raise the loss. The first
            # call returns the ordinary fixed-point update, so avoid evaluating
            # the same loss twice before acceleration has enough history.
            loss_step = float(self._loss(params))
            if len(self._aa.x_history) < 2:
                loss = loss_step
            else:
                loss_aa = float(self._loss(params_aa))
                if loss_aa <= loss_step:
                    params = params_aa
                    loss = loss_aa
                else:
                    loss = loss_step

            losses.append(loss)

            if verbose:
                if iteration == 0:
                    print(f"iter {iteration:3d}: loss = {loss:.6e}")
                else:
                    rel = abs(losses[-2] - loss) / (abs(losses[-2]) + 1e-30)
                    print(f"iter {iteration:3d}: loss = {loss:.6e}  "
                          f"rel_Δ = {rel:.2e}")

            if iteration > 0:
                rel = abs(losses[-2] - loss) / (abs(losses[-2]) + 1e-30)
                if rel < tol:
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    converged = True
                    break

        return {
            'params': params,
            'losses': losses,
            'converged': converged,
            'n_iter': iteration + 1,
        }
