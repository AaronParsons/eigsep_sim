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
    Type-I Anderson Acceleration for fixed-point iteration.

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

        # Need at least 2 iterates for acceleration
        if len(self.x_history) < 2:
            return x_new.astype(x_new.dtype)

        # Build the Gram matrix of residual differences
        k = len(self.x_history) - 1
        fx_diffs = np.array([self.fx_diff_history[i + 1] - self.fx_diff_history[i]
                             for i in range(k)])  # (k, n)

        # Gram matrix: G[i,j] = (fx_diff_i, fx_diff_j)
        G = fx_diffs @ fx_diffs.T  # (k, k)

        # RHS: -fx_diff_history[-1]
        rhs = -self.fx_diff_history[-1]  # (n,)
        g_rhs = fx_diffs @ rhs  # (k,)

        # Solve regularized normal equations: (G + epsilon*I) alpha = g_rhs
        try:
            alpha = np.linalg.solve(G + self.tol * np.eye(k), g_rhs)
        except np.linalg.LinAlgError:
            # Degenerate case: return unaccelerated
            return x_new.astype(x_new.dtype)

        # Accelerated iterate: x_acc = sum_i alpha_i * x_{n-k+i}
        x_acc = np.zeros_like(x)
        x_acc += (1.0 - np.sum(alpha)) * x  # Contribution from x_n
        for i in range(k):
            x_acc += alpha[i] * self.x_history[i]

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

        # Anderson accelerator
        self._aa = AndersonAccelerator(m=m_anderson)

        # Cache initial nominal beam coefficients for regularization
        self._beam_nom = None

        # Precomputed geometry (cached from init_params or fit)
        self._geom = None

    def init_params(self, times=None) -> Dict[str, np.ndarray]:
        """
        Initialize parameters with default values.

        Uses zero coefficients for sky, and initial beam coefficients.
        Precomputes geometry if times are provided.

        Parameters
        ----------
        times : list of Time, optional
            Observation times. If provided, precomputes geometry.

        Returns
        -------
        params : dict
            Initial parameters {'sky_coeffs', 'beam_coeffs'}.
        """
        sky_npix = self.fwd.sky.npix
        sky_nmodes = self.fwd.sky.nmodes
        beam_coeffs = self.fwd.beam.coeffs.astype(DTYPE_R_NPY)

        params = {
            'sky_coeffs': np.zeros((sky_npix, sky_nmodes), dtype=DTYPE_R_NPY),
            'beam_coeffs': beam_coeffs.copy(),
        }

        # Cache nominal beam for regularization
        self._beam_nom = beam_coeffs.copy()

        # Precompute geometry if times provided
        if times is not None:
            self._geom = self.fwd.precompute_geometry(times)

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
        data_flat = jnp.reshape(jnp.asarray(self._data), (-1, pred.shape[-1]))
        inv_noise_var_flat = jnp.reshape(jnp.asarray(self._inv_noise_var), (-1, pred.shape[-1]))

        # Data residual
        resid = pred_flat - data_flat
        loss = jnp.mean(inv_noise_var_flat * resid**2)

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

        sky_new = (sky_jax.ravel() + delta).reshape(sky_jax.shape)
        loss_new = float(loss_sky(sky_new))

        if loss_new < loss_before:
            params_new = params.copy()
            params_new['sky_coeffs'] = np.asarray(sky_new, dtype=DTYPE_R_NPY)
            return params_new

        # CG failed to improve; fall back to gradient descent with line search
        current_lr = 1.0
        for _ in range(20):
            sky_new = sky_jax - current_lr * grad_val
            loss_new = float(loss_sky(sky_new))
            if loss_new <= loss_before:
                params_new = params.copy()
                params_new['sky_coeffs'] = np.asarray(sky_new, dtype=DTYPE_R_NPY)
                return params_new
            current_lr *= 0.5

        return params.copy()

    def beam_cg_step(self, params: Dict[str, np.ndarray],
                     n_cg: int = 50, lam: float = 1e-4) -> Dict[str, np.ndarray]:
        """
        Beam coefficient update via Newton-CG (mirrors sky_step for the beam).

        The loss is NOT quadratic in beam_coeffs (the beam normalization
        denominator introduces nonlinearity), so this is the inner CG solve
        of a truncated Newton iteration — multiple outer calls are needed.
        Near the solution the convergence is superlinear; far away it still
        beats gradient descent by using curvature information from jax.jvp.

        Parameters
        ----------
        params : dict
        n_cg : int, optional
            Max CG iterations per call (default 50).
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
            hvp_flat, -grad_val.ravel(), maxiter=n_cg, tol=1e-3
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
        Joint sky+beam Newton-CG step.

        Treats sky_coeffs and beam_coeffs as a single flat parameter vector and
        applies one Newton step via conjugate gradient on the joint Hessian system.
        The Hessian-vector product is computed by JAX autodiff (one JVP-of-grad
        pass over the full parameter vector).

        Unlike the alternating sky_step/beam_step approach, this includes the
        off-diagonal Hessian blocks H_{sky,beam} that couple sky and beam updates.
        This breaks the trap where a converged sky absorbs beam error and makes
        beam updates appear loss-increasing.

        The loss is quadratic in sky_coeffs but nonlinear in beam_coeffs, so CG
        acts as the inner solver of a truncated Newton method.  Convergence is
        superlinear in the outer loop (one call = one Newton iteration).

        Parameters
        ----------
        params : dict
            Current parameters.
        n_cg : int, optional
            Max CG iterations per Newton step (default 100).
        lam : float, optional
            Tikhonov regularization relative to gradient magnitude (default 1e-4).

        Returns
        -------
        params_new : dict
            Updated parameters with jointly optimized sky_coeffs and beam_coeffs.
        """
        sky_jax = jnp.asarray(params['sky_coeffs'])
        beam_jax = jnp.asarray(params['beam_coeffs'])
        sky_shape, beam_shape = sky_jax.shape, beam_jax.shape
        n_sky = sky_jax.size

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

        rng_key = jax.random.PRNGKey(0)
        v_probe = jax.random.rademacher(rng_key, theta.shape, dtype=theta.dtype)
        _, h_probe = jax.jvp(grad_fn, (theta,), (v_probe,))
        h_diag_est = float(jnp.sum(h_probe * v_probe) / jnp.sum(v_probe * v_probe))
        lam_abs = lam * max(abs(h_diag_est), 1e-12) + 1e-12

        def hvp_flat(v):
            _, h = jax.jvp(grad_fn, (theta,), (v,))
            return h + lam_abs * v

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
            times=None,
            max_iter: int = 30,
            tol: float = 1e-6,
            verbose: bool = True,
            use_cg: bool = True) -> Dict:
        """
        Run calibration with Anderson-accelerated alternating sky/beam iteration.

        Each iteration:
          1. sky_step  — near-exact Newton-CG solve (quadratic in sky_coeffs)
          2. beam_cg_step (use_cg=True) or beam_step (use_cg=False)
          3. Anderson Acceleration on the beam coefficients

        Parameters
        ----------
        params : dict, optional
            Initial parameters. If None, calls init_params().
        times : list of Time, optional
            Observation times. Required if geometry not yet precomputed.
        max_iter : int, optional
            Maximum iterations (default 30).
        tol : float, optional
            Convergence tolerance on relative loss change (default 1e-6).
        verbose : bool, optional
            Print per-iteration progress (default True).
        use_cg : bool, optional
            If True (default), use beam_cg_step (Newton-CG with curvature
            information). If False, use beam_step (gradient descent with
            line search) — cheaper per iteration but slower to converge.

        Returns
        -------
        result : dict
            - 'params': final optimized parameters
            - 'losses': loss at each iteration
            - 'converged': whether tolerance was met
            - 'n_iter': iterations completed
        """
        if params is None:
            params = self.init_params(times=times)
        elif times is not None and self._geom is None:
            self._geom = self.fwd.precompute_geometry(times)

        if self._geom is None:
            raise ValueError("Provide times or call init_params(times=...) first")

        self._aa.reset()
        losses = []
        converged = False

        for iteration in range(max_iter):
            beam_old = params['beam_coeffs'].copy()

            # Sky step: near-exact linear solve (always improves sky given beam)
            params = self.sky_step(params)

            # Beam step: Newton-CG (with GD fallback) or plain gradient descent
            if use_cg:
                params = self.beam_cg_step(params)
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

            # Accept AA point only if it doesn't raise the loss
            loss_step = float(self._loss(params))
            loss_aa   = float(self._loss(params_aa))
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
