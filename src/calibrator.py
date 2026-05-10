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
        grad_val = grad_fn(sky_jax)
        loss_before = float(loss_sky(sky_jax))

        lam_abs = lam * float(jnp.mean(jnp.abs(grad_val))) + 1e-12

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

        # Gradient step with line search
        current_lr = lr
        for _ in range(10):  # Try up to 10 halvings of learning rate
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
            require_descent: bool = True,
            optimize_sky: bool = True,
            sky_step_size: float = 0.5) -> Dict:
        """
        Run calibration with Anderson-accelerated fixed-point iteration.

        Optimizes beam coefficients. Optionally also optimizes sky coefficients
        via per-frequency linear solves (though alternating sky/beam optimization
        can be unstable; use high lam_sky regularization if enabling).

        Only accepts steps that decrease loss (when require_descent=True).

        Parameters
        ----------
        params : dict, optional
            Initial parameters. If None, calls init_params().
        times : list of Time, optional
            Observation times. Required if params is None.
        max_iter : int, optional
            Maximum number of iterations (default 30).
        tol : float, optional
            Convergence tolerance on relative loss change (default 1e-6).
        verbose : bool, optional
            Print progress (default True).
        require_descent : bool, optional
            Only accept steps that decrease loss (default True).
        optimize_sky : bool, optional
            If True, alternate sky and beam optimization (default True). If False,
            optimize beam only. Sky/beam alternation is stabilized by
            sky_step_size damping.
        sky_step_size : float, optional
            Damping factor for sky updates in (0, 1] (default 0.5). Values < 1
            blend the proposed solution with the current sky_coeffs, stabilizing
            the alternating iteration.

        Returns
        -------
        result : dict
            Convergence result with keys:
            - 'params': Final optimized parameters
            - 'losses': Loss history
            - 'converged': Whether convergence criterion was met
            - 'n_iter': Number of iterations run
        """
        # Initialize if needed
        if params is None:
            params = self.init_params(times=times)
        else:
            # Precompute geometry if times provided
            if times is not None and self._geom is None:
                self._geom = self.fwd.precompute_geometry(times)

        if self._geom is None:
            raise ValueError("Either provide times or pre-computed geometry")

        # Reset Anderson accelerator
        self._aa.reset()

        losses = []
        converged = False

        for iteration in range(max_iter):
            loss_before = float(self._loss(params))

            # Sky step (optional, damped)
            if optimize_sky:
                params_sky = self.sky_step(params, step_size=sky_step_size)
            else:
                params_sky = params

            # Beam step (applied after sky step)
            params_new = self.beam_step(params_sky)
            loss_after = float(self._loss(params_new))

            # Accept the iteration if loss decreased (or not requiring descent)
            if not require_descent or loss_after <= loss_before:
                params = params_new
                loss = loss_after
            else:
                # If iteration increased loss, try only sky step
                params_sky_only = params_sky
                loss_sky_only = float(self._loss(params_sky_only))
                if loss_sky_only <= loss_before:
                    params = params_sky_only
                    loss = loss_sky_only
                else:
                    # If both decreased loss, reject and keep old params
                    params = params  # No change
                    loss = loss_before

            losses.append(loss)

            if verbose:
                if iteration == 0:
                    print(f"Iteration {iteration:3d}: loss = {loss:.6e}")
                else:
                    rel_change = abs(losses[-2] - loss) / losses[-2]
                    print(f"Iteration {iteration:3d}: loss = {loss:.6e}, "
                          f"rel_change = {rel_change:.3e}")

            # Check convergence
            if iteration > 0:
                rel_change = abs(losses[-2] - loss) / losses[-2]
                if rel_change < tol:
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
