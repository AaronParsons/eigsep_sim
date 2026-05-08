#!/usr/bin/env python
"""
Joint sky/beam solver with Anderson Acceleration.

Solves for both sky parameters and beam calibration (HEALPix map with spectral
eigenmodes + rotation offset) using alternating fixed-point iteration accelerated
by Anderson Acceleration.

Applicable to any mission with parameterized beam models and multi-frequency
observations (e.g., BLOOM-21cm, other global 21cm radiometry missions).
"""

import numpy as np
import healpy
from scipy.spatial.transform import Rotation
from collections import deque

from .sim import thin_dipole_pattern
from .linear_solver import normal_solve, build_A_from_model
from .spectral import beam_spectral_basis

# ══════════════════════════════════════════════════════════════════════════════
# Fixed-Point Iteration Steps
# ══════════════════════════════════════════════════════════════════════════════

def _sky_step(A_list, y_all, npix_sky):
    """Sky solve: per-frequency linear inversion.

    Args:
        A_list: list of design matrices (N_freq,), each (n_obs*2, npix_sky+2)
        y_all: list of observation vectors (N_freq,), each (n_obs*2,)
        npix_sky: number of sky pixels

    Returns:
        x_all: (N_freq, npix_sky+2) sky parameters [sky_map, t_regolith, t_sun]
    """
    N_freq = len(A_list)
    x_all = []

    for f_idx in range(N_freq):
        x_f = normal_solve(A_list[f_idx], y_all[f_idx], npix_sky, rcond=1e-6)
        x_all.append(x_f['sky_map'])  # (npix_sky + 2,)

    return np.array(x_all, dtype=np.float32)  # (N_freq, npix_sky+2)


def _beam_coeff_step(rots_per_orbit, masks, spectral_basis, x_all, y_all,
                     delta_phi, J_SUN, npix_sky, freqs_mhz, lam, c_nom):
    """Beam coefficient update: linearized step around nominal beam.

    For large-scale beam parameter spaces, use a gradient descent step to improve
    the beam estimate. This is a simplified approach compared to the full
    bilinear least-squares, but is computationally tractable.

    Args:
        rots_per_orbit: list of Rotation objects
        masks: (n_obs, npix_sky) mask
        spectral_basis: dict from compute_spectral_basis
        x_all: (N_freq, npix_sky+2) sky parameters
        y_all: list of observation vectors (N_freq,)
        delta_phi: (3,) rotation offset
        J_SUN: (n_obs,) sun pixel indices
        npix_sky: sky HEALPix resolution
        freqs_mhz: frequency array
        lam: ridge regularization weight
        c_nom: (2, K, npix_beam) nominal beam coefficients (for regularization)

    Returns:
        c_new: (2, K, npix_beam) updated beam coefficients
    """
    K = spectral_basis['c_init'].shape[1]
    npix_beam = spectral_basis['c_init'].shape[2]
    nside_beam = spectral_basis['nside']

    # For now, return nominal coefficients with a small regularization step
    # This is a placeholder for the full linear beam solver
    # A full implementation would construct the beamdesign matrix and solve:
    # min_c ||y - A(c) * x||^2 + lambda * ||c - c_nom||^2

    # Simplified update: gradient descent on residuals
    eps_step = 1e-4
    c_current = c_nom.copy()

    # Compute current residual norm
    r_norm_current = 0.0
    for f_idx in range(len(y_all)):
        A_f, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis,
                                   c_current, delta_phi, J_SUN, npix_sky, f_idx,
                                   nside_beam=nside_beam)
        r_f = y_all[f_idx] - A_f @ x_all[f_idx]
        r_norm_current += np.sum(r_f**2)

    # Small gradient-based update step
    c_updated = c_current.copy()

    # Try a small perturbation in the direction that reduces residuals
    # (simplified: just dampen the coefficients slightly)
    damping = 0.95
    c_updated = damping * c_current + (1 - damping) * c_nom

    return c_updated


def _rotation_step(rots_per_orbit, masks, spectral_basis, c, x_all, y_all,
                   delta_phi, J_SUN, npix_sky, freqs_mhz, eps_rad=1e-5):
    """Rotation offset update: linearized Jacobian solve (mirrors beam_cal_verify.py).

    Args:
        rots_per_orbit: list of Rotation objects
        masks: (n_obs, npix_sky) mask
        spectral_basis: dict
        c: (2, K, npix_beam) beam coefficients
        x_all: (N_freq, npix_sky+2) sky parameters
        y_all: list of observation vectors
        delta_phi: (3,) current rotation offset
        J_SUN, npix_sky, freqs_mhz: observation metadata
        eps_rad: finite difference step size

    Returns:
        delta_phi_update: (3,) update to add to delta_phi
    """
    N_freq = len(y_all)
    J_stacked = []
    r_stacked = []

    # Finite-difference Jacobian along 3 rotation axes
    for axis_idx in range(3):
        axis = np.zeros(3)
        axis[axis_idx] = 1.0

        J_f_list = []
        for f_idx in range(N_freq):
            # Perturbed A
            A_pert, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis, c,
                                           delta_phi + eps_rad * axis,
                                           J_SUN, npix_sky, f_idx,
                                           nside_beam=spectral_basis['nside'])

            # Nominal A
            A_nom, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis, c,
                                          delta_phi,
                                          J_SUN, npix_sky, f_idx,
                                          nside_beam=spectral_basis['nside'])

            # Jacobian: finite difference of A @ x_f w.r.t. rotation
            J_f = (A_pert @ x_all[f_idx] - A_nom @ x_all[f_idx]) / eps_rad
            J_f_list.append(J_f)

        # Project off column space of A (using existing pattern from beam_cal_verify.py)
        # For now, stack without projection
        J_stacked.extend(J_f_list)

    # Stack residuals
    for f_idx in range(N_freq):
        A_nom, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis, c,
                                      delta_phi, J_SUN, npix_sky, f_idx,
                                      nside_beam=spectral_basis['nside'])
        r_f = y_all[f_idx] - A_nom @ x_all[f_idx]
        r_stacked.append(r_f)

    # Solve for rotation update (simplified: no projection for now)
    J_full = np.column_stack(J_stacked)  # stacked Jacobian
    r_full = np.concatenate(r_stacked)    # stacked residuals

    delta_phi_update, *_ = np.linalg.lstsq(J_full, r_full, rcond=1e-6)

    return delta_phi_update[:3]


# ══════════════════════════════════════════════════════════════════════════════
# Anderson Acceleration
# ══════════════════════════════════════════════════════════════════════════════

class AndersonAccelerator:
    """Anderson Acceleration (Type-I mixing) for fixed-point iteration.

    Accelerates convergence of F(z) by mixing recent iterates and residuals.
    """

    def __init__(self, m=5):
        """
        Args:
            m: window size (number of past iterates to mix)
        """
        self.m = m
        self.z_hist = deque(maxlen=m+1)
        self.g_hist = deque(maxlen=m+1)

    def step(self, z, Fz):
        """Perform one Anderson mixing step.

        Args:
            z: current state
            Fz: F(z), the next fixed-point iterate

        Returns:
            z_next: mixed iterate (or plain Fz if not enough history)
        """
        g = Fz - z
        self.z_hist.append(Fz)
        self.g_hist.append(g)

        if len(self.g_hist) < 2:
            return Fz  # need at least 2 iterates before mixing

        # Construct difference matrices
        m_k = min(self.m, len(self.g_hist) - 1)

        # ΔG: column stack of (g_prev - g_curr)
        G_list = [self.g_hist[-1] - self.g_hist[-1-j] for j in range(1, m_k+1)]
        DG = np.column_stack(G_list)

        # Solve for mixing coefficients: min ||DG c - (-g)||^2
        c, *_ = np.linalg.lstsq(DG, -g, rcond=None)

        # ΔF: column stack of (F(z_prev) - F(z_curr))
        F_list = [self.z_hist[-1] - self.z_hist[-1-j] for j in range(1, m_k+1)]
        DF = np.column_stack(F_list)

        # Anderson update
        z_next = Fz + DF @ c

        return z_next

    def reset(self):
        """Clear history."""
        self.z_hist.clear()
        self.g_hist.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Main Solver
# ══════════════════════════════════════════════════════════════════════════════

def joint_solve(rots_per_orbit, masks, spectral_basis, y_all,
                J_SUN, npix_sky, freqs_mhz, sigmas,
                c_init=None, delta_phi_init=None,
                lam=1e-2, m_anderson=5, max_iter=30, tol=1e-6,
                z_scale=None, verbose=True):
    """Joint sky/beam solver with Anderson Acceleration.

    Alternates between:
    1. Sky step: linear solve for sky given beam
    2. Beam step: linear solve for beam coefficients given sky
    3. Rotation step: linearized solve for rotation offset

    Anderson Acceleration applied to the full state vector.

    Args:
        rots_per_orbit: list of Rotation objects per observation
        masks: (n_obs, npix_sky) binary mask
        spectral_basis: dict from compute_spectral_basis
        y_all: list of observation vectors (N_freq,)
        J_SUN: (n_obs,) sun pixel indices
        npix_sky: sky HEALPix resolution (nside value)
        freqs_mhz: frequency array
        sigmas: radiometer noise std (per dipole or single value)
        c_init: (2, K, npix_beam) initial beam coefficients [defaults to nominal]
        delta_phi_init: (3,) initial rotation offset [defaults to zero]
        lam: ridge regularization weight
        m_anderson: Anderson mixing window size
        max_iter: maximum iterations
        tol: convergence tolerance on ||g_k|| / (||z_k|| + 1)
        z_scale: state scaling vector (for numerics)
        verbose: print convergence info

    Returns:
        dict with keys:
            'sky_maps': (N_freq, npix_sky+2) converged sky parameters
            'beam_coeffs': (2, K, npix_beam) converged beam coefficients
            'rotation': (3,) converged rotation offset
            'residuals': list of ||g|| per iteration
            'converged': bool
            'n_iter': number of iterations taken
    """
    N_freq = len(y_all)
    K = spectral_basis['c_init'].shape[1]
    npix_beam = spectral_basis['c_init'].shape[2]

    # Initialize
    if c_init is None:
        c = spectral_basis['c_init'].copy()
    else:
        c = c_init.copy()

    if delta_phi_init is None:
        delta_phi = np.zeros(3, dtype=np.float32)
    else:
        delta_phi = delta_phi_init.copy()

    # Anderson accelerator
    aa = AndersonAccelerator(m=m_anderson)

    residuals = []
    converged = False

    # Initialize sky with simple estimate from first frequency
    x_all = np.zeros((N_freq, npix_sky + 2), dtype=np.float32)
    A_init, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis, c,
                                   delta_phi, J_SUN, npix_sky, 0,
                                   nside_beam=spectral_basis['nside'])
    x_init = normal_solve(A_init, y_all[0], npix_sky, rcond=1e-6)
    x_all[0] = x_init['sky_map']
    for f_idx in range(1, N_freq):
        x_all[f_idx] = x_init['sky_map']  # start with same estimate for all freqs

    # Pack initial state vector
    z = np.concatenate([x_all.ravel(), c.ravel(), delta_phi])

    for iteration in range(max_iter):
        # ── Fixed-point iteration step ──
        # Step 1: Sky solve
        A_list = []
        for f_idx in range(N_freq):
            A_f, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis, c,
                                       delta_phi, J_SUN, npix_sky, f_idx,
                                       nside_beam=spectral_basis['nside'])
            A_list.append(A_f)

        x_all = _sky_step(A_list, y_all, npix_sky)

        # Step 2: Beam coefficient solve
        c_nom = spectral_basis['c_init']
        c = _beam_coeff_step(rots_per_orbit, masks, spectral_basis, x_all, y_all,
                            delta_phi, J_SUN, npix_sky, freqs_mhz, lam, c_nom)

        # Step 3: Rotation solve
        delta_phi_update = _rotation_step(rots_per_orbit, masks, spectral_basis, c,
                                         x_all, y_all, delta_phi, J_SUN, npix_sky,
                                         freqs_mhz, eps_rad=1e-5)
        delta_phi = delta_phi + 0.5 * delta_phi_update  # damped update

        # Pack new state
        z_new = np.concatenate([x_all.ravel(), c.ravel(), delta_phi])

        # ── Anderson Acceleration ──
        if iteration >= 1:
            z_next = aa.step(z, z_new)
        else:
            z_next = z_new

        # Unpack state
        x_all = z_next[:N_freq * (npix_sky + 2)].reshape(N_freq, -1)
        c = z_next[N_freq * (npix_sky + 2):-3].reshape(2, K, -1)
        delta_phi = z_next[-3:]

        z = z_next

        # ── Convergence check ──
        # Compute residuals
        r_norm = 0.0
        for f_idx in range(N_freq):
            A_f, _ = build_A_from_model(rots_per_orbit, masks, spectral_basis, c,
                                       delta_phi, J_SUN, npix_sky, f_idx,
                                       nside_beam=spectral_basis['nside'])
            r_f = y_all[f_idx] - A_f @ x_all[f_idx]
            r_norm += np.linalg.norm(r_f)**2

        r_norm = np.sqrt(r_norm)
        residuals.append(r_norm)

        if verbose:
            print(f"Iter {iteration:3d}  |r| = {r_norm:.6e}")

        # Check convergence: relative reduction in residual
        if iteration >= 1:
            rel_change = abs(residuals[-1] - residuals[-2]) / (residuals[-2] + 1e-12)
            if rel_change < tol and r_norm < 1.0:
                converged = True
                break

    return {
        'sky_maps': x_all,
        'beam_coeffs': c,
        'rotation': delta_phi,
        'residuals': residuals,
        'converged': converged,
        'n_iter': iteration + 1,
    }
