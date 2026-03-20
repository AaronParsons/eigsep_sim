"""
Linear inversion tools for BLOOM-21cm sky recovery.

build_design_matrix — construct the observation matrix A from pre-computed
    masks, beam patterns, and Sun pixel indices.

svd_solve — thin-SVD least-squares solution returning the recovered sky map,
    regolith and sun scalars, SVD components, rank, and unobserved pixel mask.
"""

import numpy as np


def build_design_matrix(masks, beams, omega_B, J_SUN, npix):
    """
    Build the observation design matrix A.

    Row r = 2*(o*n_obs + i) + d corresponds to orbit o, time i, dipole d.
    Column layout: [sky pixels 0..npix-1 | T_regolith | T_sun].

    Parameters
    ----------
    masks : ndarray, shape (n_total, npix), float32
    beams : ndarray, shape (n_total, 2, npix), float32
    omega_B : ndarray, shape (n_total, 2)
    J_SUN : ndarray, shape (n_obs,), int
        HEALPix pixel of the Sun at each time (shared across orbits).
    npix : int
        Number of sky pixels.

    Returns
    -------
    A : ndarray, shape (n_total * 2, npix + 2), float64
    """
    n_total = masks.shape[0]
    n_obs = len(J_SUN)
    n_orbits = n_total // n_obs
    n_rows = n_total * 2

    A = np.zeros((n_rows, npix + 2), dtype=np.float64)

    for o in range(n_orbits):
        for i in range(n_obs):
            k = o * n_obs + i
            m_i = masks[k].astype(np.float64)
            sun_mask_i = m_i[J_SUN[i]]

            for d in range(2):
                r = 2 * k + d
                B = beams[k, d].astype(np.float64)
                OmB = omega_B[k, d]

                A[r, :npix]    = B * m_i / OmB
                A[r, npix]     = np.dot(B, 1.0 - m_i) / OmB
                A[r, npix + 1] = B[J_SUN[i]] * sun_mask_i / OmB

    return A


def svd_solve(A, y, npix, rcond=1e-6):
    """
    Least-squares sky recovery via thin SVD.

    Parameters
    ----------
    A : ndarray, shape (n_rows, npix + 2)
    y : ndarray, shape (n_rows,)
        Measurement vector (flattened antenna temperatures).
    npix : int
        Number of sky pixels (first npix columns of A are the sky part).
    rcond : float
        Singular values below rcond * sv[0] are treated as zero.

    Returns
    -------
    result : dict with keys
        sky_map    : (npix,) float64  — recovered sky temperature
                     (NaN = unobserved)
        t_regolith : float
        t_sun      : float
        U          : (n_rows, n_unknowns) — left singular vectors
        sv         : (n_unknowns,)       — singular values
        Vt         : (n_unknowns, n_unknowns) — right singular vectors
                     (transposed)
        rank       : int
        unobserved : (npix,) bool — pixels with negligible column norm
    """
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)

    sv_thresh = rcond * sv[0]
    sv_inv = np.where(sv > sv_thresh, 1.0 / np.where(sv > sv_thresh, sv, 1.0), 0.0)
    rank = int((sv > sv_thresh).sum())

    x_est = Vt.T @ (sv_inv * (U.T @ y))

    sky_map = x_est[:npix].copy()
    t_regolith = float(x_est[npix])
    t_sun = float(x_est[npix + 1])

    col_norms = np.linalg.norm(A[:, :npix], axis=0)
    unobserved = col_norms < 1e-6 * col_norms.max()
    sky_map[unobserved] = np.nan

    return {
        "sky_map":    sky_map,
        "t_regolith": t_regolith,
        "t_sun":      t_sun,
        "U":          U,
        "sv":         sv,
        "Vt":         Vt,
        "rank":       rank,
        "unobserved": unobserved,
    }
