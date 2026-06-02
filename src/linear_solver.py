"""
Linear inversion tools for global 21cm radiometry.

build_design_matrix          — per-pixel observation matrix A (npix + 2 or npix + 4 columns)
build_monopole_design_matrix — 4-column matrix for direct monopole recovery
build_A_from_model           — design matrix from parameterized HEALPix beam model
normal_solve                 — fast least-squares via normal equations (A^T A eigh)
svd_solve                    — thin-SVD least-squares for the per-pixel system
monopole_lstsq               — least-squares solve for (T_mono, T_reg, T_sun)
"""

import numpy as np
import healpy
from scipy.spatial.transform import Rotation

from .recovery import build_surface_design_matrix


def build_design_matrix(masks, beams, omega_B, J_SUN, npix, include_t_rx=True):
    """Build the legacy sky, regolith, Sun, and receiver-offset matrix.

    Row ``r = 2 * (orbit * n_obs + time) + dipole``. Column ordering is
    ``[sky pixels | T_regolith | T_sun | optional T_rx per dipole]``.

    This compatibility wrapper retains the historical API while delegating
    generic sky, surface, source, and receiver-column construction to
    :func:`eigsep_sim.recovery.build_surface_design_matrix`.
    """
    masks = np.asarray(masks, dtype=np.float64)
    beams = np.asarray(beams, dtype=np.float64)
    omega_B = np.asarray(omega_B, dtype=np.float64)
    if beams.shape[2] != npix:
        raise ValueError(
            f"beams have {beams.shape[2]} pixels, inconsistent with npix={npix}"
        )
    n_total, _, _ = beams.shape
    n_obs = len(J_SUN)
    if n_obs == 0 or n_total % n_obs != 0:
        raise ValueError("J_SUN length must evenly divide the observation count")
    weights = beams / omega_B[:, :, None]
    sun_pixels = np.tile(np.asarray(J_SUN, dtype=int), n_total // n_obs)
    sun_beam = weights[np.arange(n_total), :, sun_pixels]
    sun_mask = masks[np.arange(n_total), sun_pixels]
    sun_column = sun_beam * sun_mask[:, None]
    return build_surface_design_matrix(
        weights,
        masks,
        source_columns={"sun": sun_column},
        include_receiver_offsets=include_t_rx,
    )


def build_monopole_design_matrix(masks, beams, omega_B, J_SUN):
    """
    Build a 4-column design matrix for direct monopole recovery.

    Treats the sky as a single monopole temperature rather than resolving
    individual pixels.  Column layout:
    [T_monopole | T_regolith | T_sun | T_rx_diff].

    ``T_rx_diff`` is the only independently constrained receiver-temperature
    degree of freedom.  Because ``col_mono + col_reg = 1`` (sky + regolith
    fractions always sum to unity) and ``col_rx0 + col_rx1 = 1`` (every row
    belongs to exactly one dipole), there is an **exact null space**:
    shifting T_mono and T_reg up by C while shifting both T_rx values down by
    C leaves every measurement unchanged.  As a result the *common-mode*
    receiver temperature (mean of the two dipoles) is degenerate with the sky
    and regolith monopoles; only the *differential* T_rx_1 − T_rx_0 is
    determined from the data alone.

    Row numbering matches :func:`build_design_matrix`.

    Parameters
    ----------
    masks    : ndarray, shape (n_total, npix), float32
    beams    : ndarray, shape (n_total, 2, npix), float32
    omega_B  : ndarray, shape (n_total, 2)
    J_SUN    : ndarray, shape (n_obs,), int

    Returns
    -------
    A : ndarray, shape (n_total * 2, 4), float64
    """
    n_total = masks.shape[0]
    n_obs   = len(J_SUN)
    n_rows  = n_total * 2
    n_orbits = n_total // n_obs

    A = np.zeros((n_rows, 4), dtype=np.float64)

    for o in range(n_orbits):
        for i in range(n_obs):
            k = o * n_obs + i
            m   = masks[k].astype(np.float64)
            sun = m[J_SUN[i]]

            for d in range(2):
                r   = 2 * k + d
                B   = beams[k, d].astype(np.float64)
                OmB = omega_B[k, d]

                A[r, 0] = np.dot(B, m) / OmB        # monopole
                A[r, 1] = np.dot(B, 1.0 - m) / OmB  # regolith
                A[r, 2] = B[J_SUN[i]] * sun / OmB   # sun
                # T_rx_diff column: the recovered parameter equals T_rx_1 - T_rx_0.
                # T_rx_mean = (T_rx_0+T_rx_1)/2 is absorbed into T_mono and T_reg.
                A[r, 3] = 0.5 if d == 1 else -0.5

    return A


def monopole_lstsq(A, y, rcond=1e-10):
    """
    Least-squares solve for (T_monopole, T_regolith, T_sun, T_rx_diff).

    The recovered ``t_rx_diff`` equals T_rx_1 − T_rx_0 (the full inter-dipole
    difference).  The common-mode receiver temperature
    (mean of the two dipoles) is degenerate with T_monopole + T_regolith and
    cannot be determined from the data alone; it is absorbed into those
    estimates via the minimum-norm SVD solution.

    Parameters
    ----------
    A : ndarray, shape (n_rows, 4)
        From :func:`build_monopole_design_matrix`.
    y : ndarray, shape (n_rows,)
    rcond : float
        Threshold for zeroing small singular values (relative to largest).

    Returns
    -------
    result : dict with keys
        t_mono      : float — recovered sky monopole + common T_rx [K]
        t_regolith  : float
        t_sun       : float
        t_rx_diff   : float — T_rx_1 - T_rx_0 [K]
        sv          : ndarray — singular values of A
        rank        : int
    """
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    sv_thresh = rcond * sv[0]
    sv_inv = np.where(sv > sv_thresh, 1.0 / np.where(sv > sv_thresh, sv, 1.0), 0.0)
    rank = int((sv > sv_thresh).sum())
    x = Vt.T @ (sv_inv * (U.T @ y))
    return {
        "t_mono":     float(x[0]),
        "t_regolith": float(x[1]),
        "t_sun":      float(x[2]),
        "t_rx_diff":  float(x[3]),
        "sv":         sv,
        "rank":       rank,
    }


def normal_solve(A, y, npix, rcond=1e-6):
    """
    Least-squares sky recovery via normal equations (A^T A eigendecomposition).

    Equivalent to :func:`svd_solve` but significantly faster for tall matrices
    (e.g. 57600 × 772) because it reduces the problem to a small symmetric
    eigensystem via BLAS dgemm rather than running LAPACK dgesdd on the full A.

    The threshold on eigenvalues corresponds to the same relative condition
    number as the singular-value threshold in :func:`svd_solve`:
    ``lam_thresh = (rcond * sigma_max)^2 = rcond^2 * lam_max``.

    Accepts both the ``include_t_rx=True`` (npix + 4 columns) and
    ``include_t_rx=False`` (npix + 2 columns) forms of the design matrix from
    :func:`build_design_matrix`.  When T_rx columns are absent, ``t_rx_0``
    and ``t_rx_1`` are not present in the returned dict.

    Parameters
    ----------
    A : ndarray, shape (n_rows, npix + 2) or (n_rows, npix + 4)
    y : ndarray, shape (n_rows,)
        Measurement vector (flattened antenna temperatures).
    npix : int
        Number of sky pixels (first npix columns of A are the sky part).
    rcond : float
        Eigenvalues below rcond^2 * lam_max are treated as zero.

    Returns
    -------
    result : dict with keys
        sky_map       : (npix,) float64  — recovered sky temperature
                        (NaN = unobserved)
        t_regolith    : float
        t_sun         : float
        t_rx_0        : float — only present when A has npix + 4 columns
        t_rx_1        : float — only present when A has npix + 4 columns
        eigenvalues   : ascending eigenvalues of A^T A
        eigenvectors  : V such that A^T A = V diag(lam) V^T
        inv_eigenvalues : regularised 1/lam (zero for degenerate modes)
        rank          : int
        unobserved    : (npix,) bool — pixels with negligible column norm
    """
    AtA = A.T @ A
    Aty = A.T @ y
    lam, V = np.linalg.eigh(AtA)          # ascending eigenvalues
    lam_thresh = rcond ** 2 * lam[-1]
    inv_lam = np.where(lam > lam_thresh, 1.0 / np.where(lam > lam_thresh, lam, 1.0), 0.0)
    rank = int((lam > lam_thresh).sum())
    x_est = V @ (inv_lam * (V.T @ Aty))

    sky_map = x_est[:npix].copy()
    col_norms_sq = np.diag(AtA)[:npix]
    unobserved = col_norms_sq < (1e-6 ** 2) * col_norms_sq.max()
    sky_map[unobserved] = np.nan

    result = {
        "sky_map":         sky_map,
        "t_regolith":      float(x_est[npix]),
        "t_sun":           float(x_est[npix + 1]),
        "eigenvalues":     lam,
        "eigenvectors":    V,
        "inv_eigenvalues": inv_lam,
        "rank":            rank,
        "unobserved":      unobserved,
    }
    if A.shape[1] == npix + 4:
        result["t_rx_0"] = float(x_est[npix + 2])
        result["t_rx_1"] = float(x_est[npix + 3])
    return result


def svd_solve(A, y, npix, rcond=1e-6):
    """
    Least-squares sky recovery via thin SVD.

    Parameters
    ----------
    A : ndarray, shape (n_rows, npix + 4)
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
        t_rx_0     : float — receiver temperature, dipole 0 [K]
        t_rx_1     : float — receiver temperature, dipole 1 [K]
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
    t_sun      = float(x_est[npix + 1])
    t_rx_0     = float(x_est[npix + 2])
    t_rx_1     = float(x_est[npix + 3])

    col_norms = np.linalg.norm(A[:, :npix], axis=0)
    unobserved = col_norms < 1e-6 * col_norms.max()
    sky_map[unobserved] = np.nan

    return {
        "sky_map":    sky_map,
        "t_regolith": t_regolith,
        "t_sun":      t_sun,
        "t_rx_0":     t_rx_0,
        "t_rx_1":     t_rx_1,
        "U":          U,
        "sv":         sv,
        "Vt":         Vt,
        "rank":       rank,
        "unobserved": unobserved,
    }


def build_A_from_model(rots_per_orbit, masks, spectral_basis, c, delta_phi,
                       J_SUN, npix_sky, freq_idx, nside_beam=8):
    """
    Construct design matrix from parameterized HEALPix beam model.

    Evaluates a beam model specified via spectral decomposition (K spectral
    eigenmodes) and spatial coefficients at a given frequency, applies a
    rotation offset to all observations, interpolates the body-frame beam
    to sky coordinates via nearest-neighbor at the beam nside resolution,
    then calls :func:`build_design_matrix` to construct A.

    Parameters
    ----------
    rots_per_orbit : list of Rotation
        Spacecraft attitudes per observation (from eigsep_sim.observer.LunarOrbit).
    masks : ndarray, shape (n_obs, npix_sky), float32
        Binary masks (1 = sky, 0 = regolith).
    spectral_basis : dict
        Output from :func:`eigsep_sim.spectral.beam_spectral_basis`.
        Keys: 'psi' (2, K, N_freq), 'c_init' (2, K, npix_beam), 'nside'.
    c : ndarray, shape (2, K, npix_beam)
        Beam spatial coefficients (per dipole, per spectral mode, per pixel).
    delta_phi : ndarray, shape (3,)
        Rotation offset vector [rad] (applied to all observations uniformly).
    J_SUN : ndarray, shape (n_obs,), int
        HEALPix sun pixel per observation.
    npix_sky : int
        Sky resolution (nside value).
    freq_idx : int
        Index into spectral_basis['psi'] for the current frequency.
    nside_beam : int
        HEALPix resolution of the beam model (default 8; ~7° resolution).

    Returns
    -------
    A : ndarray, shape (n_obs*2, npix_sky + 2)
        Design matrix [sky_map_1..npix_sky | T_regolith | T_sun].
    omega_B : ndarray, shape (n_obs, 2)
        Beam solid angles per observation and dipole.
    """
    n_obs = len(rots_per_orbit)
    npix_beam = healpy.nside2npix(nside_beam)
    K = c.shape[1]

    # Apply rotation offset to attitudes
    R_offset = Rotation.from_rotvec(delta_phi)
    rots_eff = [R_offset * rot for rot in rots_per_orbit]

    # Evaluate body-frame beam at current frequency
    # psi: (2, K, N_freq), c: (2, K, npix_beam)
    psi_f = spectral_basis['psi'][:, :, freq_idx]  # (2, K)
    B_body_f = np.zeros((2, npix_beam), dtype=np.float32)
    for d in range(2):
        B_body_f[d] = np.einsum('k,kp->p', psi_f[d], c[d])  # (npix_beam,)

    # Rotate beam to galactic frame for each observation
    N_GAL = np.array(healpy.pix2vec(npix_sky, np.arange(healpy.nside2npix(npix_sky))))

    beams_gal = np.zeros((n_obs, 2, healpy.nside2npix(npix_sky)), dtype=np.float32)
    omega_B = np.zeros((n_obs, 2), dtype=np.float32)

    for k, rot_eff in enumerate(rots_eff):
        # Transform sky directions to body frame
        N_BODY = rot_eff.inv().apply(N_GAL.T).T  # (3, npix_sky)

        # Find nearest beam pixel for each sky pixel
        body_pix_idx = healpy.vec2pix(nside_beam, N_BODY[0], N_BODY[1], N_BODY[2])

        # Evaluate beam
        for d in range(2):
            beams_gal[k, d] = B_body_f[d, body_pix_idx]

        omega_B[k] = beams_gal[k].sum(axis=1)

    # Build design matrix with interpolated beams
    A = build_design_matrix(masks, beams_gal, omega_B, J_SUN,
                           healpy.nside2npix(npix_sky), include_t_rx=False)

    return A, omega_B
