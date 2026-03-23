#!/usr/bin/env python
"""
beam_cal_verify.py — Beam/orientation self-calibration via spectral modulation.

Quantifies the ability to recover a fixed rotation offset between the star-tracker
frame and the dipole axes, using the spectral modulation of the sky signal — i.e.,
the time-varying dipole power spectra y_f(t) that enter the linear inversion.

Method
------
Given a fixed offset R_δ applied to all spacecraft attitudes:
  rots_true = R_δ * rots_nominal  (same offset every timestep)

1. Simulate observations with the TRUE (offset) orientations.
2. Recover T̂_sky using the NOMINAL (zero-offset) A matrix.
3. Compute residuals r_f = y_f − A_nom_f @ x̂_f  (x̂_f includes T̂_sky, T̂_reg, T̂_sun).
4. Compute Jacobian J_f[:,k] = ∂(A_f(δ) x̂_f)/∂δ_k|_{δ=0}  (finite differences).
5. Project J_f off A_nom column space: J̃_f = (I − H_f) J_f.
6. Stack over frequencies and solve: δ̂ = pinv(J̃_stacked) r_stacked.
7. Report recovery error and Cramér-Rao bound.

The residuals live in the null space of A_f^T (dimension ≈ N_obs×2 − N_pix per
frequency), giving ~56,000 independent equations per channel vs 3 unknowns —
enormous overdetermination that enables sub-arcsecond in-flight calibration.

Note: occultation masks are held fixed at the nominal attitudes (valid for
offsets ≲5°, since mask changes are negligible at the NSIDE=8 resolution).

Run from the repo root:
    python notebooks/beam_cal_verify.py
    python notebooks/beam_cal_verify.py --offset_deg 0.1 --noise_scale 0
    python notebooks/beam_cal_verify.py --sweep
"""

import argparse
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from astropy.coordinates import get_body
import astropy.units as u
import healpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from eigsep_sim.lunar_orbit import OrbiterMission
from eigsep_sim.sky import SkyModel
from eigsep_sim.sim import compute_masks_and_beams, compute_beams, simulate_observations
from eigsep_sim.linear_solver import build_design_matrix, normal_solve
from eigsep_sim.models import T21cmModel
from sim_cache import config_fingerprint, try_load_setup

_YAML = os.path.join(_HERE, "bloom_config.yaml")
cfg = OrbiterMission(_YAML)

N_DAYS        = cfg.observation.n_days
N_ORBITS      = cfg.observation.n_orbits
N_OBS         = cfg.observation.n_obs
NSIDE         = cfg.observation.nside
NPIX          = cfg.observation.npix
FREQ_MIN_MHZ  = cfg.observation.freq_min_mhz
FREQ_MAX_MHZ  = cfg.observation.freq_max_mhz
NCHAN_SCIENCE = cfg.observation.nchan_science
FREQS_MHZ     = np.linspace(FREQ_MIN_MHZ, FREQ_MAX_MHZ, NCHAN_SCIENCE)
N_FREQ        = len(FREQS_MHZ)
T_RX_K        = cfg.antenna.t_rx
T_REGOLITH    = cfg.observation.t_regolith
T_SUN         = cfg.observation.t_sun
T_INTEGRATION = cfg.observation.t_integration

# STM reference values (from pointing_error_deg1.0_seed99_8812aa07.npz)
STM_BIAS_PER_DEG_MK = 122.29   # mK bias for 1° systematic offset (eigenmode-filtered)
STM_SIGMA_MONO_MK   = 1.64     # mK per-channel σ_mono (mean over science band)
REQ_SYS_DEG         = 0.10 * STM_SIGMA_MONO_MK / STM_BIAS_PER_DEG_MK
REQ_SYS_ARCSEC      = REQ_SYS_DEG * 3600.0   # ≈ 5 arcsec


# ─── Setup ────────────────────────────────────────────────────────────────────

def _setup():
    """Return nominal attitudes, masks, sky models, and Sun pixels."""
    if cfg.observation.fixed_spin:
        phi = np.linspace(0.0, 2.0 * np.pi, N_OBS, endpoint=False)
        rot_fixed = Rotation.from_rotvec(np.outer(phi, cfg.antenna.l_hat))
        rots_nom = [rot_fixed for _ in range(N_ORBITS)]
    else:
        rots_nom = [
            Rotation.random(N_OBS, random_state=42 + o)
            for o in range(N_ORBITS)
        ]

    orbits_list = cfg.observation.make_orbits(rot_spin_vec=(0, 0, 1), spin_period=0.0)
    t_obs_s     = np.linspace(0.0, N_DAYS * 86400.0, N_OBS, endpoint=False)
    obs_times   = cfg.observation.obs_epoch + t_obs_s * u.s

    print("Loading GSM …", flush=True)
    sky_mf   = SkyModel(FREQS_MHZ * 1e6, nside=NSIDE, srcs=None)
    gsm_maps = np.asarray(sky_mf.map)          # (npix, N_FREQ)
    if gsm_maps.ndim == 1:
        gsm_maps = gsm_maps[:, np.newaxis]

    models_21cm = T21cmModel()
    T_21_INJ = models_21cm(FREQS_MHZ * 1e6, model_index=0)

    print("Querying Sun positions …", flush=True)
    sun_coords = get_body("sun", obs_times)
    sun_gal    = sun_coords.galactic
    l_s, b_s   = sun_gal.l.rad, sun_gal.b.rad
    J_SUN = healpy.vec2pix(NSIDE,
                           np.cos(b_s) * np.cos(l_s),
                           np.cos(b_s) * np.sin(l_s),
                           np.sin(b_s))

    fp     = config_fingerprint(cfg)
    cached = try_load_setup(fp)
    if cached is not None:
        masks, _ = cached
    else:
        print("Computing occultation masks (not in cache) …", flush=True)
        masks, _, _ = compute_masks_and_beams(
            orbits_list, obs_times, rots_nom,
            cfg.antenna.u_body, cfg.antenna.kh(FREQS_MHZ[N_FREQ // 2]),
            NSIDE, verbose=False,
        )

    print(f"  n_total = {masks.shape[0]},  npix = {masks.shape[1]},  "
          f"n_rows/freq = {masks.shape[0] * 2}")
    return rots_nom, masks, gsm_maps, T_21_INJ, J_SUN


def _make_rots_true(rots_nom, delta_rotvec):
    """Apply a fixed body-frame rotation offset to all nominal attitudes."""
    R_offset = Rotation.from_rotvec(delta_rotvec)
    return [R_offset * rots_o for rots_o in rots_nom]


# ─── Core calibration ─────────────────────────────────────────────────────────

def _project_off_A(A, v, res):
    """Return (I − H_A) v  using the eigendecomposition stored in ``res``."""
    # H_A v = A (A^T A)^{-1} A^T v
    # (A^T A)^{-1} A^T v  ←  from eigendecomposition already in res
    V       = res["eigenvectors"]     # (ncols, ncols)
    inv_lam = res["inv_eigenvalues"]  # (ncols,)
    coeff   = V @ (inv_lam * (V.T @ (A.T @ v)))   # (ncols,)
    return v - A @ coeff


def _calibrate_freq(fi, rots_nom, rots_true, masks, gsm_maps, T_21_INJ, J_SUN,
                    noise_scale, eps_rad, seed_offset=0):
    """
    Run one-frequency calibration step.

    Returns
    -------
    J_tilde : ndarray (n_rows, 3)  projected Jacobian
    r       : ndarray (n_rows,)    residuals
    sigma_f : float                noise std used
    """
    f_mhz = FREQS_MHZ[fi]
    print(f"  [{fi+1:2d}/{N_FREQ}] {f_mhz:.1f} MHz …", end=" ", flush=True)
    kh_f  = cfg.antenna.kh(f_mhz)
    gsm_f = gsm_maps[:, fi]
    sky_f = gsm_f + T_21_INJ[fi]
    sigma_f = cfg.antenna.sigma_noise(
        f_mhz, cfg.observation.delta_nu, T_INTEGRATION,
        t_gsm_avg=float(gsm_f.mean()),
    ) * noise_scale   # shape (2,): per-dipole noise std

    # ── True beams → simulate y_f ──────────────────────────────────────────
    beams_true, omega_B_true = compute_beams(rots_true, cfg.antenna.u_body, kh_f, NSIDE)
    _, y_f = simulate_observations(
        masks, beams_true, omega_B_true,
        sky_f, T_REGOLITH, T_SUN, J_SUN,
        sigma_f,
        rng=np.random.default_rng(fi + seed_offset),
        t_rx=np.full(2, T_RX_K),
    )

    # ── Nominal beams → recover sky ─────────────────────────────────────────
    beams_nom, omega_B_nom = compute_beams(rots_nom, cfg.antenna.u_body, kh_f, NSIDE)
    A_nom = build_design_matrix(masks, beams_nom, omega_B_nom, J_SUN, NPIX,
                                include_t_rx=False)
    res   = normal_solve(A_nom, y_f, NPIX)

    # Full solution vector [sky (NPIX), T_regolith, T_sun]
    sky_hat = res["sky_map"].copy()
    sky_hat[res["unobserved"]] = float(np.nanmean(sky_hat))   # fill unobserved
    x_hat = np.empty(NPIX + 2)
    x_hat[:NPIX]   = sky_hat
    x_hat[NPIX]    = res["t_regolith"]
    x_hat[NPIX + 1] = res["t_sun"]

    # Residuals  r = y − A_nom x̂
    y_hat = A_nom @ x_hat
    r     = y_f - y_hat

    # ── Jacobian  J[:,k] = ∂(A_f(δ) x̂)/∂δ_k|_{δ=0} ────────────────────────
    J = np.empty((len(r), 3))
    for k in range(3):
        axis_k = np.zeros(3)
        axis_k[k] = 1.0
        rots_pert = _make_rots_true(rots_nom, eps_rad * axis_k)
        beams_pert, omega_B_pert = compute_beams(rots_pert, cfg.antenna.u_body, kh_f, NSIDE)
        A_pert = build_design_matrix(masks, beams_pert, omega_B_pert, J_SUN, NPIX,
                                     include_t_rx=False)
        J[:, k] = (A_pert @ x_hat - y_hat) / eps_rad

    # ── Project J off A_nom column space: J̃ = (I − H) J ────────────────────
    J_tilde = np.empty_like(J)
    for k in range(3):
        J_tilde[:, k] = _project_off_A(A_nom, J[:, k], res)

    # Per-row sigma: rows alternate dipole-0 and dipole-1 for each observation
    # Use physics sigma even for noiseless case (for CRB); caller handles noise_scale=0
    n_total = masks.shape[0]
    sigma_nom = cfg.antenna.sigma_noise(
        f_mhz, cfg.observation.delta_nu, T_INTEGRATION,
        t_gsm_avg=float(gsm_f.mean()),
    )  # unscaled, shape (2,)
    sigma_per_row = np.tile(sigma_nom, n_total)   # [s0, s1, s0, s1, ...]

    print("done", flush=True)
    return J_tilde, r, sigma_per_row


def calibrate_offset(rots_nom, rots_true, masks, gsm_maps, T_21_INJ, J_SUN,
                     noise_scale=1.0, eps_deg=0.01, freq_indices=None,
                     seed_offset=0):
    """
    Estimate a fixed rotation offset δ by stacking residuals over frequencies.

    Parameters
    ----------
    freq_indices : array-like of int or None
        Frequency channels to include; None = all.

    Returns
    -------
    delta_hat   : ndarray (3,)   estimated rotation vector [rad]
    delta_sigma : ndarray (3,)   1-sigma CRB per axis [rad]
    """
    if freq_indices is None:
        freq_indices = np.arange(N_FREQ)

    eps_rad = np.deg2rad(eps_deg)

    J_parts = []
    r_parts = []
    w_parts = []   # per-row weight = 1/sigma^2

    for fi in freq_indices:
        J_tilde, r, sigma_per_row = _calibrate_freq(
            fi, rots_nom, rots_true, masks, gsm_maps, T_21_INJ, J_SUN,
            noise_scale, eps_rad, seed_offset,
        )
        J_parts.append(J_tilde)
        r_parts.append(r)
        w_parts.append(1.0 / sigma_per_row**2)

    J_stack = np.vstack(J_parts)     # (n_rows_total, 3)
    r_stack = np.concatenate(r_parts)
    w_stack = np.concatenate(w_parts)

    # Noiseless case: weights are still the physical sigma (CRB is geometric)
    if not np.all(np.isfinite(w_stack)):
        w_stack = np.ones(len(r_stack))

    # Weighted least squares  δ̂ = (J^T W J)^{-1} J^T W r
    w_sqrt  = np.sqrt(w_stack)
    J_w     = J_stack * w_sqrt[:, np.newaxis]
    r_w     = r_stack * w_sqrt
    delta_hat, _, _, _ = np.linalg.lstsq(J_w, r_w, rcond=None)

    # Cramér-Rao bound from Fisher information matrix
    FIM        = J_w.T @ J_w        # (3, 3)
    CRB_cov    = np.linalg.inv(FIM)
    delta_sigma = np.sqrt(np.diag(CRB_cov))   # [rad]

    return delta_hat, delta_sigma, J_stack, r_stack


# ─── Reporting ────────────────────────────────────────────────────────────────

def _print_result(delta_true, delta_hat, delta_sigma, J_stack, r_stack, n_freq_used):
    true_deg  = np.rad2deg(np.linalg.norm(delta_true))
    hat_deg   = np.rad2deg(np.linalg.norm(delta_hat))
    err_asec  = np.rad2deg(np.linalg.norm(delta_hat - delta_true)) * 3600.0
    crb_asec  = np.rad2deg(delta_sigma) * 3600.0
    crb_tot   = np.sqrt(np.sum(crb_asec**2))

    print("\n" + "─" * 70)
    print("Calibration results")
    print("─" * 70)
    print(f"  True  |δ|         : {true_deg:.4f}°  = {true_deg * 60:.2f} arcmin")
    print(f"  Recov |δ̂|         : {hat_deg:.4f}°")
    print(f"  Recovery error    : {err_asec:.3f} arcsec")
    print(f"  CRB 1-σ (x,y,z)  : {crb_asec[0]:.3f}  {crb_asec[1]:.3f}  "
          f"{crb_asec[2]:.3f}  arcsec")
    print(f"  CRB total 1-σ     : {crb_tot:.3f} arcsec  (quad. sum)")
    print(f"  N_freq used       : {n_freq_used}")
    print(f"  N_rows in stack   : {J_stack.shape[0]:,}")
    print(f"  |r| / sqrt(N_rows): {np.linalg.norm(r_stack) / np.sqrt(len(r_stack)):.4f} K")

    print("\n" + "─" * 70)
    print("Comparison to STM requirements")
    print("─" * 70)
    print(f"  STM systematic req (pre-launch): < {REQ_SYS_ARCSEC:.0f} arcsec")
    print(f"  Spectral-cal CRB  (in-flight)  : {crb_tot:.2f} arcsec", end="  ")
    if crb_tot < REQ_SYS_ARCSEC:
        margin = REQ_SYS_ARCSEC / crb_tot
        print(f"✓  margin = {margin:.1f}×")
        print(f"  → In-flight spectral calibration can relax the pre-launch "
              f"requirement by {margin:.1f}×.")
    else:
        print("✗")
        print(f"  → In-flight spectral calibration alone does not meet the "
              f"{REQ_SYS_ARCSEC:.0f}-arcsec requirement.")


def _sweep(rots_nom, masks, gsm_maps, T_21_INJ, J_SUN,
           axis, freq_indices, noise_scale, eps_deg):
    offsets_deg = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"\nSweeping offsets ({len(freq_indices)} freqs, noise_scale={noise_scale}):")
    print(f"{'Offset':>8}  {'|δ̂|':>8}  {'Error':>12}  {'CRB':>12}  {'S/N':>8}")
    print(f"{'[°]':>8}  {'[°]':>8}  {'[arcsec]':>12}  {'[arcsec]':>12}  {'':>8}")
    print("─" * 58)

    for off_deg in offsets_deg:
        delta_true  = np.deg2rad(off_deg) * axis
        rots_true   = _make_rots_true(rots_nom, delta_true)
        dhat, dsig, Js, rs = calibrate_offset(
            rots_nom, rots_true, masks, gsm_maps, T_21_INJ, J_SUN,
            noise_scale=noise_scale, eps_deg=eps_deg,
            freq_indices=freq_indices,
        )
        err_asec = np.rad2deg(np.linalg.norm(dhat - delta_true)) * 3600.0
        crb_asec = np.sqrt(np.sum((np.rad2deg(dsig) * 3600.0) ** 2))
        hat_deg  = np.rad2deg(np.linalg.norm(dhat))
        # Signal-to-noise: Js @ delta_true vs noise (residual rms)
        signal   = np.linalg.norm(Js @ delta_true)
        noise_rms = np.sqrt(np.mean(rs ** 2)) * np.sqrt(len(rs))
        snr      = signal / noise_rms if noise_rms > 0 else np.inf
        print(f"{off_deg:8.2f}  {hat_deg:8.4f}  {err_asec:12.3f}  {crb_asec:12.3f}  {snr:8.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantify in-flight pointing calibration via spectral modulation")
    parser.add_argument("--offset_deg", type=float, default=1.0,
                        help="Fixed rotation offset magnitude [deg] (default: 1.0)")
    parser.add_argument("--offset_axis", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                        help="Rotation axis unit vector (default: x)")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Noise scale factor (0=noiseless, default: 1.0)")
    parser.add_argument("--n_freq", type=int, default=None,
                        help="Number of frequency channels to use (default: all 30)")
    parser.add_argument("--eps_deg", type=float, default=0.01,
                        help="Finite-difference step for Jacobian [deg] (default: 0.01)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over offset magnitudes to test linearity")
    args = parser.parse_args()

    print("=" * 70)
    print("BLOOM-21CM: Beam Orientation Self-Calibration (Spectral Modulation)")
    print("=" * 70)

    rots_nom, masks, gsm_maps, T_21_INJ, J_SUN = _setup()

    # Frequency subset
    if args.n_freq is not None and args.n_freq < N_FREQ:
        freq_indices = np.round(np.linspace(0, N_FREQ - 1, args.n_freq)).astype(int)
        freq_indices = np.unique(freq_indices)
    else:
        freq_indices = np.arange(N_FREQ)
    print(f"\nFrequencies: {len(freq_indices)} channels  "
          f"({FREQS_MHZ[freq_indices[0]]:.1f}–{FREQS_MHZ[freq_indices[-1]]:.1f} MHz)")

    axis = np.array(args.offset_axis, dtype=float)
    axis /= np.linalg.norm(axis)

    if args.sweep:
        _sweep(rots_nom, masks, gsm_maps, T_21_INJ, J_SUN,
               axis, freq_indices, args.noise_scale, args.eps_deg)
        return

    delta_true = np.deg2rad(args.offset_deg) * axis
    rots_true  = _make_rots_true(rots_nom, delta_true)
    print(f"\nOffset: {args.offset_deg:.3f}°  along axis {axis}  "
          f"(noise_scale={args.noise_scale})")

    print("Running calibration …", flush=True)
    delta_hat, delta_sigma, J_stack, r_stack = calibrate_offset(
        rots_nom, rots_true, masks, gsm_maps, T_21_INJ, J_SUN,
        noise_scale=args.noise_scale, eps_deg=args.eps_deg,
        freq_indices=freq_indices,
    )

    _print_result(delta_true, delta_hat, delta_sigma, J_stack, r_stack,
                  len(freq_indices))


if __name__ == "__main__":
    main()
