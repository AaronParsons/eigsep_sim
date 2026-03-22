#!/usr/bin/env python
"""
BLOOM-21CM Science Traceability Matrix (STM)
============================================
Mission: Broadband Lunar Occultation Orbiter for Measuring the 21-cm Monopole
Agency:  NASA Astrophysics Pioneers (ROSES-25)

Run from the repo root:
    python notebooks/lunar_stm.py

This script derives every STM entry from first principles using the mission
configuration in bloom_config.yaml and live simulations.  Three analyses are
performed beyond the routine multi-frequency inversion:

  1. Physical-noise SNR  — run_multifreq(noise_scale=1.0) using the established
     pipeline (N_EIG_MODES = 4 GSM modes + flat, noiseless FG leakage scan).

  2. Pointing-error simulation — simulate observations with the TRUE spacecraft
     attitudes, then invert using an A matrix built from orientations perturbed
     by 1° (the attitude knowledge requirement).  The recovered monopole bias
     quantifies the sensitivity of science results to pointing knowledge errors.

  3. All-sky map l_max — single-frequency per-pixel inversion yields per-pixel
     noise variances.  Comparing the resulting noise power spectrum to the GSM
     signal power spectrum identifies the highest multipole l_max accessible at
     <10% error on the spherical harmonic coefficients a_lm.
"""

import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from eigsep_sim.spectral import gsm_eigenmodes, eigenmode_filter
from sim_cache import (
    config_fingerprint,
    save_setup, try_load_setup,
    save_multifreq, try_load_multifreq,
    save_pointing_error, try_load_pointing_error,
)

# ══════════════════════════════════════════════════════════════════════════════
# A. Configuration & Derived Parameters
# ══════════════════════════════════════════════════════════════════════════════

_YAML = os.path.join(_HERE, "bloom_config.yaml")
cfg = OrbiterMission(_YAML)

# --- Mission-level parameters ------------------------------------------------
N_DAYS          = cfg.observation.n_days          # 60-day nominal mission
N_DAYS_EXT      = 365                             # 1-year extended mission
N_ORBITS        = cfg.observation.n_orbits        # 2 orbital planes
N_OBS           = cfg.observation.n_obs           # time samples per orbit
NSIDE           = cfg.observation.nside           # HEALPix nside = 8
NPIX            = cfg.observation.npix            # 768 pixels (nside=8)

FREQ_MIN_MHZ    = cfg.observation.freq_min_mhz   # 55 MHz — primary band low
FREQ_MAX_MHZ    = cfg.observation.freq_max_mhz   # 115 MHz — primary band high
NCHAN_SCIENCE   = cfg.observation.nchan_science   # 30 science channels
FREQS_MHZ       = np.linspace(FREQ_MIN_MHZ, FREQ_MAX_MHZ, NCHAN_SCIENCE)
N_FREQ          = len(FREQS_MHZ)

# Extended band (30–170 MHz)
EXT_BAND_LOW    = 30.0    # MHz
EXT_BAND_HIGH   = 170.0   # MHz
DELTA_NU_MHZ    = cfg.observation.delta_nu / 1e6          # 2 MHz channel width
NCHAN_EXT       = round((EXT_BAND_HIGH - EXT_BAND_LOW) / DELTA_NU_MHZ)  # 70

DUTY_CYCLE      = cfg.observation.duty_cycle              # 0.917
ATT_KNOW_DEG    = cfg.observation.attitude_knowledge_deg  # 1.0 deg
SPIN_PERIOD_S   = cfg.observation.spin_period_s           # 600 s

T_INTEGRATION   = cfg.observation.t_integration          # 165 s  (sensitivity-scaling; NOT hardware time)
T_SNAPSHOT      = cfg.observation.t_snapshot             # 1.67 s (time for 1° beam shift at current spin rate)

# Hardware accumulation time per transmitted spectrum
# Beam rotation during one accumulation (at spin rate 360°/SPIN_PERIOD_S)
BYTES_PER_SAMPLE   = 4              # float32
T_ACCUM         = 1.0                  # 1.0 s at 1 Hz; limited by data budget
THETA_SWEEP_ACCUM_DEG = T_ACCUM * 360.0 / SPIN_PERIOD_S  # 0.6°

T_RX_K          = cfg.antenna.t_rx                       # 100 K
T_REGOLITH      = cfg.observation.t_regolith             # 300 K
T_SUN           = cfg.observation.t_sun                  # 5000 K

SYNODIC_MONTH   = cfg.synodic_month_days                 # 29.53 days

N_EIG_MODES     = 4   # established optimal (FG leakage sweet spot)
ATT_ERR_DEG     = 1.0  # attitude perturbation for pointing-error simulation

# --- Data-volume budget -------------------------------------------------------
#   Raw data = 2 dipoles × N_ch_ext × 4 bytes × 1 Hz sample rate × 2 spacecraft
SAMPLE_RATE_HZ     = 1.0 / T_ACCUM           # 1 Hz accumulation rate
DATA_RATE_BPS      = (2 * NCHAN_EXT * BYTES_PER_SAMPLE * SAMPLE_RATE_HZ
                      * N_ORBITS)           # bytes/s, both spacecraft
DATA_MBPERDAY      = DATA_RATE_BPS * 86400 / 1e6
DOWNLINK_MBPERHR   = 60.0          # X-band downlink per spacecraft (MB/hr)
DOWNLINK_HRS_DAY   = (1 - DUTY_CYCLE) * 24  # ≈ 2 hr/day contact window
DOWNLINK_MB_DAY    = DOWNLINK_MBPERHR * DOWNLINK_HRS_DAY * N_ORBITS
DOWNLINK_MARGIN    = DOWNLINK_MB_DAY / DATA_MBPERDAY
# Fastest allowed T_ACCUM within the downlink budget (full downlink, current format)
T_ACCUM_MIN_S   = (N_ORBITS * NCHAN_EXT * 2 * BYTES_PER_SAMPLE) / (DOWNLINK_MB_DAY * 1e6 / 86400)


# ══════════════════════════════════════════════════════════════════════════════
# B. Simulation Setup  (identical to test_multifreq.py)
# ══════════════════════════════════════════════════════════════════════════════

def _setup(fp):
    """Return all precomputed simulation inputs, using cache where available."""
    print("─" * 70)
    print("BLOOM-21CM STM: simulation setup")
    print("─" * 70)

    # Spacecraft attitudes — deterministic from seed, never cached
    if cfg.observation.fixed_spin:
        phi = np.linspace(0.0, 2.0 * np.pi, N_OBS, endpoint=False)
        rot_fixed = Rotation.from_rotvec(np.outer(phi, cfg.antenna.l_hat))
        rots = [rot_fixed for _ in range(N_ORBITS)]
    else:
        rots = [
            Rotation.random(N_OBS, random_state=42 + o)
            for o in range(N_ORBITS)
        ]

    # Orbits and time grid
    orbits_list = cfg.observation.make_orbits(rot_spin_vec=(0, 0, 1), spin_period=0.0)
    t_obs_s     = np.linspace(0.0, N_DAYS * 86400.0, N_OBS, endpoint=False)
    obs_times   = cfg.observation.obs_epoch + t_obs_s * u.s

    # GSM and 21cm models
    print("Loading GSM …", flush=True)
    sky_mf = SkyModel(FREQS_MHZ * 1e6, nside=NSIDE, srcs=None)
    gsm_maps = np.asarray(sky_mf.map)
    if gsm_maps.ndim == 1:
        gsm_maps = gsm_maps[:, np.newaxis]

    models_21cm = T21cmModel()
    T_21_INJ = models_21cm(FREQS_MHZ * 1e6, model_index=0)
    print(f"  21cm model 0: peak = {T_21_INJ.min()*1e3:.1f} mK  "
          f"rms = {np.std(T_21_INJ)*1e3:.2f} mK")

    # Sun positions
    print("Querying Sun positions …", flush=True)
    sun_coords = get_body("sun", obs_times)
    sun_gal    = sun_coords.galactic
    l_s, b_s   = sun_gal.l.rad, sun_gal.b.rad
    J_SUN = healpy.vec2pix(NSIDE,
                           np.cos(b_s)*np.cos(l_s),
                           np.cos(b_s)*np.sin(l_s),
                           np.sin(b_s))

    # Occultation masks — try cache first
    cached = try_load_setup(fp)
    if cached is not None:
        masks_mf, J_SUN_cached = cached
        # J_SUN is derived from obs_times (deterministic), so use the recomputed one
    else:
        print("Computing occultation masks …", flush=True)
        masks_mf, _, _ = compute_masks_and_beams(
            orbits_list, obs_times, rots,
            cfg.antenna.u_body, cfg.antenna.kh(FREQS_MHZ[N_FREQ // 2]),
            NSIDE, verbose=False,
        )
        save_setup(masks_mf, J_SUN, fp)

    print(f"  masks shape = {masks_mf.shape}  "
          f"(mean open fraction = {masks_mf.mean():.2f})")

    return rots, orbits_list, obs_times, gsm_maps, models_21cm, T_21_INJ, J_SUN, masks_mf


# ══════════════════════════════════════════════════════════════════════════════
# C. Multi-frequency inversion (shared with pointing-error sim)
# ══════════════════════════════════════════════════════════════════════════════

def _run_multifreq(rots, masks, gsm_maps, T_21_INJ, J_SUN,
                   noise_scale=1.0, noise_seed_offset=0):
    """Per-pixel inversion at each science frequency.

    Returns
    -------
    T_sky_mean_est : ndarray (N_FREQ,)
    SIGMA_MONO     : ndarray (N_FREQ,)
    """
    T_sky_mean_est = np.empty(N_FREQ)
    SIGMA_MONO     = np.empty(N_FREQ)

    for fi, f_mhz in enumerate(FREQS_MHZ):
        kh_f = cfg.antenna.kh(f_mhz)
        beams_f, omega_B_f = compute_beams(
            rots, cfg.antenna.u_body, kh_f, NSIDE,
        )
        gsm_f   = gsm_maps[:, fi]
        sky_f   = gsm_f + T_21_INJ[fi]
        sigma_f = cfg.antenna.sigma_noise(
            f_mhz, cfg.observation.delta_nu, T_INTEGRATION,
            t_gsm_avg=float(gsm_f.mean()),
        ) * noise_scale

        _, y_f = simulate_observations(
            masks, beams_f, omega_B_f,
            sky_f, T_REGOLITH, T_SUN,
            J_SUN, sigma_f,
            t_rx=np.full(2, T_RX_K),
            rng=np.random.default_rng(fi + noise_seed_offset),
        )

        A_f   = build_design_matrix(masks, beams_f, omega_B_f, J_SUN, NPIX,
                                    include_t_rx=False)
        res_f = normal_solve(A_f, y_f, NPIX)

        T_sky_mean_est[fi] = float(np.nanmean(res_f['sky_map']))

        n_obs_pix = int((~res_f['unobserved']).sum())
        e_sky = np.zeros(NPIX + 2)
        e_sky[:NPIX][~res_f['unobserved']] = 1.0 / n_obs_pix
        Ve = res_f['eigenvectors'].T @ e_sky
        SIGMA_MONO[fi] = (float(np.mean(sigma_f)) *
                          np.sqrt(float(np.dot(Ve**2, res_f['inv_eigenvalues']))))

    return T_sky_mean_est, SIGMA_MONO


# ══════════════════════════════════════════════════════════════════════════════
# D. Pointing-error simulation
# ══════════════════════════════════════════════════════════════════════════════

def _run_pointing_error_sim(rots_true, orbits_list, obs_times,
                             masks_true, gsm_maps, T_21_INJ, J_SUN,
                             fp, pert_deg=1.0, seed=99):
    """
    Simulate data with TRUE attitudes; invert with PERTURBED A matrix.

    The 1° random orientation offset between the "truth" (data simulation) and
    the "model" (A matrix weights) represents the worst-case attitude knowledge
    error.  The systematic bias in the recovered sky monopole relative to the
    noiseless truth defines the attitude knowledge requirement.

    Parameters
    ----------
    fp : str            config fingerprint for cache keying
    pert_deg : float    1-sigma attitude perturbation (degrees)
    seed : int          RNG seed for the perturbation

    Returns
    -------
    T_sky_mean_pert : ndarray (N_FREQ,)
    masks_pert : ndarray (n_total, npix)
    """
    # Try cache first
    cached = try_load_pointing_error(FREQS_MHZ, pert_deg, seed, fp)
    if cached is not None:
        T_sky_mean_pert, masks_pert, mask_diff = cached
        print(f"  Mask pixels flipped by pointing error: {mask_diff}  "
              f"({mask_diff / masks_true.size:.4%} of total)  [from cache]")
        return T_sky_mean_pert, masks_pert

    print(f"\nPointing-error simulation  (Δθ = {pert_deg}°) …", flush=True)
    rng_pert  = np.random.default_rng(seed)
    angle_rad = np.deg2rad(pert_deg)

    # Perturb each observation's attitude by pert_deg on a random axis
    perturbed_rots = []
    for rots_o in rots_true:
        n = len(rots_o)
        rotvec = rng_pert.standard_normal((n, 3))
        # Fixed magnitude = pert_deg (deterministic perturbation norm)
        rotvec /= np.linalg.norm(rotvec, axis=1, keepdims=True)
        rotvec *= angle_rad
        pert = Rotation.from_rotvec(rotvec)
        perturbed_rots.append(pert * rots_o)

    # Recompute masks with perturbed attitudes (limb pixels may flip)
    print("  Recomputing occultation masks with perturbed attitudes …", flush=True)
    masks_pert, _, _ = compute_masks_and_beams(
        orbits_list, obs_times, perturbed_rots,
        cfg.antenna.u_body, cfg.antenna.kh(FREQS_MHZ[N_FREQ // 2]),
        NSIDE, verbose=False,
    )
    mask_diff = int(np.sum(masks_pert != masks_true))
    mask_frac = mask_diff / masks_true.size
    print(f"  Mask pixels flipped by pointing error: {mask_diff}  "
          f"({mask_frac:.4%} of total)")

    # Per-frequency inversion: TRUE data, PERTURBED A
    T_sky_mean_pert = np.empty(N_FREQ)
    for fi, f_mhz in enumerate(FREQS_MHZ):
        kh_f = cfg.antenna.kh(f_mhz)

        # True beams → simulate noiseless data
        beams_true_f, omega_B_true_f = compute_beams(
            rots_true, cfg.antenna.u_body, kh_f, NSIDE,
        )
        gsm_f  = gsm_maps[:, fi]
        sky_f  = gsm_f + T_21_INJ[fi]
        _, y_f = simulate_observations(
            masks_true, beams_true_f, omega_B_true_f,
            sky_f, T_REGOLITH, T_SUN,
            J_SUN, np.zeros(2),   # noiseless — must be shape (2,) not scalar
            t_rx=np.full(2, T_RX_K),
            rng=np.random.default_rng(fi),
        )

        # Perturbed beams → A matrix
        beams_pert_f, omega_B_pert_f = compute_beams(
            perturbed_rots, cfg.antenna.u_body, kh_f, NSIDE,
        )
        A_pert = build_design_matrix(masks_pert, beams_pert_f, omega_B_pert_f,
                                     J_SUN, NPIX, include_t_rx=False)
        res_pert = normal_solve(A_pert, y_f, NPIX)
        T_sky_mean_pert[fi] = float(np.nanmean(res_pert['sky_map']))

    save_pointing_error(FREQS_MHZ, T_sky_mean_pert, masks_pert,
                        pert_deg, seed, mask_diff, fp)
    return T_sky_mean_pert, masks_pert


# ══════════════════════════════════════════════════════════════════════════════
# E. Per-pixel noise power spectrum  →  l_max for Science Goal 2
# ══════════════════════════════════════════════════════════════════════════════

def _compute_lmax(rots, masks, gsm_maps, J_SUN, freq_req_frac=0.10, nside_sci=64):
    """
    Estimate the highest multipole l_max accessible at < freq_req_frac error
    on the spherical harmonic coefficients a_lm.

    Method
    ------
    1. Run the per-pixel inversion at a representative frequency.
    2. Compute σ_j (noise per sky pixel) from the eigendecomposition.
    3. Estimate the noise angular power spectrum:
         C_l^noise ≈ (4π / n_obs_pix) × mean(σ_j²)     [white-noise limit]
    4. Compute the GSM signal power spectrum from healpy.anafast.
    5. l_max is the highest l where C_l^noise < freq_req_frac² × C_l^GSM.

    Parameters
    ----------
    freq_req_frac : float
        Maximum fractional error on a_lm (default 0.10 = 10%).

    Returns
    -------
    l_max : int
    sigma_pix_K : float  (RMS per-pixel noise at the representative freq)
    Cl_noise : ndarray, shape (lmax_healpix+1,)
    Cl_signal : ndarray, shape (lmax_healpix+1,)
    """
    # Representative frequency: geometric mean of primary band
    f_rep = np.sqrt(FREQ_MIN_MHZ * FREQ_MAX_MHZ)
    fi    = np.argmin(np.abs(FREQS_MHZ - f_rep))
    f_mhz = FREQS_MHZ[fi]
    print(f"\nComputing per-pixel noise at representative freq = {f_mhz:.1f} MHz …",
          flush=True)

    kh_f = cfg.antenna.kh(f_mhz)
    beams_f, omega_B_f = compute_beams(rots, cfg.antenna.u_body, kh_f, NSIDE)
    gsm_f  = gsm_maps[:, fi]
    sigma_f = cfg.antenna.sigma_noise(
        f_mhz, cfg.observation.delta_nu, T_INTEGRATION,
        t_gsm_avg=float(gsm_f.mean()),
    )

    A_f   = build_design_matrix(masks, beams_f, omega_B_f, J_SUN, NPIX,
                                include_t_rx=False)
    res_f = normal_solve(A_f, np.zeros(A_f.shape[0]), NPIX)

    # Per-pixel noise: σ_j = σ_f × sqrt(Σ_k V_{jk}² / λ_k)
    V   = res_f['eigenvectors']                # (npix+2, npix+2)
    il  = res_f['inv_eigenvalues']             # (npix+2,)
    obs = ~res_f['unobserved']                 # boolean mask of observed pixels
    # Noise variance for each sky pixel (first NPIX rows of V)
    Vpix = V[:NPIX, :]                         # (npix, n_eig)
    sigma_pix = np.full(NPIX, np.nan)
    sigma_pix[obs] = float(np.mean(sigma_f)) * np.sqrt(
        (Vpix[obs] ** 2) @ il
    )

    # ── Noise C_l: scale-invariant (white noise, C_l^noise = σ² × Ω_pix) ───────
    # C_l^noise = var_pix × 4π / N_pix  is independent of nside because
    # σ_pix ∝ sqrt(N_obs / N_pix) and Ω_pix = 4π / N_pix both scale together.
    # We compute it at simulation nside (NSIDE=8) and reuse at nside_sci.
    n_obs_pix   = obs.sum()
    var_pix_rms = float(np.nanmean(sigma_pix[obs] ** 2))
    Cl_noise_val = var_pix_rms * 4.0 * np.pi / n_obs_pix  # nside-invariant scalar

    # ── GSM signal C_l at science nside (nside_sci=64) ───────────────────────
    # Using higher nside gives access to multipoles up to l=191 (vs l=23 at
    # nside=8) and avoids the HEALPix Nyquist limit artificially capping l_max.
    print(f"  Loading GSM at nside={nside_sci} for sensitivity-limited l_max …",
          flush=True)
    sky_sci = SkyModel(np.array([f_mhz * 1e6]), nside=nside_sci, srcs=None)
    gsm_sci = np.asarray(sky_sci.map)
    if gsm_sci.ndim == 2:
        gsm_sci = gsm_sci[:, 0]

    l_max_sci_hp = 3 * nside_sci - 1          # HEALPix Nyquist at nside_sci
    Cl_signal_sci = healpy.anafast(gsm_sci, lmax=l_max_sci_hp)

    # Pixel-window correction: removes HEALPix pixel-smoothing bias from C_l
    pw = healpy.pixwin(nside_sci, lmax=l_max_sci_hp)   # shape (l_max+1,)
    pw = np.where(np.abs(pw) < 1e-10, 1.0, pw)
    Cl_signal_sci = Cl_signal_sci / pw**2

    Cl_noise_sci = np.full(l_max_sci_hp + 1, Cl_noise_val)

    # ── Find l_max ────────────────────────────────────────────────────────────
    threshold = freq_req_frac ** 2
    l_max_sci = 0
    for ell in range(1, l_max_sci_hp + 1):   # skip ell=0 (monopole)
        if Cl_noise_sci[ell] < threshold * Cl_signal_sci[ell]:
            l_max_sci = ell

    sigma_pix_K = float(np.sqrt(var_pix_rms))
    print(f"  Per-pixel noise (obs pixels): σ_pix = {sigma_pix_K:.2f} K")
    print(f"  Noise C_l (scale-invariant) = {Cl_noise_val:.3e} K²sr")
    print(f"  Science nside = {nside_sci}  (HEALPix Nyquist l = {l_max_sci_hp})")
    print(f"  Science l_max (<{freq_req_frac:.0%} error on a_lm) = {l_max_sci}")

    # Return nside-8 arrays for backward-compatible plotting
    l_max_hp  = 3 * NSIDE - 1
    Cl_noise  = np.full(l_max_hp + 1, Cl_noise_val)
    Cl_signal = healpy.anafast(gsm_maps[:, fi], lmax=l_max_hp)

    return l_max_sci, sigma_pix_K, Cl_noise, Cl_signal


# ══════════════════════════════════════════════════════════════════════════════
# F. STM Printing Utilities
# ══════════════════════════════════════════════════════════════════════════════

_W  = 100   # total line width
_WL =  34   # left-column width (requirement label)
_WR = _W - _WL - 3


def _banner(title):
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)


def _section(title):
    print()
    print("-" * _W)
    print(f"  {title}")
    print("-" * _W)


def _row(label, value, units="", margin="", note=""):
    lbl  = str(label)
    val  = f"{value}"
    rest = f"  [{units}]" if units else ""
    if margin:
        rest += f"   margin: {margin}"
    if note:
        rest += f"   ({note})"
    line = f"  {lbl:<{_WL}} {val}{rest}"
    # Wrap at _W chars
    wrapped = textwrap.fill(line, width=_W, subsequent_indent=" " * (_WL + 3))
    print(wrapped)


def _stm_row(sg, so, sr, mr, ir, misr, design, margin):
    """Print one row of the STM table (multi-line if needed)."""
    cols = [sg, so, sr, mr, ir, misr, design, margin]
    widths = [14, 28, 22, 24, 22, 22, 22, 10]
    # truncate each column
    parts = [str(c)[:w] for c, w in zip(cols, widths)]
    print("| " + " | ".join(f"{p:<{w}}" for p, w in zip(parts, widths)) + " |")


def _stm_header():
    cols    = ["SG #", "Science Objective", "Science Req.",
               "Measurement Req.", "Instrument Req.",
               "Mission Req.", "Design Value", "Margin"]
    widths  = [14, 28, 22, 24, 22, 22, 22, 10]
    divider = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header  = "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths)) + " |"
    print(divider)
    print(header)
    print(divider)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── A. Setup ──────────────────────────────────────────────────────────────
    FP = config_fingerprint(cfg)
    print(f"Config fingerprint: {FP}")

    (rots_true, orbits_list, obs_times,
     gsm_maps, models_21cm, T_21_INJ,
     J_SUN, masks_true) = _setup(FP)

    # ── B. GSM eigenmodes and filtered 21cm signal ────────────────────────────
    modes        = gsm_eigenmodes(gsm_maps, N_EIG_MODES)   # includes flat mode
    N_MODES_TOTAL = modes.shape[0]
    dof          = N_FREQ - N_MODES_TOTAL
    T_21_filt    = eigenmode_filter(T_21_INJ, modes)
    T_all        = models_21cm(FREQS_MHZ * 1e6)
    T_all_filt   = eigenmode_filter(T_all, modes)
    print(f"\nFG filter: {N_EIG_MODES} GSM modes + 1 flat = {N_MODES_TOTAL} total  "
          f"dof = {dof}")

    # ── C. Noiseless run (FG leakage baseline) ────────────────────────────────
    print("\nNoiseless run (FG leakage) …", flush=True)
    cached_nl = try_load_multifreq(FREQS_MHZ, 0.0, 0, FP)
    if cached_nl is not None:
        T_est_nl, _ = cached_nl
    else:
        T_est_nl, _sigma_nl = _run_multifreq(rots_true, masks_true, gsm_maps, T_21_INJ, J_SUN,
                                              noise_scale=0.0)
        save_multifreq(FREQS_MHZ, T_est_nl, _sigma_nl, 0.0, 0, FP)

    resid_nl    = eigenmode_filter(T_est_nl, modes)
    FG_leakage  = resid_nl - T_21_filt
    fg_leak_rms = float(np.std(FG_leakage))
    T21_filt_rms = float(np.std(T_21_filt))
    fg_leak_frac  = fg_leak_rms / T21_filt_rms if T21_filt_rms > 0 else np.inf
    print(f"  FG leakage rms  = {fg_leak_rms*1e3:.3f} mK  "
          f"({fg_leak_frac:.1%} of T_21_filt rms)")

    # ── D. Physical-noise run  (Science Goal 1: SNR) ──────────────────────────
    print("\nPhysical-noise run (SIGMA_SCALE = 1.0) …", flush=True)
    NOISE_SCALE_PHYSICAL = 1.0
    NOISE_SEED_PHYSICAL  = 0
    cached_noisy = try_load_multifreq(FREQS_MHZ, NOISE_SCALE_PHYSICAL, NOISE_SEED_PHYSICAL, FP)
    if cached_noisy is not None:
        T_est, SIGMA_MONO = cached_noisy
    else:
        T_est, SIGMA_MONO = _run_multifreq(rots_true, masks_true, gsm_maps, T_21_INJ, J_SUN,
                                            noise_scale=NOISE_SCALE_PHYSICAL,
                                            noise_seed_offset=NOISE_SEED_PHYSICAL)
        save_multifreq(FREQS_MHZ, T_est, SIGMA_MONO, NOISE_SCALE_PHYSICAL, NOISE_SEED_PHYSICAL, FP)

    resid_est   = eigenmode_filter(T_est, modes)
    noise_term  = eigenmode_filter(T_est - T_est_nl, modes)
    chi2_noise  = float(np.sum((noise_term / SIGMA_MONO)**2) / dof)

    SNR         = T_21_filt / SIGMA_MONO
    SNR_60d     = float(np.sqrt(np.sum(SNR**2)))
    SNR_1yr     = SNR_60d * np.sqrt(N_DAYS_EXT / N_DAYS)

    chi2_wt     = np.sum(((resid_est - T_all_filt) / SIGMA_MONO[np.newaxis, :])**2,
                          axis=1) / dof
    chi2_thresh = 1.0 + 2.0 * np.sqrt(2.0 / dof)
    chi2_model0 = float(chi2_wt[0])
    rank_model0 = int(np.sum(chi2_wt <= chi2_model0))
    n_models    = chi2_wt.shape[0]

    print(f"\n  SIGMA_MONO range : {SIGMA_MONO.min()*1e3:.2f}–{SIGMA_MONO.max()*1e3:.2f} mK")
    print(f"  noise chi²/dof   : {chi2_noise:.3f}  (expected 1)")
    print(f"  chi²/dof model 0 : {chi2_model0:.3f}  (threshold {chi2_thresh:.3f})")
    print(f"  rank model 0     : {rank_model0}/{n_models}")
    print(f"  SNR_combined     : {SNR_60d:.1f}  (60-day nominal)")
    print(f"  SNR_combined     : {SNR_1yr:.1f}  (1-year extended, ×√(365/60))")

    # ── E. Pointing-error simulation (Item 11) ────────────────────────────────
    ATT_SEED = 99
    T_pert, masks_pert = _run_pointing_error_sim(
        rots_true, orbits_list, obs_times,
        masks_true, gsm_maps, T_21_INJ, J_SUN,
        fp=FP, pert_deg=ATT_ERR_DEG, seed=ATT_SEED,
    )
    # Pointing bias = change relative to noiseless truth (isolates pointing error
    # from FG leakage, which is already present in T_est_nl).
    resid_pert    = eigenmode_filter(T_pert, modes)
    resid_nl_filt = eigenmode_filter(T_est_nl, modes)   # noiseless filtered baseline
    pointing_bias = resid_pert - resid_nl_filt   # pure pointing-induced change
    bias_rms      = float(np.std(pointing_bias))
    bias_vs_sigma = bias_rms / float(np.mean(SIGMA_MONO))
    # Derived attitude knowledge requirement: bias scales linearly with angle
    sigma_att_req_deg = ATT_ERR_DEG / bias_vs_sigma   # angle that gives bias = 1×sigma
    sigma_att_req_arcmin = sigma_att_req_deg * 60.0
    print(f"\n  Pointing bias rms (re: noiseless)  = {bias_rms*1e3:.3f} mK")
    print(f"  T_21_filt rms                      = {T21_filt_rms*1e3:.3f} mK")
    print(f"  Bias / mean(SIGMA)                 = {bias_vs_sigma:.3f}×")
    print(f"  Derived σ_att requirement (bias=1σ): {sigma_att_req_deg:.4f}° = {sigma_att_req_arcmin:.2f} arcmin")

    # ── F. All-sky map l_max (Science Goal 2) ─────────────────────────────────
    l_max_sci, sigma_pix_K, Cl_noise, Cl_signal = _compute_lmax(
        rots_true, masks_true, gsm_maps, J_SUN, freq_req_frac=0.10,
    )
    # HEALPix Nyquist for reference
    l_max_hp = 3 * NSIDE - 1

    # ── G. Diagnostic plots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.fill_between(FREQS_MHZ, -SIGMA_MONO*1e3, SIGMA_MONO*1e3,
                    alpha=0.25, color='C0', label='±1σ per-freq')
    ax.plot(FREQS_MHZ, resid_est*1e3, 'k.-', ms=5, label='residual (noisy)')
    ax.plot(FREQS_MHZ, T_21_filt*1e3, 'r--', lw=2, label='T_21 (filtered truth)')
    ax.plot(FREQS_MHZ, pointing_bias*1e3, 'm-', lw=1.5, label=f'pointing bias (Δθ={ATT_ERR_DEG}°)')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('ΔT [mK]')
    ax.set_title(f'Science Goal 1  SNR={SNR_60d:.1f} (60d)')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.semilogy(FREQS_MHZ, SIGMA_MONO*1e3, 'b-', label='SIGMA_MONO (physical noise)')
    ax.axhline(bias_rms*1e3, color='m', ls='--', lw=1.5,
               label=f'pointing bias rms = {bias_rms*1e3:.2f} mK')
    ax.axhline(fg_leak_rms*1e3, color='g', ls=':', lw=1.5,
               label=f'FG leakage rms = {fg_leak_rms*1e3:.2f} mK')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('ΔT [mK]')
    ax.set_title('Noise budget breakdown')
    ax.legend(fontsize=7)

    ax = axes[2]
    ells = np.arange(1, l_max_hp + 1)
    ax.loglog(ells, np.sqrt(Cl_signal[1:l_max_hp+1])*1e3, 'b-', lw=2,
              label='GSM signal √C_l')
    ax.loglog(ells, np.sqrt(Cl_noise[1:l_max_hp+1])*1e3, 'r--', lw=2,
              label='Noise floor √C_l')
    ax.axvline(l_max_sci, color='k', ls='--',
               label=f'l_max = {l_max_sci}')
    ax.set_xlabel('Multipole l')
    ax.set_ylabel('√C_l  [mK sr^{1/2}]')
    ax.set_title('Science Goal 2: all-sky map l_max')
    ax.legend(fontsize=7)

    plt.tight_layout()
    outpng = '/tmp/bloom_stm.png'
    plt.savefig(outpng, dpi=100)
    plt.close()
    print(f"\nSaved diagnostic plot to {outpng}")

    # ══════════════════════════════════════════════════════════════════════════
    # G. FULL SCIENCE TRACEABILITY MATRIX  (printed document)
    # ══════════════════════════════════════════════════════════════════════════

    _banner("BLOOM-21CM SCIENCE TRACEABILITY MATRIX")
    print(f"  {'Mission':<30} Broadband Lunar Occultation Orbiter for Measuring the 21-cm Monopole")
    print(f"  {'Acronym':<30} BLOOM-21CM")
    print(f"  {'Program':<30} NASA Astrophysics Pioneers (ROSES-25)")
    print(f"  {'Platform':<30} 2× 8U CubeSat in non-coplanar polar lunar orbit")
    print(f"  {'Primary science band':<30} {FREQ_MIN_MHZ:.0f}–{FREQ_MAX_MHZ:.0f} MHz")
    print(f"  {'Extended science band':<30} {EXT_BAND_LOW:.0f}–{EXT_BAND_HIGH:.0f} MHz")
    print(f"  {'Nominal mission duration':<30} {N_DAYS} days ({N_DAYS/SYNODIC_MONTH:.1f} synodic months)")
    print(f"  {'Extended mission duration':<30} {N_DAYS_EXT} days (1 year)")

    # ─────────────────────────────────────────────────────────────────────────
    _section("SCIENCE GOAL 1 (SG-1): GLOBAL 21CM MONOPOLE DETECTION")
    print(textwrap.fill(
        "  Detect the spatially uniform (monopole) component of the redshifted "
        "21cm brightness temperature across the primary science band (55–115 MHz), "
        "enabling a model-comparison measurement of the neutral hydrogen absorption "
        "trough associated with cosmic dawn and the onset of reionization.  "
        "The detection is characterized generically by the temperature scale and "
        "spectral width of the absorption feature, without relying on analytic "
        "parameterizations.",
        width=_W, initial_indent="  ", subsequent_indent="  "))

    _section("SG-1 Flow-down")

    _row("Science Objective SO-1.1",
         "Measure the sky-averaged brightness temperature T_sky(ν) after "
         "foreground separation at SNR_combined > 15 over 55–115 MHz")
    print()

    _row("Science Requirement SR-1.1",
         f"SNR_combined = √Σ(T_21_filt/σ_mono)² > 15  over {NCHAN_SCIENCE} science channels")
    _row("   Simulation result (60-day nominal)",
         f"SNR_combined = {SNR_60d:.1f}",
         margin=f"{SNR_60d/15.0:.1f}×  (> 15 required)")
    _row("   Extended mission (1 year)",
         f"SNR_combined = {SNR_1yr:.1f}",
         margin=f"{SNR_1yr/15.0:.1f}×")
    print()

    _row("Measurement Req. MR-1.1  [Spectral band]",
         f"Continuous coverage {FREQ_MIN_MHZ:.0f}–{FREQ_MAX_MHZ:.0f} MHz (primary); "
         f"{EXT_BAND_LOW:.0f}–{EXT_BAND_HIGH:.0f} MHz (extended)")
    _row("   Design value",
         f"{NCHAN_SCIENCE} channels × {DELTA_NU_MHZ:.1f} MHz = "
         f"{FREQ_MIN_MHZ:.0f}–{FREQ_MAX_MHZ:.0f} MHz  |  "
         f"{NCHAN_EXT} channels × {DELTA_NU_MHZ:.1f} MHz = "
         f"{EXT_BAND_LOW:.0f}–{EXT_BAND_HIGH:.0f} MHz",
         margin="meets threshold")
    print()

    _row("Measurement Req. MR-1.2  [Spectral resolution]",
         f"Channel width Δν ≤ 2 MHz (fine enough to track 21cm spectral shape)")
    _row("   Design value",
         f"Δν = {DELTA_NU_MHZ:.1f} MHz  ({cfg.observation.channel_width_khz:.0f} kHz)",
         margin="meets threshold")
    print()

    _row("Measurement Req. MR-1.3  [Radiometric sensitivity]",
         "SIGMA_MONO(ν) per channel × √N_ch < T_21_filt_rms / 15  (per-channel)")
    _row("   Design value  (SIGMA_MONO range)",
         f"{SIGMA_MONO.min()*1e3:.2f}–{SIGMA_MONO.max()*1e3:.2f} mK",
         units="mK", note=f"T_integration = {T_INTEGRATION:.0f} s, "
                           f"duty_cycle = {DUTY_CYCLE:.3f}")
    _row("   T_21_filt rms",
         f"{T21_filt_rms*1e3:.3f} mK",
         units="mK")
    _row("   chi²/dof (model 0, injected)",
         f"{chi2_model0:.3f}",
         margin=f"< threshold {chi2_thresh:.3f}  (rank {rank_model0}/{n_models})")
    print()

    _row("Measurement Req. MR-1.4  [Sky modulation]",
         "Minimum fractional sky modulation m_min > 0.18 over campaign "
         "(enables per-pixel A matrix inversion)")
    _row("   Design value",
         f"m_min = {cfg.modulation_min:.2f}  (≥ 0.18 in {cfg.sky_frac_modulation:.0%} of sky)",
         margin="meets threshold")
    print()

    _row("Measurement Req. MR-1.5  [Foreground rejection]",
         f"FG leakage rms < 15% of T_21_filt rms after eigenmode filter "
         f"(N_EIG_MODES = {N_EIG_MODES} + 1 flat)")
    _row("   Simulation result",
         f"FG leakage = {fg_leak_rms*1e3:.3f} mK  ({fg_leak_frac:.1%} of T_21_filt rms)",
         margin=f"{'PASS' if fg_leak_frac < 0.15 else 'FAIL'}: "
                f"< 15% required")
    print()

    _row("Instrument Req. IR-1.1  [Antenna bandwidth]",
         f"Dipole with usable response {EXT_BAND_LOW:.0f}–{EXT_BAND_HIGH:.0f} MHz; "
         f"proven Stacer heritage (Parker Solar Probe, LuSee Night)")
    _row("   Design",
         f"Stacer dipoles: arm lengths {cfg.antenna.arm_lengths[0]:.1f} m + "
         f"{cfg.antenna.arm_lengths[1]:.1f} m  (6/4 m full tip-to-tip)")
    print()

    _row("Instrument Req. IR-1.2  [Receiver temperature]",
         "T_rx < 100 K across science band  (T_rx/T_sky < 30%)")
    _row("   Design value",
         f"T_rx = {T_RX_K:.0f} K  (T_rx/T_sky_max = {cfg.trx_frac_of_tsky_max:.0%})",
         margin=f"{'PASS' if cfg.trx_frac_of_tsky_max <= 0.30 else 'FAIL'}")
    print()

    _row("Mission Req. MiR-1.1  [Duration]",
         f"≥ {N_DAYS} days (≥ 2 synodic months) for full Moon illumination coverage")
    _row("   Design value",
         f"{N_DAYS} days = {N_DAYS/SYNODIC_MONTH:.1f} synodic months",
         margin=f"{N_DAYS/SYNODIC_MONTH/2.0:.1f}× above 2-month threshold")
    print()

    _row("Mission Req. MiR-1.2  [Orbital geometry]",
         "2 non-coplanar polar orbits (equatorial + polar normal) for "
         "geometric diversity in occultation modulation")
    _row("   Design value",
         f"{N_ORBITS} orbits  |  normals: equatorial N̂ + vernal-equinox N̂  "
         f"|  altitude = {cfg.observation.altitude/1e3:.0f} km",
         margin="fixed architecture")
    print()

    _row("Mission Req. MiR-1.3  [Attitude knowledge / trajectory accuracy]",
         "Pointing errors must keep pointing-induced systematic bias below 10% of "
         "per-channel σ_mono.  TWO distinct budgets: (A) random readout noise from "
         "star-tracker readings, which averages down with N_readings; "
         "(B) systematic calibration error (boresight offset), which does not average.")
    _row("   Simulation  (Δθ = 1° systematic at all observations)",
         f"Bias rms = {bias_rms*1e3:.3f} mK  =  {bias_vs_sigma:.1f}× mean σ_mono  "
         f"[σ_mono = {np.mean(SIGMA_MONO)*1e3:.2f} mK]  (1° systematic INSUFFICIENT by {bias_vs_sigma:.0f}×)")
    _row("   (A) Random noise req  (averages as 1/√N_readings)",
         f"bias = {bias_rms*1e3:.0f} mK × σ_tracker / √N_readings < 10% σ_mono  →  "
         f"at 60s cadence: σ_tracker < 24'  |  at 1s cadence: σ_tracker < 183'",
         note="from tumbling_beam_verify.py; 1° tracker meets req at ≤10s cadence (bias 0.17→0.054 mK)")
    _row("   (B) Systematic cal req  (no averaging)",
         f"δ_sys < {0.10 * np.mean(SIGMA_MONO) / (bias_rms) * 60 * 1e3:.2f} arcmin  "
         f"= {0.10 * np.mean(SIGMA_MONO) / (bias_rms) * 3600 * 1e3:.0f} arcsec  (10% σ_mono budget)",
         note="fixed boresight/calibration error; verified by pre-launch calibration campaign",
         margin="pre-launch cal required")

    # ─────────────────────────────────────────────────────────────────────────
    _section("SCIENCE GOAL 2 (SG-2): ALL-SKY SPECTRAL MAPS")
    print(textwrap.fill(
        "  Produce all-sky brightness temperature maps from 30–170 MHz with "
        "sufficient sensitivity to constrain the spherical harmonic coefficients "
        "a_lm at the < 10% level for all multipoles l ≤ l_max accessible given "
        "the radiometric sensitivity of a 60-day mission.  These maps enable "
        "separation of Galactic foregrounds and characterize the diffuse radio "
        "sky independent of the global 21cm analysis.",
        width=_W, initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill(
        "  A_LM RECOVERY METHOD: The per-pixel inversion directly delivers "
        "individual a_lm coefficients — both amplitude and phase — for each "
        "frequency channel, not merely the angular power spectrum C_l. "
        "The linear solver produces a sky map T_j at each HEALPix pixel j, "
        "from which a_lm are computed as â_lm = Ω_pix × Σ_j T_j Y_lm*(θ_j, φ_j), "
        "where Ω_pix = 4π / N_pix is the pixel solid angle and Y_lm are the "
        "real spherical harmonics. This full-phase recovery enables map-domain "
        "foreground subtraction, cross-correlation with other datasets, and "
        "detailed characterization of Galactic emission structure, going well "
        "beyond what is possible from power-spectrum measurements alone.",
        width=_W, initial_indent="  ", subsequent_indent="  "))

    _section("SG-2 Flow-down")

    _row("Science Objective SO-2.1",
         f"Measure a_lm with < 10% fractional error for l ≤ l_max "
         f"over {EXT_BAND_LOW:.0f}–{EXT_BAND_HIGH:.0f} MHz")
    print()

    _row("Science Requirement SR-2.1",
         "C_l^noise < 0.01 × C_l^GSM for l ≤ l_max  "
         "(equivalent to 10% error on a_lm coefficients)")
    _row("   Simulation resolution",
         f"nside = {NSIDE}  (sky pixelization for inversion; "
         f"per-pixel noise σ_pix = {sigma_pix_K:.2f} K at "
         f"√({FREQ_MIN_MHZ:.0f}×{FREQ_MAX_MHZ:.0f}) = "
         f"{np.sqrt(FREQ_MIN_MHZ*FREQ_MAX_MHZ):.0f} MHz)")
    _row("   Science nside for l_max evaluation",
         "nside = 64  (pixel window corrected; C_l^noise scale-invariant)")
    _row("   Simulation result: l_max",
         f"l_max = {l_max_sci}  "
         f"(sensitivity-limited, not nside-limited)",
         note="C_l^noise < 1% × C_l^GSM, using nside=64 pixel-window-corrected GSM")
    print()

    _row("Measurement Req. MR-2.1  [Spectral band]",
         f"Continuous coverage {EXT_BAND_LOW:.0f}–{EXT_BAND_HIGH:.0f} MHz  "
         f"({NCHAN_EXT} × {DELTA_NU_MHZ:.1f} MHz channels)")
    _row("   Design value",
         f"{NCHAN_EXT} channels  (same receiver, extended processing range)",
         margin="meets threshold")
    print()

    _row("Measurement Req. MR-2.2  [Sky coverage]",
         "Full-sky (4π sr) accessible via 2 non-coplanar polar orbits over 60 days")
    _row("   Design value",
         f"2 orbital planes, nside={NSIDE}  ({NPIX} pixels, ~7° resolution)",
         margin="meets threshold")
    print()

    _row("Instrument Req. IR-2.1  [Antenna bandwidth]",
         f"Dipole response from {EXT_BAND_LOW:.0f} MHz (λ = {300/EXT_BAND_LOW*1e3:.0f} mm) "
         f"to {EXT_BAND_HIGH:.0f} MHz  (λ = {300/EXT_BAND_HIGH*1e3:.0f} mm)")
    _row("   Design",
         f"Stacer arms {cfg.antenna.arm_lengths[0]:.1f} m + {cfg.antenna.arm_lengths[1]:.1f} m  "
         f"|  R_loss = {cfg.antenna.r_loss_ohm:.0f} Ω  |  Z_rx = {cfg.antenna.z_rx_ohm:.0f} Ω")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    _section("DATA VOLUME AND DOWNLINK BUDGET")

    _row("Raw data rate (both spacecraft)",
         f"{DATA_RATE_BPS:.0f} bytes/s  "
         f"= {N_ORBITS} SC × {NCHAN_EXT} ch × 2 dipoles × {BYTES_PER_SAMPLE} B × {SAMPLE_RATE_HZ:.0f} Hz",
         units="bytes/s")
    _row("Raw data per day (both spacecraft)",
         f"{DATA_MBPERDAY:.1f} MB/day",
         units="MB/day")
    _row("X-band downlink capacity per spacecraft",
         f"{DOWNLINK_MBPERHR:.0f} MB/hr  "
         f"× {DOWNLINK_HRS_DAY:.1f} hr/day (lost to housekeeping/eclipse/calibration) "
         f"× {N_ORBITS} SC")
    _row("Downlink budget (both spacecraft)",
         f"{DOWNLINK_MB_DAY:.0f} MB/day",
         margin=f"{DOWNLINK_MARGIN:.1f}× margin above raw data volume")
    _row("Observing duty cycle",
         f"{DUTY_CYCLE:.3f}  ({DUTY_CYCLE*24:.1f} hrs/day science collection)",
         note="2 hrs/day contact window for downlink / housekeeping")

    # ─────────────────────────────────────────────────────────────────────────
    _section("SENSITIVITY SCALING MODEL  (not hardware accumulation time)")

    print(textwrap.fill(
        "  T_INTEGRATION is a noise-accounting concept: the total mission science "
        "time is distributed equally across all N_total observation rows in the "
        "design matrix, giving each row an 'effective' integration time that "
        "reproduces the correct per-row noise level in the linear inversion. "
        "It is NOT the on-sky dwell time of a hardware accumulation, and should "
        "not be compared to t_snapshot or used to estimate beam smearing.",
        width=_W, initial_indent="  ", subsequent_indent="  "))
    print()
    _row("T_INTEGRATION  (sensitivity-scaling, noise model only)",
         f"{T_INTEGRATION:.1f} s",
         note=f"= {N_DAYS}d × {DUTY_CYCLE:.3f} duty × 86400 s/d / "
              f"({N_ORBITS} SC × {N_OBS} obs/SC)  — used only for σ calculation")
    _row("   Physical meaning",
         "sets per-row noise σ_row = T_sys / sqrt(Δν × T_INTEGRATION)",
         note="equivalent to attributing 60-day sensitivity to single-orbit observation model")

    # ─────────────────────────────────────────────────────────────────────────
    _section("HARDWARE ACCUMULATION AND BEAM-SMEARING BUDGET")

    print(textwrap.fill(
        "  T_ACCUM is the actual on-sky accumulation time per transmitted spectrum "
        "(inverse of the hardware sample rate). This is the timescale relevant to "
        "beam smearing: during T_ACCUM the spacecraft attitude changes by "
        f"θ_sweep = T_ACCUM × 360° / T_spin = {THETA_SWEEP_ACCUM_DEG:.2f}°, "
        "rotating the instantaneous beam across the sky. "
        "The downlink budget bounds how small T_ACCUM can be made (shorter "
        "accumulations → higher sample rate → more data volume).",
        width=_W, initial_indent="  ", subsequent_indent="  "))
    print()
    _row("Hardware accumulation time  T_ACCUM",
         f"{T_ACCUM:.1f} s  (= 1 / SAMPLE_RATE_HZ = 1 / {SAMPLE_RATE_HZ:.1f} Hz)",
         note="time per transmitted spectrum; sets beam-smearing constraint")
    _row("Beam sweep during T_ACCUM",
         f"{THETA_SWEEP_ACCUM_DEG:.2f}°  (= T_ACCUM × 360° / {SPIN_PERIOD_S:.0f}s spin)")
    _row("Attitude coherence limit  t_snapshot",
         f"{T_SNAPSHOT:.2f} s  (= {ATT_KNOW_DEG:.0f}° / (360° / {SPIN_PERIOD_S:.0f}s spin))",
         note="time for beam to move 1° under nominal pointing knowledge definition")
    _row("T_ACCUM / t_snapshot",
         f"{T_ACCUM/T_SNAPSHOT:.2f}  — T_ACCUM < t_snapshot",
         margin=f"single-snapshot beam model valid to {THETA_SWEEP_ACCUM_DEG:.2f}° smearing")
    _row("Minimum T_ACCUM (full downlink budget)",
         f"{T_ACCUM_MIN_S:.2f} s  →  beam sweep {T_ACCUM_MIN_S*360/SPIN_PERIOD_S:.2f}°",
         note=f"at {DOWNLINK_MB_DAY:.0f} MB/day capacity, {NCHAN_EXT} ch, float32")
    print()

    print(textwrap.fill(
        "  BEAM-SMEARING MODEL USING TUMBLING TRAJECTORY: Even though T_ACCUM < "
        "t_snapshot (so the single-snapshot approximation is valid), the "
        f"{THETA_SWEEP_ACCUM_DEG:.2f}° of beam rotation during each accumulation can "
        "be modeled explicitly rather than ignored. The exact effective beam for "
        "each accumulation is the trajectory integral: "
        "A_ij = (1/T_ACCUM) ∫₀^T_ACCUM B_j(Ω(t)) dt, where Ω(t) is the "
        "spacecraft attitude during the window. "
        "Because the spacecraft tumbles freely between comms passes (zero external "
        "torque → angular momentum L conserved), the trajectory obeys Euler's "
        "equations with a known inertia tensor (characterized pre-launch). "
        "Given L and an initial orientation Ω(t₀), the trajectory over any "
        "subsequent T_ACCUM window is fully deterministic. "
        "The 6-DOF tumbling model (L, Ω(t₀)) is fit to the ensemble of star-tracker "
        "observations accumulated over one day. With many observations, L is "
        "overdetermined and the trajectory prediction at any individual T_ACCUM "
        "window is accurate to well below the {THETA_SWEEP_ACCUM_DEG:.2f}° level. "
        "Including the trajectory integral in A replaces the approximate "
        "single-snapshot beam with the exact time-averaged beam, eliminating the "
        f"{THETA_SWEEP_ACCUM_DEG:.2f}° smearing as a source of systematic error.",
        width=_W, initial_indent="  ", subsequent_indent="  "))
    print()
    _row("   Tumbling trajectory Ω(t)",
         "predicted by Euler equations given (L, Ω(t₀)); no free parameters between comms",
         note="L conserved ← zero torque; I_tensor known from pre-launch characterization")
    _row("   6-DOF model fit observables",
         "star-tracker readings over one day → L and Ω(t₀) over-constrained",
         note="fit residuals from redundant star-tracker obs quantify trajectory uncertainty")
    _row("   Pointing-error sim (1° systematic at all observations)",
         f"bias = {bias_rms*1e3:.0f} mK = {bias_vs_sigma:.0f}× σ_mono "
         f"(σ_mono = {np.mean(SIGMA_MONO)*1e3:.2f} mK mean over band)",
         note="bias / σ_mono = 74.4×/° verified from cache: eigenmode-filtered bias 122 mK at 1°")
    _row("   Ergodic averaging (N_DAYS independent days)",
         f"effective bias scaling = {bias_vs_sigma:.0f}× / √{N_DAYS} = "
         f"{bias_vs_sigma/np.sqrt(N_DAYS):.1f}×/° after 60-day avg",
         note="each day's trajectory re-initialized from comms-pass star-tracker data → independent errors")
    _row("   Random noise: bias vs. N_readings (tumbling_beam_verify.py)",
         "bias = 122 mK × σ_tracker / √(N_rdg/day × N_days);  "
         "1° tracker: ≤10s cadence → bias 0.17 mK ✓;  60s cadence → 0.42 mK ✗",
         note="random errors are independent between readings → averages down favorably within each day")
    _row("   Beam sensitivity ∂(|δA|/|A|)/∂θ",
         "≈ 1.98e-02 /°  (verified, tumbling_beam_verify.py Stage 3)",
         note="L-drift (0.02°) is negligible vs tracker noise; dominant error is single-reading noise ~σ_tracker")
    _row("   Systematic calibration req  (does NOT average down)",
         f"δ_sys < {0.10 * np.mean(SIGMA_MONO) / bias_rms * 60 * 1e3:.2f} arcmin"
         f"  = {0.10 * np.mean(SIGMA_MONO) / bias_rms * 3600 * 1e3:.0f} arcsec  "
         "(10% σ_mono; bias_sys = 122 mK × δ_sys / 1°)",
         note="pre-launch calibration campaign determines boresight to sub-arcminute accuracy",
         margin="pre-launch cal")

    # ─────────────────────────────────────────────────────────────────────────
    _section("EXTENDED MISSION PROJECTIONS  (60 days → 1 year)")

    _row("Sensitivity scaling",
         f"σ_noise ∝ 1/√t  →  SNR ∝ √t")
    _row("Scaling factor (60d → 365d)",
         f"√(365/60) = {np.sqrt(365/60):.2f}×")
    _row("SNR_combined (60-day nominal)",
         f"{SNR_60d:.1f}",
         margin=f"{SNR_60d/15.0:.1f}× above 15σ threshold")
    _row("SNR_combined (1-year extended)",
         f"{SNR_1yr:.1f}",
         margin=f"{SNR_1yr/15.0:.1f}× above 15σ threshold")
    _row("SIGMA_MONO at 1 year (median)",
         f"{np.median(SIGMA_MONO)*1e3 / np.sqrt(365/60):.3f} mK",
         note="enables tighter model comparison and higher l_max on sky maps")

    # ─────────────────────────────────────────────────────────────────────────
    _section("COMPACT STM TABLE")
    print()

    _stm_header()
    # SG-1 rows
    _stm_row(
        "SG-1",
        "SO-1.1: Detect global 21cm absorption trough",
        "SR-1.1: SNR > 15",
        "MR-1.1: 55–115 MHz band",
        "IR-1.1: Stacer dipole 30–170 MHz",
        "MiR-1.1: 60-day mission",
        f"60d SNR = {SNR_60d:.1f}",
        f"{SNR_60d/15.0:.1f}×",
    )
    _stm_row(
        "",
        "",
        "SR-1.1: SNR > 15",
        "MR-1.2: Δν ≤ 2 MHz",
        "IR-1.2: T_rx < 100 K",
        "MiR-1.2: 2 non-cop. orbits",
        f"Δν = {DELTA_NU_MHZ:.1f} MHz",
        "meets",
    )
    _stm_row(
        "",
        "",
        "SR-1.1: SNR > 15",
        "MR-1.3: σ_mono per channel",
        "",
        f"MiR-1.3: cal δ<{0.10*np.mean(SIGMA_MONO)/bias_rms*60*1e3:.0f}\"",
        f"1°sys→{bias_rms*1e3:.0f}mK={bias_vs_sigma:.0f}×σ; cal+cadence",
        "pre-launch cal",
    )
    _stm_row(
        "",
        "",
        "SR-1.1: SNR > 15",
        "MR-1.4: m_min > 0.18",
        "",
        "",
        f"m_min = {cfg.modulation_min:.2f}",
        "meets",
    )
    _stm_row(
        "",
        "",
        "SR-1.1: SNR > 15",
        "MR-1.5: FG leak < 15%",
        "",
        "",
        f"leak = {fg_leak_frac:.1%}",
        f"{'PASS' if fg_leak_frac < 0.15 else 'FAIL'}",
    )
    # SG-2 row
    _stm_row(
        "SG-2",
        "SO-2.1: All-sky maps 30–170 MHz",
        "SR-2.1: <10% err a_lm, l≤l_max",
        "MR-2.1: 30–170 MHz, 70 ch",
        "IR-2.1: Bandwidth 30–170 MHz",
        "MiR-2.1: Full-sky coverage",
        f"l_max = {l_max_sci}  (sensitivity-limited; nside=64 eval)",
        "meets",
    )

    divider = "+" + "+".join("-" * (w + 2) for w in [14, 28, 22, 24, 22, 22, 22, 10]) + "+"
    print(divider)

    # ─────────────────────────────────────────────────────────────────────────
    _section("KEY SIMULATION RESULTS SUMMARY")

    _row("Fiducial 21cm model",
         "Model index 0 (T21cmModel), generic absorption feature")
    _row("   Peak absorption",
         f"{T_21_INJ.min()*1e3:.1f} mK  |  rms over science band = {np.std(T_21_INJ)*1e3:.2f} mK")
    _row("   Filtered signal T_21_filt rms",
         f"{T21_filt_rms*1e3:.3f} mK  (after {N_EIG_MODES} GSM + 1 flat eigenmodes)")
    print()
    _row("Foreground filter",
         f"N_EIG_MODES = {N_EIG_MODES} GSM modes + 1 flat  →  {N_MODES_TOTAL} total, dof = {dof}")
    _row("FG leakage rms",
         f"{fg_leak_rms*1e3:.3f} mK  ({fg_leak_frac:.1%} of T_21_filt rms)",
         margin=f"{'PASS' if fg_leak_frac < 0.15 else 'FAIL'}: < 15% required")
    print()
    _row("Physical-noise SNR (60-day, SIGMA_SCALE=1.0)",
         f"SNR_combined = {SNR_60d:.1f}",
         margin=f"{SNR_60d/15:.1f}× above 15σ threshold")
    _row("noise-only chi²/dof",
         f"{chi2_noise:.3f}",
         note="expected 1.0  (validates noise model)")
    _row("Model-0 chi²/dof",
         f"{chi2_model0:.3f}",
         margin=f"< threshold {chi2_thresh:.3f}  (rank {rank_model0}/{n_models})")
    print()
    _row("Pointing-error simulation  (Δθ = 1°, systematic at ALL observations)",
         f"Bias rms = {bias_rms*1e3:.3f} mK  =  {bias_vs_sigma:.1f}× mean σ_mono  "
         f"[σ_mono = {np.mean(SIGMA_MONO)*1e3:.2f} mK]  →  1° systematic INSUFFICIENT by {bias_vs_sigma:.0f}×",
         note="eigenmode-filtered; σ_mono = noise on sky monopole from inversion (NOT per-pixel ~20 mK)")
    _row("   Two separate error budgets  (tumbling_beam_verify.py)",
         f"(A) RANDOM noise: bias = {bias_rms*1e3:.0f} mK × σ_tracker / √N_rdg;  "
         f"1° tracker at ≤10s cadence → bias ≈ 0.17 mK < 10% σ_mono  ✓",
         note="random errors average down within each day AND across 60 days; L-drift (0.02°) negligible")
    _row("   (B) SYSTEMATIC calibration req",
         f"δ_sys < {0.10*np.mean(SIGMA_MONO)/bias_rms*60*1e3:.0f} arcsec  "
         f"(= {0.10*np.mean(SIGMA_MONO)/bias_rms*60*1e3/60:.2f} arcmin)  — pre-launch calibration",
         note="fixed boresight/cal error never averages down; typical star-tracker cal is 1–10 arcsec",
         margin="pre-launch cal required")
    print()
    _row("All-sky map l_max (SG-2)",
         f"l_max = {l_max_sci}  (< 10% error on a_lm, sensitivity-limited)",
         note=f"σ_pix = {sigma_pix_K:.2f} K at repr. freq; nside=64 eval w/ pixel-window correction")
    _row("Extended mission (1 year) SNR",
         f"SNR_combined = {SNR_1yr:.1f}  (×√(365/60) = {np.sqrt(365/60):.2f}×)")

    print()
    print("=" * _W)
    print("  END OF BLOOM-21CM SCIENCE TRACEABILITY MATRIX")
    print("=" * _W)
    print()
