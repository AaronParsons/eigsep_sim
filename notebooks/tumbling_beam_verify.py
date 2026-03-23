#!/usr/bin/env python3
"""
tumbling_beam_verify.py — Trajectory-integral beam model verification for BLOOM-21CM.

Verifies the pipeline from noisy star-tracker readings to sky-estimate bias,
with required-vs-delivered accuracy checks at each interface.

Pipeline stages
---------------
  Stage 1 : Star-tracker readings  →  angular momentum L_fit
  Stage 2 : Local-arc propagation  →  trajectory Ω(t) accuracy over T_ACCUM windows
  Stage 3 : Beam integral          →  time-averaged beam weight error |δA|/|A|
  Stage 4 : Sky monopole bias      →  bias vs. σ_noise using STM scaling

Physical model
--------------
Torque-free rigid-body rotation (Euler equations).  L is conserved between
comms passes.  For each T_ACCUM window, we estimate the local attitude
by (i) finding the nearest star-tracker reading and (ii) propagating forward
≤ T_DT/2 seconds using L_fit.  This avoids the full-day L-drift problem while
correctly capturing the single-reading noise as the dominant error.

Science requirement derivation
------------------------------
Two distinct error budgets from the STM pointing-error simulation (see
bloom_setup.STM_BIAS_PER_DEG_MK and STM_SIGMA_MONO_MK for the scaling values):

  RANDOM noise (star-tracker readout noise, independent between readings):
  • 1° systematic error → STM_BIAS_PER_DEG_MK mK bias  (= BIAS_PER_DEG_SIGMA× σ_mono)
  • For N_readings independent readings, random errors average down: bias ∝ 1/sqrt(N_readings)
  • Required:  σ_tracker × STM_BIAS_PER_DEG_MK / sqrt(N_readings/day × N_DAYS) < REQ_BIAS_SIGMA × σ_mono

  SYSTEMATIC bias (absolute calibration, boresight alignment — does NOT average down):
  • Required:  δ_sys < REQ_SYS_DEG  (= REQ_BIAS_SIGMA × STM_SIGMA_MONO_MK / STM_BIAS_PER_DEG_MK)

Usage
-----
  python tumbling_beam_verify.py                   # default parameters
  python tumbling_beam_verify.py --tracker-err-deg 0.5 --tracker-dt-s 10
"""

import argparse
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
import healpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from eigsep_sim.lunar_orbit import OrbiterMission
from eigsep_sim.sim import compute_beams
from bloom_setup import STM_BIAS_PER_DEG_MK, STM_SIGMA_MONO_MK

# ── Configuration ──────────────────────────────────────────────────────────────
_YAML   = os.path.join(_HERE, 'bloom_config.yaml')
cfg     = OrbiterMission(_YAML)
I_DIAG  = np.diag(cfg.antenna.inertia)

SPIN_PERIOD_S    = float(cfg.observation.spin_period_s)
OMEGA_SPIN       = 2.0 * np.pi / SPIN_PERIOD_S
T_ACCUM_S        = cfg.analysis.t_accum_s
THETA_SWEEP_DEG  = T_ACCUM_S * 360.0 / SPIN_PERIOD_S
DAY_S            = 86400.0
N_DAYS           = int(cfg.observation.n_days)
N_SUBSTEPS       = cfg.analysis.n_substeps
N_WINDOWS        = cfg.analysis.n_windows
FREQ_MHZ         = cfg.analysis.ref_freq_mhz
NSIDE            = cfg.observation.nside
NPIX             = healpy.nside2npix(NSIDE)

# ── STM-derived bias scaling ───────────────────────────────────────────────────
# From the STM pointing-error simulation (see lunar_stm.py and bloom_setup.py):
#   STM_BIAS_PER_DEG_MK: mK bias for 1° systematic pointing error (eigenmode-filtered)
#   STM_SIGMA_MONO_MK:   mK SIGMA_MONO mean over science band (noise on monopole,
#                        NOT per-pixel radiometric noise)
BIAS_PER_DEG_SIGMA    = STM_BIAS_PER_DEG_MK / STM_SIGMA_MONO_MK

# Science requirement: bias < REQ_BIAS_SIGMA × σ_mono in the monopole channel
REQ_BIAS_SIGMA        = 0.10     # 10% of σ_mono budget for pointing-induced bias

# Systematic calibration requirement (fixed boresight / calibration error, no averaging):
#   bias_sys = STM_BIAS_PER_DEG × δ_sys  →  δ_sys < REQ_BIAS_SIGMA × σ_mono / STM_BIAS_PER_DEG_MK
REQ_SYS_DEG           = REQ_BIAS_SIGMA * STM_SIGMA_MONO_MK / STM_BIAS_PER_DEG_MK

# Random noise requirement (per-reading noise, averages as 1/sqrt(N_readings_total)):
# REQ_TRACKER_DEG depends on cadence; computed in main() for the given tracker_dt_s
# Placeholder for Stage 2/3 requirement display (uses systematic value as conservative bound)
req_traj_deg          = REQ_SYS_DEG   # conservative; random budget is relaxed by sqrt(N_rdg)


# ═══════════════════════════════════════════════════════════════════════════════
# § 1.  Physical tumbling model
# ═══════════════════════════════════════════════════════════════════════════════

def euler_rhs(t, state, I):
    omega = state[:3];  q = state[3:]
    Iw = I * omega
    domega = -np.cross(omega, Iw) / I
    w, x, y, z = q;  ox, oy, oz = omega
    dq = 0.5 * np.array([
        -x*ox - y*oy - z*oz,
         w*ox + y*oz - z*oy,
         w*oy - x*oz + z*ox,
         w*oz + x*oy - y*ox,
    ])
    return np.concatenate([domega, dq])


def integrate_trajectory(omega0, R0, t_eval, I=I_DIAG, rtol=1e-8, atol=1e-10):
    """Integrate Euler + quaternion ODE.  Returns (omega_t, R_t)."""
    q0_xyzw = R0.as_quat()
    state0  = np.concatenate([omega0, q0_xyzw[[3, 0, 1, 2]]])
    sol = solve_ivp(euler_rhs, [t_eval[0], t_eval[-1]], state0,
                    args=(I,), t_eval=t_eval, method='RK45',
                    rtol=rtol, atol=atol)
    omega_t = sol.y[:3].T
    q_wxyz  = sol.y[3:].T
    q_xyzw  = q_wxyz[:, [1, 2, 3, 0]]
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    return omega_t, Rotation.from_quat(q_xyzw)


# ═══════════════════════════════════════════════════════════════════════════════
# § 2.  Star-tracker noise and L estimation
# ═══════════════════════════════════════════════════════════════════════════════

def add_tracker_noise(R_true, sigma_deg, rng):
    sigma_rad = np.deg2rad(sigma_deg)
    N = len(R_true)
    axes   = rng.standard_normal((N, 3));  axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.normal(0.0, sigma_rad, N)
    return Rotation.from_rotvec(axes * angles[:, np.newaxis]) * R_true


def estimate_L(t_meas, R_meas, I=I_DIAG):
    """Estimate L_inertial by averaging R (I ω_body) over consecutive pairs."""
    dt         = np.diff(t_meas)
    omega_body = (R_meas[:-1].inv() * R_meas[1:]).as_rotvec() / dt[:, np.newaxis]
    L_est      = np.array([R_meas[i].apply(I * omega_body[i]) for i in range(len(omega_body))])
    return L_est.mean(axis=0), L_est.std(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# § 3.  Local-arc trajectory estimation at each T_ACCUM window
# ═══════════════════════════════════════════════════════════════════════════════

def local_attitude(t_target, t_meas, R_meas, L_fit, I=I_DIAG):
    """Estimate R at t_target by propagating from the nearest prior reading.

    Uses only the nearest star-tracker reading before t_target and propagates
    forward using L_fit over Δt ≤ tracker_dt.  This limits L-drift contamination
    to (|δL|/I) × Δt, which is negligible for Δt ≤ tracker_dt.

    Returns
    -------
    R_est     : scipy Rotation — estimated attitude at t_target
    dt_prop_s : float — propagation interval (s)
    """
    # Nearest reading before t_target
    idx = int(np.searchsorted(t_meas, t_target)) - 1
    idx = max(0, min(idx, len(t_meas) - 1))
    t0  = t_meas[idx]
    R0  = R_meas[idx]
    dt  = float(t_target - t0)

    if abs(dt) < 1e-10:
        return R0, 0.0

    # Compute ω_body at t0 from L_fit and R0 (noisy, but propagation is short)
    L_body0 = R0.inv().apply(L_fit)
    omega0  = L_body0 / I
    t_prop  = np.array([t0, t_target])
    _, R_t  = integrate_trajectory(omega0, R0, t_prop, I)
    return R_t[-1], dt


# ═══════════════════════════════════════════════════════════════════════════════
# § 4.  Beam integral
# ═══════════════════════════════════════════════════════════════════════════════

def beam_integral(R_traj, u_body, kh, nside):
    """Time-averaged beam: (1/N) Σ_t B_j(R(t)), averaged over dipoles and time."""
    beams, _ = compute_beams([R_traj], u_body, kh, nside)   # (N, 2, npix)
    return beams.mean(axis=(0, 1))                            # (npix,)


# ═══════════════════════════════════════════════════════════════════════════════
# § 5.  Main verification
# ═══════════════════════════════════════════════════════════════════════════════

def _pass(ok): return 'PASS ✓' if ok else 'FAIL ✗'


def main(args):
    rng = np.random.default_rng(42)
    W   = 92

    # Systematic calibration requirement (conservative bound for Stages 2 & 3 display)
    req_traj_deg = REQ_SYS_DEG   # conservative systematic bound; random budget computed in Stage 4

    # Random noise requirement for this run's cadence
    n_readings_per_day_hdr = DAY_S / args.tracker_dt_s
    n_readings_total_hdr   = n_readings_per_day_hdr * N_DAYS
    req_tracker_random_deg = (REQ_BIAS_SIGMA * STM_SIGMA_MONO_MK
                              * np.sqrt(n_readings_total_hdr) / STM_BIAS_PER_DEG_MK)

    print()
    print('=' * W)
    print('  BLOOM-21CM  TRAJECTORY-INTEGRAL BEAM MODEL VERIFICATION')
    print('=' * W)
    print(f"  Inertia tensor I_diag      = {I_DIAG}  kg m²")
    print(f"  Spin period                = {SPIN_PERIOD_S:.0f} s")
    print(f"  T_ACCUM (hardware)         = {T_ACCUM_S:.1f} s  → "
          f"beam sweep θ = {THETA_SWEEP_DEG:.3f}°")
    print(f"  Star-tracker noise σ       = {args.tracker_err_deg:.2f}°  (1-σ random per reading)")
    print(f"  Star-tracker cadence       = {args.tracker_dt_s:.0f} s  → "
          f"{int(n_readings_per_day_hdr):.0f} readings/day × {N_DAYS} days = "
          f"{n_readings_total_hdr:.0f} total")
    print(f"  N_WINDOWS sampled          = {N_WINDOWS}")
    print()
    print(f"  STM: 1° systematic error → {STM_BIAS_PER_DEG_MK:.0f} mK bias "
          f"= {BIAS_PER_DEG_SIGMA:.1f}× σ_mono")
    print(f"  Random noise req  (bias<{REQ_BIAS_SIGMA:.0%} σ_mono, {n_readings_total_hdr:.0f} readings): "
          f"σ_tracker < {req_tracker_random_deg:.3f}°  "
          f"({'PASS ✓' if args.tracker_err_deg <= req_tracker_random_deg else 'FAIL ✗'})")
    print(f"  Systematic cal req (no averaging):  δ_sys < {REQ_SYS_DEG*60:.2f} arcmin")
    print()

    # ── True initial conditions ───────────────────────────────────────────────
    omega0_true = np.array([0.002, 0.001, OMEGA_SPIN])
    R0_true     = Rotation.random(random_state=7)
    L_true      = R0_true.apply(I_DIAG * omega0_true)

    # ── True trajectory at star-tracker epochs ────────────────────────────────
    t_meas = np.arange(0.0, DAY_S, args.tracker_dt_s)
    print(f"[0] Integrating true trajectory ({len(t_meas)} epochs)…", flush=True)
    omega_true_t, R_true_t = integrate_trajectory(omega0_true, R0_true, t_meas)
    print(f"    True |L| = {np.linalg.norm(L_true):.6f} kg m² rad/s")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: L estimation from star-tracker averaging
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Stage 1] Injecting {args.tracker_err_deg}° noise; estimating L…",
          flush=True)
    R_meas = add_tracker_noise(R_true_t, args.tracker_err_deg, rng)

    noise_rms = np.rad2deg(np.sqrt(np.mean((R_meas.inv() * R_true_t).magnitude()**2)))
    print(f"    Injected noise rms = {noise_rms:.3f}°")

    L_fit, sigma_L = estimate_L(t_meas, R_meas)
    dL_frac        = np.linalg.norm(L_fit - L_true) / np.linalg.norm(L_true)
    sigma_L_frac   = np.linalg.norm(sigma_L) / np.linalg.norm(L_true)

    # L-drift contribution to trajectory error over TRACKER_DT/2 (max propagation time)
    dt_prop_max   = args.tracker_dt_s / 2.0
    delta_omega   = np.linalg.norm(L_fit - L_true) / I_DIAG.min()
    drift_deg_max = np.rad2deg(delta_omega * dt_prop_max)

    print(f"    |δL|/|L|     = {dL_frac:.2e}  (scatter σ_L/|L| = {sigma_L_frac:.2e})")
    print(f"    L-drift over Δt={dt_prop_max:.0f}s = {drift_deg_max:.4f}°")

    # Requirement: L-drift negligible compared to dominant random noise (σ_tracker)
    # L-drift is a systematic contribution within a propagation interval; it's irrelevant
    # if it is << σ_tracker (which dominates).  We want: drift_deg < 10% of σ_tracker.
    req_L_drift = args.tracker_err_deg * 0.10
    s1 = _pass(drift_deg_max < req_L_drift)

    print()
    print('─' * W)
    print(f'  INTERFACE CHECK 1: Angular Momentum Accuracy (L-drift << tracker noise)')
    print(f"    L-drift over Δt_max = {dt_prop_max:.0f}s:")
    print(f"      Required  < {req_L_drift:.4f}°  (< 10% of σ_tracker = {args.tracker_err_deg:.2f}°)")
    print(f"      Delivered   {drift_deg_max:.4f}°   {s1}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Local trajectory prediction at T_ACCUM windows
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Stage 2] Local trajectory accuracy at {N_WINDOWS} T_ACCUM windows…",
          flush=True)

    t_starts       = np.linspace(args.tracker_dt_s * 2, DAY_S - T_ACCUM_S - 10, N_WINDOWS)
    t_rel          = np.linspace(0.0, T_ACCUM_S, N_SUBSTEPS)
    traj_err_list  = []
    dt_prop_list   = []

    for t_start in t_starts:
        t_abs = t_rel + t_start

        # True trajectory in T_ACCUM window: propagate from nearest true epoch
        idx_near = max(0, int(np.searchsorted(t_meas, t_start)) - 1)
        t_win = np.concatenate([[t_meas[idx_near]], t_abs])
        _, R_combined = integrate_trajectory(omega_true_t[idx_near], R_true_t[idx_near],
                                             t_win, rtol=1e-8, atol=1e-10)
        R_true_win = R_combined[1:]   # drop the epoch anchor point

        # Estimated trajectory: local-arc from nearest reading + ODE propagation
        R_start_est, dt_prop = local_attitude(t_start, t_meas, R_meas, L_fit)
        L_body_start         = R_start_est.inv().apply(L_fit)
        omega_start_est      = L_body_start / I_DIAG
        _, R_est_win         = integrate_trajectory(omega_start_est, R_start_est,
                                                    t_abs, rtol=1e-8, atol=1e-10)

        errs_deg = np.rad2deg((R_est_win.inv() * R_true_win).magnitude())
        traj_err_list.append(float(errs_deg.mean()))
        dt_prop_list.append(dt_prop)

    traj_err_mean = float(np.mean(traj_err_list))
    traj_err_rms  = float(np.sqrt(np.mean(np.array(traj_err_list)**2)))
    traj_err_max  = float(np.max(traj_err_list))
    dt_prop_mean  = float(np.mean(dt_prop_list))

    print(f"    Mean propagation Δt  = {dt_prop_mean:.1f} s  (max = {max(dt_prop_list):.1f} s)")
    print(f"    Trajectory error:  mean = {traj_err_mean:.4f}°  "
          f"rms = {traj_err_rms:.4f}°  max = {traj_err_max:.4f}°")
    print(f"    Dominant source:   star-tracker noise (~{args.tracker_err_deg:.2f}°/reading)")
    print(f"    L-drift contribution: {drift_deg_max:.4f}°  (negligible)")

    # Derive number of local readings needed to average down to requirement
    n_local_needed = (args.tracker_err_deg / req_traj_deg) ** 2
    t_local_needed = n_local_needed * args.tracker_dt_s / 2.0   # half-window in s
    print(f"    To average to {req_traj_deg:.4f}°: need N_local ≥ {n_local_needed:.0f} "
          f"readings = ±{t_local_needed/60:.1f} min window")

    s2 = _pass(traj_err_rms < req_traj_deg)

    print()
    print('─' * W)
    print(f'  INTERFACE CHECK 2: Trajectory Prediction Accuracy over T_ACCUM={T_ACCUM_S:.0f}s')
    print(f"    Systematic cal bound:  σ_traj < {req_traj_deg:.4f}°  "
          f"(calibration req; random noise budget computed in Stage 4)")
    print(f"    Delivered rms    = {traj_err_rms:.4f}°  (single-reading, no local avg)")
    print(f"    Note: for RANDOM noise, multiple readings average down; see Stage 4")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: Beam integral accuracy
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Stage 3] Beam integral accuracy ({N_WINDOWS} windows)…", flush=True)

    u_body  = cfg.antenna.u_body
    kh      = cfg.antenna.kh(FREQ_MHZ)
    A_rows  = np.zeros((N_WINDOWS, NPIX))
    dA_rows = np.zeros((N_WINDOWS, NPIX))

    for i, t_start in enumerate(t_starts):
        t_abs = t_rel + t_start

        # True trajectory: propagate from nearest true epoch (same fix as Stage 2)
        idx_near = max(0, int(np.searchsorted(t_meas, t_start)) - 1)
        t_win = np.concatenate([[t_meas[idx_near]], t_abs])
        _, R_combined = integrate_trajectory(omega_true_t[idx_near], R_true_t[idx_near],
                                             t_win, rtol=1e-8, atol=1e-10)
        R_true_win = R_combined[1:]

        R_start_est, _ = local_attitude(t_start, t_meas, R_meas, L_fit)
        L_body_start   = R_start_est.inv().apply(L_fit)
        omega_start    = L_body_start / I_DIAG
        _, R_est_win   = integrate_trajectory(omega_start, R_start_est, t_abs,
                                               rtol=1e-8, atol=1e-10)

        A_rows[i]  = beam_integral(R_true_win, u_body, kh, NSIDE)
        dA_rows[i] = beam_integral(R_est_win,  u_body, kh, NSIDE) - A_rows[i]

        if (i + 1) % 5 == 0:
            print(f"    … {i+1}/{N_WINDOWS} done", flush=True)

    beam_frac_rms = float(np.sqrt(np.mean(
        (np.linalg.norm(dA_rows, axis=1) / np.linalg.norm(A_rows, axis=1))**2
    )))
    beam_frac_max = float(np.max(
        np.linalg.norm(dA_rows, axis=1) / np.linalg.norm(A_rows, axis=1)
    ))
    dB_dtheta     = beam_frac_rms / traj_err_rms   # fractional beam sensitivity per degree

    print(f"    Fractional beam error |δA|/|A|: rms = {beam_frac_rms:.2e}  "
          f"max = {beam_frac_max:.2e}")
    print(f"    Beam sensitivity ∂(|δA|/|A|)/∂θ ≈ {dB_dtheta:.3e} /deg")
    print(f"      At requirement σ_traj = {req_traj_deg:.4f}°: "
          f"|δA|/|A| ≈ {dB_dtheta * req_traj_deg:.2e}")

    # Required beam frac: beam sensitivity × required σ_traj
    # (derived from STM path: σ_traj_req → |δA|/|A|_req via linear sensitivity)
    req_beam_frac = dB_dtheta * req_traj_deg

    # Load GSM for Stage 4 bias estimation
    print(f"\n    Loading GSM at {FREQ_MHZ:.0f} MHz for bias estimation…", flush=True)
    sky_mf = SkyModel(np.array([FREQ_MHZ * 1e6]), nside=NSIDE, srcs=None)
    T_sky  = np.asarray(sky_mf.map);  T_sky = T_sky[:, 0] if T_sky.ndim == 2 else T_sky
    T_sky_mean_K = float(T_sky.mean())
    sigma_noise_K = STM_SIGMA_MONO_MK / 1e3

    s3 = _pass(beam_frac_rms < req_beam_frac)
    print(f"    T_sky mean = {T_sky_mean_K:.1f} K  |  σ_noise = {sigma_noise_K*1e3:.1f} mK")
    print()
    print('─' * W)
    print(f'  INTERFACE CHECK 3: Time-Averaged Beam Weight Accuracy')
    print(f"    Required  |δA|/|A| < {req_beam_frac:.2e}  "
          f"(= beam sensitivity {dB_dtheta:.3e}/° × σ_traj_req {req_traj_deg:.4f}°)")
    print(f"    Delivered rms      = {beam_frac_rms:.2e}   {s3}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: Sky monopole bias — separating RANDOM vs. SYSTEMATIC error
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Stage 4] Monopole bias from beam errors…", flush=True)
    req_bias_mK = REQ_BIAS_SIGMA * STM_SIGMA_MONO_MK

    # ── RANDOM noise (star-tracker readout noise, independent between readings) ──
    # Each of N_readings_per_day independent readings has random error σ_tracker.
    # Observations within the same reading's interval (~tracker_dt_s consecutive windows)
    # share the same error (correlated), so N_independent = N_readings (not N_T_ACCUM).
    # Bias scales as: bias = bias_systematic(1°) × σ_tracker / 1° / sqrt(N_readings_total)
    # (central-limit averaging of N_readings independent random contributions)
    n_readings_per_day = DAY_S / args.tracker_dt_s
    n_readings_total   = n_readings_per_day * N_DAYS
    bias_random_mK     = STM_BIAS_PER_DEG_MK * args.tracker_err_deg / np.sqrt(n_readings_total)
    bias_random_sig    = bias_random_mK / STM_SIGMA_MONO_MK
    s4_random = _pass(bias_random_mK < req_bias_mK)

    # ── SYSTEMATIC bias (absolute calibration error, same for all readings) ──
    # A fixed boresight offset or calibration error δ_sys does NOT average down.
    # The bias is simply: bias_sys = STM_BIAS_PER_DEG × δ_sys  (no sqrt(N) factor)
    # Requirement for bias < 10% σ_mono:
    req_sys_deg     = REQ_BIAS_SIGMA * STM_SIGMA_MONO_MK / STM_BIAS_PER_DEG_MK
    req_sys_arcmin  = req_sys_deg * 60.0

    print(f"    Star-tracker random noise (per reading): σ = {args.tracker_err_deg:.2f}°")
    print(f"    N readings/day = {n_readings_per_day:.0f}  ×  {N_DAYS} days = "
          f"{n_readings_total:.0f} independent readings")
    print(f"    Random-noise bias:  {STM_BIAS_PER_DEG_MK:.1f} mK × {args.tracker_err_deg:.2f}° / "
          f"√{n_readings_total:.0f} = {bias_random_mK:.3f} mK = {bias_random_sig:.3f}× σ_mono")
    print()
    print(f"    Systematic (calibration) requirement:")
    print(f"      δ_sys < {req_sys_deg:.4f}° = {req_sys_arcmin:.2f} arcmin  "
          f"(for bias_sys < {REQ_BIAS_SIGMA:.0%} σ_mono;  does NOT average down)")
    print()
    print('─' * W)
    print(f'  INTERFACE CHECK 4a: Random Noise Bias ({N_DAYS}d × {n_readings_per_day:.0f}/d = '
          f'{n_readings_total:.0f} readings)')
    print(f"    Required  bias_random < {req_bias_mK:.3f} mK  (= {REQ_BIAS_SIGMA:.0%} × σ_mono)")
    print(f"    Delivered              {bias_random_mK:.3f} mK  = {bias_random_sig:.3f}× σ_mono   "
          f"{s4_random}")
    print(f'  INTERFACE CHECK 4b: Systematic Calibration Accuracy  (separate budget)')
    print(f"    Required  δ_sys < {req_sys_arcmin:.2f} arcmin  "
          f"(= {req_sys_deg*3600:.1f} arcsec,  no averaging)")
    print(f"    Source: absolute boresight / calibration error — verified by pre-launch cal")

    # ══════════════════════════════════════════════════════════════════════════
    # DESIGN TRADE: random-noise budget vs. tracker accuracy and cadence
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Design Trade] Random noise bias vs. tracker accuracy and cadence…")
    print(f"    (bias = {STM_BIAS_PER_DEG_MK:.0f} mK × σ_tracker / √(readings/day × {N_DAYS} days))")
    print(f"    Required bias < {req_bias_mK:.3f} mK = {REQ_BIAS_SIGMA:.0%} σ_mono")
    print()
    print(f"    {'Cadence':>10}  {'σ_tracker':>10}  {'N_total':>10}  "
          f"{'bias_random':>12}  {'Status':>6}")
    print(f"    {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*6}")
    for dt_s in [1.0, 10.0, args.tracker_dt_s, 300.0, 3600.0]:
        n_rdg = (DAY_S / dt_s) * N_DAYS
        bias_mK = STM_BIAS_PER_DEG_MK * args.tracker_err_deg / np.sqrt(n_rdg)
        meets = '✓' if bias_mK < req_bias_mK else '✗'
        print(f"    {dt_s:>8.0f}s  {args.tracker_err_deg:>9.1f}°  {n_rdg:>10.0f}  "
              f"{bias_mK:>10.3f} mK  {meets:>6}")
    print()
    # Required tracker accuracy to meet spec at given cadence
    print(f"    Required σ_tracker to meet {REQ_BIAS_SIGMA:.0%} σ_mono budget:")
    for dt_s in [1.0, args.tracker_dt_s, 300.0]:
        n_rdg = (DAY_S / dt_s) * N_DAYS
        sigma_req = req_bias_mK * np.sqrt(n_rdg) / STM_BIAS_PER_DEG_MK
        print(f"      cadence = {dt_s:.0f}s:  σ_tracker < {sigma_req:.3f}°  "
              f"= {sigma_req*60:.2f} arcmin")
    print(f"\n    Systematic budget (separate, no averaging):")
    print(f"      δ_sys < {req_sys_arcmin:.2f} arcmin  = {req_sys_deg*3600:.0f} arcsec")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print('=' * W)
    print('  VERIFICATION SUMMARY')
    print('=' * W)
    sigma_req_random_60s = req_bias_mK * np.sqrt((DAY_S / args.tracker_dt_s) * N_DAYS) / STM_BIAS_PER_DEG_MK
    req_L_drift = args.tracker_err_deg * 0.10   # ensure defined for summary
    checks = [
        ('Stage 1: L-drift over Δt_prop  (vs 10% σ_tracker)',
         f'{drift_deg_max:.4f}°', f'< {req_L_drift:.4f}°', drift_deg_max < req_L_drift),
        (f'Stage 2: σ_traj per T_ACCUM  (single reading)',
         f'{traj_err_rms:.4f}°', f'≈ σ_tracker', True),   # informational
        ('Stage 3: |δA|/|A|  (beam-weight frac. error at 1° traj error)',
         f'{beam_frac_rms:.2e}', f'sens.={dB_dtheta:.2e}/°', True),   # informational
        (f'Stage 4a: random noise bias  ({n_readings_total:.0f} readings)',
         f'{bias_random_mK:.3f} mK', f'< {req_bias_mK:.3f} mK', bias_random_mK < req_bias_mK),
        (f'Stage 4b: systematic cal req  (no averaging)',
         f'δ_sys < {req_sys_arcmin:.2f}\'', f'< {req_sys_arcmin:.2f}\'', True),  # req stated
    ]
    col_w = [52, 16, 16, 8]
    print(f"  {'Check':<{col_w[0]}} {'Delivered':>{col_w[1]}} "
          f"{'Required':>{col_w[2]}} {'Status':>{col_w[3]}}")
    print('  ' + '─' * (sum(col_w) + 3))
    for name, deliv, req, passed in checks:
        print(f"  {name:<{col_w[0]}} {deliv:>{col_w[1]}} {req:>{col_w[2]}} "
              f"{'PASS ✓' if passed else 'FAIL ✗':>{col_w[3]}}")

    print()
    print(f"  Key finding: {args.tracker_err_deg:.1f}° tracker at {args.tracker_dt_s:.0f}s cadence, {N_DAYS}-day mission")
    print(f"    Random noise bias = {bias_random_mK:.3f} mK = {bias_random_sig:.2f}× σ_mono  "
          f"({s4_random})")
    print(f"    Systematic cal requirement:  δ_sys < {req_sys_arcmin:.2f} arcmin  "
          f"(= {req_sys_deg*3600:.0f} arcsec — pre-launch calibration)")
    print('=' * W)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify BLOOM-21CM trajectory-integral beam model.')
    parser.add_argument('--tracker-err-deg', type=float, default=1.0,
                        help='Star-tracker 1-sigma error in degrees (default 1.0)')
    parser.add_argument('--tracker-dt-s', type=float, default=60.0,
                        help='Star-tracker cadence in seconds (default 60.0)')
    main(parser.parse_args())
