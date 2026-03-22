"""
test_multifreq.py — standalone diagnostic script for BLOOM-21cm multi-frequency analysis.

Extracted from notebooks/Lunar Orbit Multi Freq.ipynb.
Run from the repo root:
    cd /home/aparsons/projects/global21cm/eigsep_sim
    python notebooks/test_multifreq.py

Findings log (append-only):
  2026-03-21  Root cause of chi²/dof >> 1 was the `/ 100` XXX hack on sigma_f.
              Removing it gives chi²/dof ≈ 0.8 with per-frequency sigma weighting.
              Foreground leakage after eigenmode filtering is subdominant (<10% of sigma_res).
              Per-frequency sigma weighting is more principled than mean sigma because
              SIGMA_MONO varies ~10x across the 55–115 MHz band.

  2026-03-21  Added FG-leakage scan (N_EIG_MODES 1–15) and sigma_scale selection.
              T_est_nl is reused for all N_EIG_MODES values in the scan — valid because
              run_multifreq() does not depend on modes (it only returns T_sky_mean_est).
              N_EIG_MODES=4 is the sweet spot: FG_leak/T_21_filt = 13.5% (lowest ratio),
              drops sharply from N=3 (5500%) to N=4 (13.5%) then rises again for N>=5
              (overfit removes signal power too).
              sigma_scale=0.1230 (TARGET_SNR=2) chosen so noise rms ≈ T_21_filt_rms / 2.
              Signal recovery (model 0): chi²/dof=1.43 < threshold 1.57, rank=2/646 (PASS).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from astropy.coordinates import get_body
import astropy.units as u
import healpy

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
)

# ── Configuration ─────────────────────────────────────────────────────────────
_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bloom_config.yaml")
cfg = OrbiterMission(_YAML)
print(repr(cfg))
FP = config_fingerprint(cfg)
print(f"Config fingerprint: {FP}")

# ── Frequency grid (from config) ──────────────────────────────────────────────
FREQS_MHZ = np.linspace(
    cfg.observation.freq_min_mhz,
    cfg.observation.freq_max_mhz,
    cfg.observation.nchan_science,
)
N_FREQ      = len(FREQS_MHZ)
print(f"Frequency grid: {FREQS_MHZ[0]:.1f} – {FREQS_MHZ[-1]:.1f} MHz  ({N_FREQ} channels)")
print(f"delta_nu      = {cfg.observation.delta_nu/1e6:.4f} MHz  ({cfg.observation.channel_width_khz:.1f} kHz)")
print(f"t_integration = {cfg.observation.t_integration:.2f} s  "
      f"(duty={cfg.observation.duty_cycle}, n_days={cfg.observation.n_days})")
print(f"t_snapshot    = {cfg.observation.t_snapshot:.3f} s  "
      f"(attitude limit; ratio {cfg.observation.t_integration/cfg.observation.t_snapshot:.0f}× >> 1 ✓)")
N_EIG_MODES = 5

# ── Spacecraft attitudes ──────────────────────────────────────────────────────
if cfg.observation.fixed_spin:
    phi = np.linspace(0.0, 2.0 * np.pi, cfg.observation.n_obs, endpoint=False)
    rot_fixed = Rotation.from_rotvec(np.outer(phi, cfg.antenna.l_hat))
    rots_per_orbit = [rot_fixed for _ in range(cfg.observation.n_orbits)]
else:
    rots_per_orbit = [
        Rotation.random(cfg.observation.n_obs, random_state=42 + o)
        for o in range(cfg.observation.n_orbits)
    ]

# ── Orbits and time grid ──────────────────────────────────────────────────────
orbits_list = cfg.observation.make_orbits(rot_spin_vec=(0, 0, 1), spin_period=0.0)
t_obs_s     = np.linspace(0.0, cfg.observation.n_days * 86400.0, cfg.observation.n_obs, endpoint=False)
obs_times   = cfg.observation.obs_epoch + t_obs_s * u.s

# ── Sky and signal models ─────────────────────────────────────────────────────
print("Loading GSM …", flush=True)
sky_mf   = SkyModel(FREQS_MHZ * 1e6, nside=cfg.observation.nside, srcs=None)
gsm_maps = np.asarray(sky_mf.map)
if gsm_maps.ndim == 1:
    gsm_maps = gsm_maps[:, np.newaxis]

models_21cm  = T21cmModel()
INJ_MODEL_IDX = 0
T_21_INJ = models_21cm(FREQS_MHZ * 1e6, model_index=INJ_MODEL_IDX)
print(f"Injection model {INJ_MODEL_IDX}: peak = {T_21_INJ.min()*1e3:.1f} mK")

# ── Sun positions ─────────────────────────────────────────────────────────────
print("Querying Sun positions …", flush=True)
sun_coords = get_body("sun", obs_times)
sun_gal    = sun_coords.galactic
l_s, b_s   = sun_gal.l.rad, sun_gal.b.rad
J_SUN = healpy.vec2pix(cfg.observation.nside,
                       np.cos(b_s)*np.cos(l_s),
                       np.cos(b_s)*np.sin(l_s),
                       np.sin(b_s))

# ── Pre-compute frequency-independent masks (cached) ──────────────────────────
_cached_setup = try_load_setup(FP)
if _cached_setup is not None:
    masks_mf, _ = _cached_setup
else:
    print("Computing occultation masks …", flush=True)
    masks_mf, _, _ = compute_masks_and_beams(
        orbits_list, obs_times, rots_per_orbit,
        cfg.antenna.u_body, cfg.antenna.kh(FREQS_MHZ[0]),
        cfg.observation.nside, verbose=False,
    )
    save_setup(masks_mf, J_SUN, FP)

# ── GSM eigenmodes ────────────────────────────────────────────────────────────
modes = gsm_eigenmodes(gsm_maps, N_EIG_MODES)   # includes flat mode
N_MODES_TOTAL = modes.shape[0]
dof = N_FREQ - N_MODES_TOTAL
print(f"Filter modes: {N_EIG_MODES} GSM + {N_MODES_TOTAL - N_EIG_MODES} flat = {N_MODES_TOTAL}, dof={dof}")

T_21_filt  = eigenmode_filter(T_21_INJ, modes)
T_all      = models_21cm(FREQS_MHZ * 1e6)
T_all_filt = eigenmode_filter(T_all, modes)


# ── Core multi-frequency loop ─────────────────────────────────────────────────
def run_multifreq(noise_seed_offset=0, noise_scale=1.0):
    """
    Run the 30-frequency per-pixel inversion.

    T_rx is absorbed into the sky monopole (include_t_rx=False).
    Observable: nanmean(sky_map).
    Noise estimate: sigma * sqrt(e^T (A^T A)^{-1} e) from eigendecomposition.
    """
    T_sky_mean_est = np.empty(N_FREQ)
    SIGMA_MONO     = np.empty(N_FREQ)

    for fi, f_mhz in enumerate(FREQS_MHZ):
        kh_f = cfg.antenna.kh(f_mhz)
        beams_f, omega_B_f = compute_beams(
            rots_per_orbit, cfg.antenna.u_body, kh_f, cfg.observation.nside,
        )
        gsm_f  = gsm_maps[:, fi]
        sky_f  = gsm_f + T_21_INJ[fi]
        sigma_f = cfg.antenna.sigma_noise(
            f_mhz, cfg.observation.delta_nu, cfg.observation.t_integration,
            t_gsm_avg=float(gsm_f.mean()),
        ) * noise_scale

        _, y_f = simulate_observations(
            masks_mf, beams_f, omega_B_f,
            sky_f, cfg.observation.t_regolith, cfg.observation.t_sun,
            J_SUN, sigma_f,
            t_rx=np.full(2, cfg.antenna.t_rx),
            rng=np.random.default_rng(fi + noise_seed_offset),
        )

        A_f   = build_design_matrix(masks_mf, beams_f, omega_B_f, J_SUN, cfg.observation.npix,
                                     include_t_rx=False)
        res_f = normal_solve(A_f, y_f, cfg.observation.npix)

        T_sky_mean_est[fi] = float(np.nanmean(res_f['sky_map']))

        n_obs_pix = int((~res_f['unobserved']).sum())
        e_sky = np.zeros(cfg.observation.npix + 2)
        e_sky[:cfg.observation.npix][~res_f['unobserved']] = 1.0 / n_obs_pix
        Ve = res_f['eigenvectors'].T @ e_sky
        SIGMA_MONO[fi] = float(np.mean(sigma_f)) * np.sqrt(
            float(np.dot(Ve ** 2, res_f['inv_eigenvalues']))
        )

    return T_sky_mean_est, SIGMA_MONO


# ── Noiseless run — foreground leakage (cached) ───────────────────────────────
print("\nNoiseless run …", flush=True)
_nl_cached = try_load_multifreq(FREQS_MHZ, 0.0, 0, FP)
if _nl_cached is not None:
    T_est_nl, _ = _nl_cached
else:
    T_est_nl, _sigma_nl = run_multifreq(noise_scale=0.0)
    save_multifreq(FREQS_MHZ, T_est_nl, _sigma_nl, 0.0, 0, FP)
resid_nl     = eigenmode_filter(T_est_nl, modes)
FG_leakage   = resid_nl - T_21_filt
print(f"  FG leakage rms  = {np.std(FG_leakage)*1e3:.4f} mK")
print(f"  T_21_filt rms   = {np.std(T_21_filt)*1e3:.4f} mK")

# ── Noisy run ─────────────────────────────────────────────────────────────────
print("\nNoisy run …", flush=True)
T_est, SIGMA_MONO = run_multifreq()
resid_est  = eigenmode_filter(T_est, modes)
noise_term = eigenmode_filter(T_est - T_est_nl, modes)

sigma_mean = float(np.mean(SIGMA_MONO))
chi2_mean  = np.sum(((resid_est - T_all_filt) / sigma_mean)**2, axis=1) / dof
chi2_wt    = np.sum(((resid_est - T_all_filt) / SIGMA_MONO[np.newaxis, :])**2, axis=1) / dof
chi2_noise = np.sum((noise_term / SIGMA_MONO)**2) / dof   # should be ~1

print(f"\n  SIGMA_MONO range  : {SIGMA_MONO.min()*1e3:.2f} – {SIGMA_MONO.max()*1e3:.2f} mK")
print(f"  sigma_res (mean)  : {sigma_mean*1e3:.4f} mK")
print(f"  resid rms         : {np.std(resid_est)*1e3:.4f} mK")
print(f"  noise term rms    : {np.std(noise_term)*1e3:.4f} mK")
print(f"  FG leakage rms    : {np.std(FG_leakage)*1e3:.4f} mK  "
      f"({np.std(FG_leakage)/sigma_mean:.2f}× sigma_mean)")
print(f"\n  chi²/dof (mean σ)    : min={chi2_mean.min():.3f}  median={np.median(chi2_mean):.3f}")
print(f"  chi²/dof (per-freq σ): min={chi2_wt.min():.3f}  median={np.median(chi2_wt):.3f}")
print(f"  chi²/dof (noise only): {chi2_noise:.4f}  ← expected ~1")
print(f"  Best model (per-freq σ): {np.argmin(chi2_wt)}  (injected: {INJ_MODEL_IDX})")

if 0.3 < chi2_noise < 3.0:
    print(f"\n  PASS: noise-only chi²/dof = {chi2_noise:.3f}")
else:
    print(f"\n  FAIL: noise-only chi²/dof = {chi2_noise:.3f} is outside [0.3, 3.0]")

# ── Per-frequency and combined SNR ────────────────────────────────────────────
SNR          = T_21_filt / SIGMA_MONO        # signed per-frequency SNR (N_FREQ,)
SNR_combined = float(np.sqrt(np.sum(SNR**2)))
print(f"\n  {'Freq [MHz]':>10}  {'T_21_filt [mK]':>15}  {'σ [mK]':>8}  {'SNR':>7}")
print("  " + "-"*46)
for f, s21, sig, snr in zip(FREQS_MHZ, T_21_filt*1e3, SIGMA_MONO*1e3, SNR):
    print(f"  {f:10.1f}  {s21:15.3f}  {sig:8.3f}  {snr:7.3f}")
print(f"\n  Combined SNR = sqrt(Σ SNR²) = {SNR_combined:.2f}")

# ── Diagnostic plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.fill_between(FREQS_MHZ, -SIGMA_MONO*1e3, SIGMA_MONO*1e3,
                alpha=0.2, color='C0', label='±1σ per-freq')
ax.axhspan(-sigma_mean*1e3, sigma_mean*1e3, alpha=0.1, color='gray', label='±1σ mean')
ax.plot(FREQS_MHZ, resid_est*1e3, 'k.-', ms=5, label='resid_est')
ax.plot(FREQS_MHZ, T_21_filt*1e3, 'r--', lw=2, label='T_21_filt (injected, filtered)')
ax.plot(FREQS_MHZ, FG_leakage*1e3, 'g-', alpha=0.7, label=f'FG leakage')
ax.set_xlabel('Frequency [MHz]')
ax.set_ylabel('ΔT [mK]')
ax.set_title(f'Residual  (chi²/dof min={chi2_wt.min():.2f} per-freq σ)')
ax.legend(fontsize=8)

ax = axes[1]
ax.semilogy(FREQS_MHZ, SIGMA_MONO*1e3, 'b-', label='SIGMA_MONO')
ax.axhline(sigma_mean*1e3, color='k', ls='--', label=f'mean = {sigma_mean*1e3:.2f} mK')
ax.set_xlabel('Frequency [MHz]')
ax.set_ylabel('σ [mK]')
ax.set_title('Per-frequency noise estimate')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/tmp/test_multifreq.png', dpi=100)
plt.close()
print("\nSaved /tmp/test_multifreq.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: FG leakage scan — vary N_EIG_MODES from 1 to 15
# ═══════════════════════════════════════════════════════════════════════════════
# T_est_nl is reused for all scanned values: run_multifreq() does not use
# `modes` — it only returns nanmean(sky_map), which is independent of N_EIG_MODES.
print("\n" + "="*70)
print("STEP 1: FG leakage scan — N_EIG_MODES from 1 to 15")
print("="*70)

FG_TARGET_FRAC = 0.10   # target: FG leakage rms < 10% of T_21_filt_rms

scan_results = []   # list of (n, fg_rms_mK, sig21_rms_mK, fg_frac)
for n in range(1, 16):
    modes_n       = gsm_eigenmodes(gsm_maps, n)           # (n+1, N_FREQ) with flat
    T_21_filt_n   = eigenmode_filter(T_21_INJ, modes_n)  # injected signal after filter
    resid_nl_n    = eigenmode_filter(T_est_nl, modes_n)  # noiseless residual
    fg_leak_n     = resid_nl_n - T_21_filt_n             # FG leakage
    fg_rms        = float(np.std(fg_leak_n))
    sig21_rms     = float(np.std(T_21_filt_n))
    fg_frac       = fg_rms / sig21_rms if sig21_rms > 0 else np.inf
    scan_results.append((n, fg_rms * 1e3, sig21_rms * 1e3, fg_frac))
    print(f"  N_EIG_MODES={n:2d}:  FG_leak={fg_rms*1e3:.4f} mK  "
          f"T_21_filt={sig21_rms*1e3:.4f} mK  ratio={fg_frac:.4f}")

# Find minimum N_EIG_MODES where FG leakage < FG_TARGET_FRAC * T_21_filt_rms
optimal_n = None
for n, fg_rms_mK, sig21_rms_mK, fg_frac in scan_results:
    if fg_frac < FG_TARGET_FRAC:
        optimal_n = n
        break

if optimal_n is None:
    # Use the value with the lowest FG_frac ratio (best suppression relative to signal)
    optimal_n = min(scan_results, key=lambda x: x[3])[0]
    best_frac  = min(r[3] for r in scan_results)
    print(f"\n  NOTE: No N_EIG_MODES achieved FG_frac < {FG_TARGET_FRAC:.0%}; "
          f"using N={optimal_n} (lowest FG_frac = {best_frac:.2%}).")
else:
    print(f"\n  Optimal N_EIG_MODES = {optimal_n}  "
          f"(first to achieve FG_leakage < {FG_TARGET_FRAC:.0%} * T_21_filt_rms)")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Choose sigma_scale with the optimal N_EIG_MODES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"STEP 2: Choose sigma_scale with N_EIG_MODES = {optimal_n}")
print("="*70)

# Rebuild modes and filtered quantities for the optimal N
modes_opt      = gsm_eigenmodes(gsm_maps, optimal_n)
T_21_filt_opt  = eigenmode_filter(T_21_INJ, modes_opt)
T_all_filt_opt = eigenmode_filter(models_21cm(FREQS_MHZ * 1e6), modes_opt)
resid_nl_opt   = eigenmode_filter(T_est_nl, modes_opt)
FG_leakage_opt = resid_nl_opt - T_21_filt_opt
N_MODES_OPT    = modes_opt.shape[0]
dof_opt        = N_FREQ - N_MODES_OPT

T21_filt_rms   = float(np.std(T_21_filt_opt))
fg_leak_rms    = float(np.std(FG_leakage_opt))

print(f"  T_21_filt rms = {T21_filt_rms*1e3:.4f} mK")
print(f"  FG leakage rms = {fg_leak_rms*1e3:.4f} mK  "
      f"(ratio = {fg_leak_rms/T21_filt_rms:.4f})")

# noise_term rms scales linearly with noise_scale.
# From the baseline noisy run (noise_scale=1.0), compute the noise rms at scale=1:
noise_term_baseline = eigenmode_filter(T_est - T_est_nl, modes_opt)
noise_rms_at_scale1 = float(np.std(noise_term_baseline))
print(f"  Noise rms at noise_scale=1: {noise_rms_at_scale1*1e3:.4f} mK")

# Target: noise rms ≈ T_21_filt_rms / 2  (signal clearly detectable at ~2 sigma,
# chi²/dof calibration is better with a slightly larger noise_scale than /3).
TARGET_SNR   = 2.0
target_noise = T21_filt_rms / TARGET_SNR
sigma_scale  = target_noise / noise_rms_at_scale1
print(f"  Target noise rms = T_21_filt_rms / {TARGET_SNR:.0f} = {target_noise*1e3:.4f} mK")
print(f"  => sigma_scale = {sigma_scale:.4f}")

# Sanity: ensure noise still dominates FG leakage
noise_rms_scaled = noise_rms_at_scale1 * sigma_scale
if noise_rms_scaled > fg_leak_rms:
    print(f"  Noise ({noise_rms_scaled*1e3:.4f} mK) > FG leakage ({fg_leak_rms*1e3:.4f} mK) ✓")
else:
    print(f"  WARNING: Noise ({noise_rms_scaled*1e3:.4f} mK) < FG leakage "
          f"({fg_leak_rms*1e3:.4f} mK) — sigma_scale may be too small")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Verify signal recovery with optimal N_EIG_MODES and sigma_scale
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"STEP 3: Signal recovery  N_EIG_MODES={optimal_n}  sigma_scale={sigma_scale:.4f}")
print("="*70)

T_est_s, SIGMA_MONO_s = run_multifreq(noise_scale=sigma_scale, noise_seed_offset=100)
resid_est_s  = eigenmode_filter(T_est_s, modes_opt)
noise_term_s = eigenmode_filter(T_est_s - T_est_nl, modes_opt)

# SIGMA_MONO_s is already computed with sigma_f * sigma_scale inside run_multifreq,
# so it is the correct per-frequency noise estimate — no further scaling needed.
SIGMA_s      = SIGMA_MONO_s   # per-frequency noise, already at sigma_scale

chi2_wt_s    = np.sum(((resid_est_s - T_all_filt_opt) / SIGMA_s[np.newaxis, :])**2,
                       axis=1) / dof_opt
chi2_noise_s = np.sum((noise_term_s / SIGMA_s)**2) / dof_opt

chi2_model0  = chi2_wt_s[INJ_MODEL_IDX]
rank_model0  = int(np.sum(chi2_wt_s <= chi2_model0))   # 1-based rank
n_models     = chi2_wt_s.shape[0]
pct_rank     = rank_model0 / n_models * 100.0

# Signal recovery criterion: chi²_model0/dof < 1 + 2*sqrt(2/dof)
chi2_thresh  = 1.0 + 2.0 * np.sqrt(2.0 / dof_opt)

print(f"\n  dof                         = {dof_opt}")
print(f"  SIGMA_MONO range            = {SIGMA_s.min()*1e3:.2f} – {SIGMA_s.max()*1e3:.2f} mK")
print(f"  noise rms in resid_est      = {np.std(noise_term_s)*1e3:.4f} mK")
print(f"  FG leakage rms              = {fg_leak_rms*1e3:.4f} mK")
print(f"  T_21_filt rms               = {T21_filt_rms*1e3:.4f} mK")
print(f"  chi²/dof (noise only)       = {chi2_noise_s:.4f}  (expected ~1)")
print(f"  chi²/dof model 0 (injected) = {chi2_model0:.4f}  (threshold = {chi2_thresh:.4f})")
print(f"  chi²/dof best model         = {chi2_wt_s.min():.4f}  (model {np.argmin(chi2_wt_s)})")
print(f"  Rank of model 0             = {rank_model0} / {n_models}  ({pct_rank:.1f}th percentile)")

# PASS/FAIL checks
pass_noise = 0.3 < chi2_noise_s < 3.0
pass_chi2  = chi2_model0 < chi2_thresh
pass_rank  = pct_rank <= 20.0

print()
print(f"  {'PASS' if pass_noise else 'FAIL'}: noise chi²/dof = {chi2_noise_s:.3f}  [expected 0.3–3.0]")
print(f"  {'PASS' if pass_chi2  else 'FAIL'}: model-0 chi²/dof = {chi2_model0:.3f} < threshold {chi2_thresh:.3f}")
print(f"  {'PASS' if pass_rank  else 'FAIL'}: model-0 rank = {rank_model0}/{n_models} "
      f"({pct_rank:.1f}th pct, need ≤20th)")

overall = pass_noise and pass_chi2 and pass_rank
print(f"\n  OVERALL: {'PASS — signal recovery confirmed' if overall else 'FAIL — signal NOT recovered'}")

# ── Per-frequency and combined SNR (scaled run) ───────────────────────────────
SNR_s          = T_21_filt_opt / SIGMA_s
SNR_combined_s = float(np.sqrt(np.sum(SNR_s**2)))
print(f"\n  SNR table  [sigma_scale={sigma_scale:.4f}]:")
print(f"  {'Freq [MHz]':>10}  {'T_21_filt [mK]':>15}  {'σ [mK]':>8}  {'SNR':>7}")
print("  " + "-"*46)
for f, s21, sig, snr in zip(FREQS_MHZ, T_21_filt_opt*1e3, SIGMA_s*1e3, SNR_s):
    print(f"  {f:10.1f}  {s21:15.3f}  {sig:8.3f}  {snr:7.3f}")
print(f"\n  Combined SNR = sqrt(Σ SNR²) = {SNR_combined_s:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 summary — values to plug into the notebook
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: Notebook update summary")
print("="*70)
print(f"  Set N_EIG_MODES = {optimal_n}  in cells 12 and 13 of Lunar Orbit Multi Freq.ipynb")
print(f"  sigma_scale     = {sigma_scale:.4f}  (document as a tuning parameter)")
print(f"  FG leakage rms  = {fg_leak_rms*1e3:.4f} mK  "
      f"({fg_leak_rms/T21_filt_rms:.2%} of T_21_filt_rms)")
print(f"  noise rms       = {np.std(noise_term_s)*1e3:.4f} mK  "
      f"({np.std(noise_term_s)/T21_filt_rms:.2f}× T_21_filt_rms)")

# ── Extended diagnostic plot ───────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes2[0]
ns_vals  = [r[0] for r in scan_results]
fg_vals  = [r[1] for r in scan_results]
s21_vals = [r[2] for r in scan_results]
ax.semilogy(ns_vals, fg_vals, 'r.-', ms=8, label='FG leakage rms')
ax.semilogy(ns_vals, s21_vals, 'b--', lw=1.5, label='T_21_filt rms')
thresh_line = [s * FG_TARGET_FRAC for s in s21_vals]
ax.semilogy(ns_vals, thresh_line, 'k:', lw=1, label=f'{FG_TARGET_FRAC:.0%}×T_21_filt target')
if optimal_n is not None:
    ax.axvline(optimal_n, color='g', ls='--', lw=2, label=f'N_opt={optimal_n}')
ax.set_xlabel('N_EIG_MODES')
ax.set_ylabel('rms [mK]')
ax.set_title('FG leakage vs N_EIG_MODES')
ax.legend(fontsize=8)

ax = axes2[1]
ax.fill_between(FREQS_MHZ, -SIGMA_s*1e3, SIGMA_s*1e3,
                alpha=0.2, color='C0', label='±1σ per-freq (scaled)')
ax.plot(FREQS_MHZ, resid_est_s*1e3, 'k.-', ms=5, label='resid_est')
ax.plot(FREQS_MHZ, T_21_filt_opt*1e3, 'r--', lw=2, label='T_21_filt (injected)')
ax.plot(FREQS_MHZ, FG_leakage_opt*1e3, 'g-', alpha=0.7, label='FG leakage')
ax.set_xlabel('Frequency [MHz]')
ax.set_ylabel('ΔT [mK]')
ax.set_title(f'N_EIG_MODES={optimal_n}  sigma_scale={sigma_scale:.3f}\n'
             f'chi²/dof model0={chi2_model0:.2f}  rank={rank_model0}/{n_models}')
ax.legend(fontsize=8)

ax = axes2[2]
sorted_chi2 = np.sort(chi2_wt_s)
ax.plot(sorted_chi2, np.arange(1, n_models+1) / n_models * 100, 'k-', lw=1.5)
ax.axvline(chi2_model0, color='r', ls='--', lw=2,
           label=f'Model 0  chi²/dof={chi2_model0:.2f}')
ax.axvline(1.0, color='g', ls=':', lw=1.5, label='chi²/dof=1')
ax.set_xlabel('chi²/dof')
ax.set_ylabel('Cumulative % of models')
ax.set_title('CDF of model chi²/dof')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/tmp/test_multifreq_scan.png', dpi=100)
plt.close()
print("\nSaved /tmp/test_multifreq_scan.png")
print("Done.")
