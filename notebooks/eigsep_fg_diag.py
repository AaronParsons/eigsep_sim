"""
eigsep_fg_diag.py — Diagnose EIGSEP FG leakage: chromatic beam vs terrain vs coverage.

Runs the noiseless inversion under four conditions to isolate which effect
dominates the foreground leakage seen after GSM eigenmode filtering:

  1. Full model       — chromatic beam  + actual terrain
  2. Achromatic beam  — midband beam shape at all frequencies  + actual terrain
  3. No terrain       — chromatic beam  + mask=1 everywhere
  4. Ideal            — achromatic beam + mask=1 everywhere

Also reports:
  - Sky pixel coverage (fraction always/never visible)
  - Condition number of A^T A for each variant
  - Beam-chromaticity metric: how much the beam pattern changes across the band

Usage:
    cd /home/aparsons/projects/global21cm/eigsep_sim
    python notebooks/eigsep_fg_diag.py
"""

import os
import sys
import time
import numpy as np
import healpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from astropy.time import Time
import astropy.units as u

from eigsep_sim.earth_surface import EarthSurface
from eigsep_sim.beam import Beam
from eigsep_sim.sim_jax import Terrain
from eigsep_sim.sky import SkyModel
from eigsep_sim.models import T21cmModel
from eigsep_sim.linear_solver import normal_solve
from eigsep_sim.spectral import gsm_eigenmodes, eigenmode_filter

# ── Parameters (same as production notebook) ──────────────────────────────────
NSIDE      = 8
NPIX       = healpy.nside2npix(NSIDE)
LAT_DEG, LON_DEG = 39.2, -113.4
FREQS_MHZ  = np.linspace(55.0, 150.0, 30)
N_FREQ     = len(FREQS_MHZ)
FREQS_HZ   = FREQS_MHZ * 1e6
OBS_EPOCH  = Time("2025-01-01")
N_DAYS     = 7
N_TIMES_PER_DAY = 12
N_TIMES    = N_DAYS * N_TIMES_PER_DAY          # 84
N_AZ       = 8
N_ALT      = 4
N_ORIENT   = N_AZ * N_ALT                      # 32
N_ROWS     = N_TIMES * N_ORIENT                # 2688
T_GND_K    = 300.0
T_RX_K     = 500.0
N_EIG_MODES = 4

_CACHE_DIR   = os.path.join(os.path.dirname(__file__), ".eigsep_cache")
_MASK_FILE   = os.path.join(_CACHE_DIR, f"masks_nside{NSIDE}_t{N_TIMES}_az{N_AZ}_alt{N_ALT}.npy")
_BVALS_FILE  = os.path.join(_CACHE_DIR,
    f"bvals_nside{NSIDE}_t{N_TIMES}_az{N_AZ}_alt{N_ALT}_f{N_FREQ}.npy")

# ── Precompute (or load from cache) ───────────────────────────────────────────
os.makedirs(_CACHE_DIR, exist_ok=True)

obs     = EarthSurface(lat=LAT_DEG, lon=LON_DEG)
beam    = Beam(FREQS_HZ)
terrain = Terrain(FREQS_HZ)

AZ_DEG  = np.linspace(0, 360, N_AZ, endpoint=False)
ALT_DEG = np.linspace(15, 75, N_ALT)
az_grid, alt_grid = np.meshgrid(np.deg2rad(AZ_DEG), np.deg2rad(ALT_DEG))
azs  = az_grid.ravel()
alts = alt_grid.ravel()

crds_gal = np.array(healpy.pix2vec(NSIDE, np.arange(NPIX)))   # (3, NPIX)
times    = OBS_EPOCH + np.linspace(0, N_DAYS * 86400, N_TIMES, endpoint=False) * u.s

if os.path.exists(_MASK_FILE) and os.path.exists(_BVALS_FILE):
    print("Loading precomputed masks and beam values from cache …")
    masks_all = np.load(_MASK_FILE)
    bvals_all = np.load(_BVALS_FILE)
else:
    print("Computing terrain masks …", flush=True)
    crds_top_all = np.zeros((N_TIMES, 3, NPIX), dtype=np.float64)
    masks_all    = np.zeros((N_TIMES, NPIX), dtype=np.float32)
    t0 = time.time()
    for ti, t in enumerate(times):
        obs.set_time(t)
        R = obs.rot_gal2top()
        crds_top_all[ti] = R @ crds_gal
        masks_all[ti]    = terrain.get_mask(crds_top_all[ti])
    print(f"  done in {time.time()-t0:.1f} s  mean open fraction = {masks_all.mean():.3f}")

    print("Computing beam values …", flush=True)
    bvals_all = np.zeros((N_TIMES, N_ORIENT, NPIX, N_FREQ), dtype=np.float32)
    t0 = time.time()
    for oi in range(N_ORIENT):
        beam.set_az(azs[oi])
        beam.set_alt(alts[oi])
        for ti in range(N_TIMES):
            bvals_all[ti, oi] = np.array(beam[crds_top_all[ti]])
        if (oi + 1) % 8 == 0:
            print(f"  orientation {oi+1}/{N_ORIENT}  ({time.time()-t0:.0f} s)", flush=True)
    print(f"  done in {time.time()-t0:.1f} s")

    np.save(_MASK_FILE, masks_all)
    np.save(_BVALS_FILE, bvals_all)

print(f"Precomputation done: masks {masks_all.shape}, bvals {bvals_all.shape}")

# ── Sky models ─────────────────────────────────────────────────────────────────
print("Loading GSM …", flush=True)
sky_model = SkyModel(FREQS_HZ, nside=NSIDE, srcs=None)
gsm_maps  = np.asarray(sky_model.map)                    # (NPIX, N_FREQ)
T_21_INJ  = T21cmModel()(FREQS_HZ, model_index=0)        # (N_FREQ,)

modes      = gsm_eigenmodes(gsm_maps, N_EIG_MODES)
T_21_filt  = eigenmode_filter(T_21_INJ, modes)
T21_rms    = float(np.std(T_21_filt))
dof        = N_FREQ - modes.shape[0]

print(f"GSM eigenmodes: {modes.shape[0]} total ({N_EIG_MODES} GSM + 1 flat)  dof={dof}")
print(f"T_21_filt rms = {T21_rms*1e3:.4f} mK")

# ── Build design matrix ────────────────────────────────────────────────────────

def build_A(bv_fi, masks_used):
    """
    bv_fi : (N_TIMES, N_ORIENT, NPIX) beam values at one frequency
    masks_used : (N_TIMES, NPIX) mask (1=sky, 0=terrain)
    Returns A of shape (N_ROWS, NPIX+2) with no T_rx column.
    """
    bv = bv_fi.reshape(N_ROWS, NPIX).astype(np.float64)
    m  = np.repeat(masks_used, N_ORIENT, axis=0).astype(np.float64)

    bv_sum = np.sum(bv, axis=1, keepdims=True)
    valid  = bv_sum.ravel() > 0

    A = np.zeros((N_ROWS, NPIX + 2), dtype=np.float64)
    A[valid, :NPIX] = bv[valid] * m[valid] / bv_sum[valid]
    A[valid,  NPIX] = np.sum(bv[valid] * (1.0 - m[valid]), axis=1) / bv_sum[valid, 0]
    # col NPIX+1 stays 0 (no T_rx column)
    return A, valid


def run_noiseless(bvals_used, masks_used, label):
    """
    Run noiseless inversion with given beam values and masks.
    bvals_used : (N_TIMES, N_ORIENT, NPIX, N_FREQ)
    masks_used : (N_TIMES, NPIX)
    Returns T_est_nl (N_FREQ,), cond_numbers (N_FREQ,), unobserved_counts (N_FREQ,)
    """
    T_est    = np.empty(N_FREQ)
    conds    = np.empty(N_FREQ)
    n_unobs  = np.zeros(N_FREQ, dtype=int)

    for fi in range(N_FREQ):
        gsm_f  = gsm_maps[:, fi].astype(np.float64)
        sky_f  = gsm_f + T_21_INJ[fi]
        A_f, _ = build_A(bvals_used[:, :, :, fi], masks_used)
        x_true = np.concatenate([sky_f, [T_GND_K], [0.0]])
        y_f    = A_f @ x_true + T_RX_K      # noiseless
        res    = normal_solve(A_f, y_f, NPIX)
        T_est[fi]   = float(np.nanmean(res['sky_map']))
        n_unobs[fi] = int(res['unobserved'].sum())
        lam = res['eigenvalues']
        lam_pos = lam[lam > 0]
        conds[fi] = float(lam_pos.max() / lam_pos.min()) if len(lam_pos) > 1 else np.nan

    resid_nl   = eigenmode_filter(T_est, modes)
    FG_leakage = resid_nl - T_21_filt
    fg_rms     = float(np.std(FG_leakage))
    ratio      = fg_rms / T21_rms
    cond_mean  = float(np.nanmean(conds))
    unobs_mean = float(np.mean(n_unobs))

    print(f"\n  [{label}]")
    print(f"    FG leakage rms  = {fg_rms*1e3:8.4f} mK")
    print(f"    T_21_filt rms   = {T21_rms*1e3:8.4f} mK")
    print(f"    FG / signal     = {ratio:8.4f}  ({ratio*100:.1f}%)")
    print(f"    Mean cond(A^TA) = {cond_mean:.3e}")
    print(f"    Mean unobserved = {unobs_mean:.1f} / {NPIX} pixels")

    return T_est, FG_leakage, ratio, conds


# ── Pixel coverage analysis ────────────────────────────────────────────────────
print("\n" + "="*65)
print("=== EIGSEP FG Leakage Diagnostic ===")
print("="*65)

pix_open_frac = masks_all.mean(axis=0)          # fraction of times each pixel is open
never_visible  = (pix_open_frac == 0.0)
always_visible = (pix_open_frac == 1.0)
mean_open_frac = float(masks_all.mean())

print(f"\nPixel coverage (NSIDE={NSIDE}, NPIX={NPIX}):")
print(f"  Never visible  (always below horizon): {never_visible.sum():3d} / {NPIX}"
      f"  ({never_visible.mean()*100:.1f}%)")
print(f"  Partially visible:                     "
      f"{(~never_visible & ~always_visible).sum():3d} / {NPIX}"
      f"  ({(~never_visible & ~always_visible).mean()*100:.1f}%)")
print(f"  Always visible (never blocked):        {always_visible.sum():3d} / {NPIX}"
      f"  ({always_visible.mean()*100:.1f}%)")
print(f"  Mean open fraction across all times:   {mean_open_frac:.3f}")

# Beam chromaticity: how much does the beam pattern change across the band?
# Metric: RMS variation of beam(p, fi) / beam(p, fi_mid) across frequencies
fi_mid = N_FREQ // 2
bv_mid = bvals_all[:, :, :, fi_mid]       # (N_TIMES, N_ORIENT, NPIX)
bv_norm = bvals_all / (bv_mid[:, :, :, np.newaxis] + 1e-10)   # ratio vs midband
bv_chrom_rms = float(np.std(bv_norm))     # rms variation in beam shape ratio
bv_max_var   = float(np.std(bvals_all.mean(axis=(0,1)), axis=0).max() /
                     bvals_all.mean(axis=(0,1)).max())

print(f"\nBeam chromaticity (bowtie vs BLOOM thin dipole):")
print(f"  Beam-value ratio rms  (all orient/pix/freq): {bv_chrom_rms:.4f}")
bv_mean_vs_freq = bvals_all.mean(axis=(0, 1, 2))   # mean over time/orient/pix
print(f"  Mean beam vs freq: min={bv_mean_vs_freq.min():.4f}  max={bv_mean_vs_freq.max():.4f}"
      f"  ratio={bv_mean_vs_freq.max()/bv_mean_vs_freq.min():.2f}×")

# ── Grouped-pixel design matrix and inversion ─────────────────────────────
#
# Pixels with mean visibility fraction below vis_threshold are merged into a
# single "dim sky" column (analogous to T_gnd).  Never-visible pixels have
# zero columns regardless and are already NaN-masked by normal_solve.
#
# Column layout:
#   cols 0..n_well-1 : individually-resolved well-visible pixels
#   col  n_well      : merged dim-sky group (rarely-visible pixels)
#   col  n_well+1    : T_gnd fraction
#   col  n_well+2    : pad (0, for normal_solve interface)
#
# y is generated from the full per-pixel A (realistic forward model);
# inversion uses the grouped A.  The difference tests both conditioning
# improvement and any grouping bias simultaneously.

def build_A_grouped(bv_fi, masks_used, well_vis, rarely_vis):
    n_well = int(well_vis.sum())
    bv    = bv_fi.reshape(N_ROWS, NPIX).astype(np.float64)
    m     = np.repeat(masks_used, N_ORIENT, axis=0).astype(np.float64)
    bv_sum = np.sum(bv, axis=1, keepdims=True)
    valid  = bv_sum.ravel() > 0

    A = np.zeros((N_ROWS, n_well + 3), dtype=np.float64)
    A[valid, :n_well] = bv[valid][:, well_vis] * m[valid][:, well_vis] / bv_sum[valid]
    if rarely_vis.any():
        A[valid, n_well] = (np.sum(bv[valid][:, rarely_vis] * m[valid][:, rarely_vis], axis=1)
                            / bv_sum[valid, 0])
    A[valid, n_well + 1] = np.sum(bv[valid] * (1.0 - m[valid]), axis=1) / bv_sum[valid, 0]
    # col n_well+2 stays 0 (pad)
    return A, valid


def run_grouped(bvals_used, masks_used, vis_threshold, label):
    well_vis   = pix_open_frac >= vis_threshold
    rarely_vis = (pix_open_frac > 0) & (~well_vis)
    n_well     = int(well_vis.sum())
    n_rarely   = int(rarely_vis.sum())
    n_never    = int(never_visible.sum())
    npix_eff   = n_well + 1   # sky columns seen by normal_solve

    T_est = np.empty(N_FREQ)
    conds = np.empty(N_FREQ)

    for fi in range(N_FREQ):
        gsm_f = gsm_maps[:, fi].astype(np.float64)
        sky_f = gsm_f + T_21_INJ[fi]

        # Forward model: use full per-pixel A (realistic)
        A_full, _ = build_A(bvals_used[:, :, :, fi], masks_used)
        x_true_full = np.concatenate([sky_f, [T_GND_K], [0.0]])
        y_f = A_full @ x_true_full + T_RX_K

        # Inversion: grouped A
        A_grp, _ = build_A_grouped(bvals_used[:, :, :, fi], masks_used,
                                   well_vis, rarely_vis)
        res = normal_solve(A_grp, y_f, npix_eff)

        sky_well = res['sky_map'][:n_well]
        T_dim    = float(res['sky_map'][n_well])   # dim-group temperature
        n_valid_well = int(np.sum(~np.isnan(sky_well)))
        T_est[fi] = (np.nansum(sky_well) + T_dim * n_rarely) / (n_valid_well + n_rarely)

        lam = res['eigenvalues']
        lam_pos = lam[lam > 0]
        conds[fi] = float(lam_pos.max() / lam_pos.min()) if len(lam_pos) > 1 else np.nan

    resid_nl   = eigenmode_filter(T_est, modes)
    FG_leakage = resid_nl - T_21_filt
    fg_rms     = float(np.std(FG_leakage))
    ratio      = fg_rms / T21_rms
    cond_mean  = float(np.nanmean(conds))

    print(f"\n  [{label}]")
    print(f"    vis_threshold = {vis_threshold:.2f}  "
          f"({n_well} well-vis + {n_rarely} grouped + {n_never} never-vis = {NPIX} total)")
    print(f"    FG leakage rms  = {fg_rms*1e3:8.4f} mK")
    print(f"    T_21_filt rms   = {T21_rms*1e3:8.4f} mK")
    print(f"    FG / signal     = {ratio:8.4f}  ({ratio*100:.1f}%)")
    print(f"    Mean cond(A^TA) = {cond_mean:.3e}")

    return T_est, FG_leakage, ratio, conds


# ── Run the four diagnostic tests ─────────────────────────────────────────────
print(f"\nRunning four diagnostic variants (N_ROWS={N_ROWS}, NPIX={NPIX}):")

# 1. Full model
t0 = time.time()
Te_full, FG_full, ratio_full, conds_full = run_noiseless(
    bvals_all, masks_all, "1. Full (chromatic beam + terrain)")
print(f"    Wall time: {time.time()-t0:.1f} s")

# 2. Achromatic beam: use midband beam values (fi=fi_mid) for all frequencies
bvals_achrom = np.broadcast_to(
    bvals_all[:, :, :, fi_mid:fi_mid+1], bvals_all.shape
).copy()   # (N_TIMES, N_ORIENT, NPIX, N_FREQ) — same beam shape at every freq

t0 = time.time()
Te_achrom, FG_achrom, ratio_achrom, conds_achrom = run_noiseless(
    bvals_achrom, masks_all, "2. Achromatic beam + terrain (midband beam at all freqs)")
print(f"    Wall time: {time.time()-t0:.1f} s")

# 3. No terrain: masks = 1 everywhere
masks_open = np.ones_like(masks_all)

t0 = time.time()
Te_noterr, FG_noterr, ratio_noterr, conds_noterr = run_noiseless(
    bvals_all, masks_open, "3. Chromatic beam + no terrain (full sky always visible)")
print(f"    Wall time: {time.time()-t0:.1f} s")

# 4. Ideal: achromatic beam + no terrain
t0 = time.time()
Te_ideal, FG_ideal, ratio_ideal, conds_ideal = run_noiseless(
    bvals_achrom, masks_open, "4. Ideal (achromatic beam + no terrain)")
print(f"    Wall time: {time.time()-t0:.1f} s")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY TABLE")
print("="*65)
print(f"{'Variant':<42}  {'FG/sig':>7}  {'cond(A^TA)':>12}")
print("-"*65)
for label, ratio, conds in [
    ("1. Full (chromatic + terrain)",       ratio_full,   conds_full),
    ("2. Achromatic beam + terrain",        ratio_achrom, conds_achrom),
    ("3. Chromatic beam + no terrain",      ratio_noterr, conds_noterr),
    ("4. Achromatic beam + no terrain",     ratio_ideal,  conds_ideal),
]:
    print(f"  {label:<40}  {ratio:7.4f}  {np.nanmean(conds):12.3e}")

# ── Attribution ───────────────────────────────────────────────────────────────
print("\nEffect attribution:")
fg_beam    = ratio_full   - ratio_noterr   # incremental effect of chromaticity
fg_terrain = ratio_full   - ratio_achrom   # incremental effect of terrain
fg_base    = ratio_ideal                   # residual with neither effect
print(f"  Achromatic ideal FG/sig:          {ratio_ideal*100:6.2f}%")
print(f"  Added by chromatic beam alone:  + {(ratio_noterr - ratio_ideal)*100:6.2f}%"
      f"  (Test3 - Test4)")
print(f"  Added by terrain alone:         + {(ratio_achrom - ratio_ideal)*100:6.2f}%"
      f"  (Test2 - Test4)")
print(f"  Interaction (beam × terrain):   + {(ratio_full - ratio_noterr - ratio_achrom + ratio_ideal)*100:6.2f}%")
print(f"  Full model total:               = {ratio_full*100:6.2f}%")

# ── Pixel-grouping threshold scan ─────────────────────────────────────────
print("\n" + "="*65)
print("=== Pixel-Grouping Threshold Scan (full chromatic beam + terrain) ===")
print("="*65)

thresholds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
grouped_results = []
for thr in thresholds:
    t0 = time.time()
    Te_g, FG_g, ratio_g, conds_g = run_grouped(
        bvals_all, masks_all, thr,
        f"Grouped vis_threshold={thr:.2f}")
    print(f"    Wall time: {time.time()-t0:.1f} s")
    grouped_results.append((thr, ratio_g, float(np.nanmean(conds_g))))

print(f"\n{'Threshold':>10}  {'n_well':>7}  {'n_grp':>6}  {'FG/sig':>8}  {'cond(A^TA)':>12}")
print("-"*50)
for thr, ratio_g, cond_g in grouped_results:
    n_w = int((pix_open_frac >= thr).sum())
    n_r = int(((pix_open_frac > 0) & (pix_open_frac < thr)).sum())
    print(f"  {thr:8.2f}  {n_w:7d}  {n_r:6d}  {ratio_g:8.4f}  {cond_g:12.3e}")
print(f"\n  (baseline full model: FG/sig={ratio_full:.4f}  cond={np.nanmean(conds_full):.3e})")
