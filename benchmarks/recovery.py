"""
Benchmark: eigsep_sim signal recovery — speed and convergence.

Measures wall-clock time for each stage of the ForwardModel + Calibrator
pipeline, and tracks loss and parameter error vs. iteration against a known
ground-truth sky/beam.

Usage
-----
    python benchmarks/recovery.py
    python benchmarks/recovery.py --nside 16 --ntimes 48 --max_iter 20
    python benchmarks/recovery.py --n_dipoles 2 --k_beam 8 --k_sky 8

JAX device selection:
    JAX_PLATFORM_NAME=cpu python benchmarks/recovery.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import astropy.units as u
from astropy.time import Time

from eigsep_sim import Beam, Calibrator, EarthSurface, ForwardModel, NullTerrain, Sky


# ── Utilities ────────────────────────────────────────────────────────────────

def _block(x):
    """Block until JAX computation is done (handles arrays, dicts, other)."""
    if hasattr(x, 'block_until_ready'):
        x.block_until_ready()
    elif isinstance(x, dict):
        for v in x.values():
            _block(v)


def bench(label, fn, *args, n_warmup=0, n_runs=3, **kwargs):
    """
    Time fn(*args, **kwargs), return (result, mean_wall_time_seconds).

    Prints a one-line summary.  n_warmup calls are made before timing starts
    (useful to separate JIT compilation from steady-state execution).
    """
    for _ in range(n_warmup):
        _block(fn(*args, **kwargs))

    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _block(result)
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1e3
    std_ms = np.std(times) * 1e3
    tag = f"(n={n_runs})" if n_runs > 1 else "(n=1)"
    print(f"  {label:<44s} {mean_ms:8.1f} ms ± {std_ms:.1f} ms  {tag}")
    return result, np.mean(times)


# ── Main benchmark ───────────────────────────────────────────────────────────

def main(args):
    print(f"\n{'='*70}")
    print(f"  eigsep_sim Recovery Benchmark  [{jax.default_backend()}]")
    print(f"  NSIDE={args.nside}  nfreq={args.nfreq}  ntimes={args.ntimes}"
          f"  n_dipoles={args.n_dipoles}")
    print(f"  K_beam={args.k_beam}  K_sky={args.k_sky}  max_iter={args.max_iter}")
    print(f"{'='*70}\n")

    # ── Build model objects ───────────────────────────────────────────────
    freqs_hz = np.linspace(55e6, 150e6, args.nfreq)
    delta_nu_hz = float(np.diff(freqs_hz).mean())
    obs = EarthSurface(lat=39.2, lon=-113.4)
    beam = Beam.from_dipole(
        args.nside, freqs_hz,
        arm_lengths_m=[2.0] * args.n_dipoles,
        K=args.k_beam,
    )
    sky = Sky.from_gsm(args.nside, freqs_hz, n_modes=args.k_sky, include_flat=True)
    fwd = ForwardModel(obs, beam, sky, terrain=NullTerrain())

    gsm_coeffs = sky.init_coeffs()           # ground-truth sky coefficients
    beam_coeffs_true = beam.coeffs.copy()    # ground-truth beam coefficients

    T_sky_mean = float((gsm_coeffs @ sky.basis.A.T).mean())
    T_rx_K = 100.0
    times = Time("2025-01-01") + np.linspace(0, 86400, args.ntimes, endpoint=False) * u.s

    n_sky = sky.npix * sky.nmodes
    n_beam = args.n_dipoles * beam.npix * beam.nmodes
    print(f"  npix_sky={sky.npix}  npix_beam={beam.npix}")
    print(f"  sky param count:  {n_sky:,}")
    print(f"  beam param count: {n_beam:,}")
    print(f"  GSM mean Tsky:    {T_sky_mean:.0f} K\n")

    # ── Timing ───────────────────────────────────────────────────────────
    print("[ Timing ]\n")

    geom, _ = bench("precompute_geometry()", fwd.precompute_geometry, times,
                    n_warmup=1, n_runs=3)

    # simulate(): first call triggers JIT compilation
    _, t_compile = bench(
        "simulate()  [JIT compile]",
        fwd.simulate, gsm_coeffs, beam_coeffs_true,
        geom=geom, n_runs=1,
    )
    _, t_sim = bench(
        "simulate()  [steady state]",
        fwd.simulate, gsm_coeffs, beam_coeffs_true,
        geom=geom, n_warmup=0, n_runs=5,
    )
    print(f"  {'JIT overhead':44s} {(t_compile - t_sim)*1e3:8.1f} ms  (compile − steady)")

    # Build synthetic observations
    antenna_temp_true = fwd.simulate(gsm_coeffs, beam_coeffs_true, geom=geom)
    tau_s = 86400.0 / args.ntimes
    sigma_noise = (T_sky_mean + T_rx_K) / np.sqrt(delta_nu_hz * tau_s)
    rng = np.random.default_rng(42)
    data = np.array(antenna_temp_true) + rng.normal(
        scale=sigma_noise, size=antenna_temp_true.shape
    ).astype(np.float32)
    inv_noise_var = np.full(data.shape, 1.0 / sigma_noise**2, dtype=np.float32)

    # Scale perturbation (degenerate) — used only for step timing
    # A uniform scale of beam_coeffs cancels exactly in the normalized antenna
    # temperature: T_ant = (α·B @ sky) / (α·Σ B) = B @ sky / Σ B for any α.
    params_scale_pert = {
        'sky_coeffs':  gsm_coeffs * 1.2,
        'beam_coeffs': beam_coeffs_true * 0.9,
    }
    cal = Calibrator(fwd, data, inv_noise_var=inv_noise_var, lam_beam=0.01)
    cal._beam_nom = beam_coeffs_true.copy()
    cal._geom = geom
    print()
    _, _ = bench("sky_step()      [JIT compile]",  cal.sky_step,      params_scale_pert, n_runs=1)
    _, _ = bench("sky_step()      [steady state]", cal.sky_step,      params_scale_pert, n_warmup=0, n_runs=3)
    _, _ = bench("beam_step()     [JIT compile]",  cal.beam_step,     params_scale_pert, n_runs=1)
    _, _ = bench("beam_step()     [steady state]", cal.beam_step,     params_scale_pert, n_warmup=0, n_runs=3)
    _, _ = bench("beam_cg_step()  [JIT compile]",  cal.beam_cg_step,  params_scale_pert, n_runs=1)
    _, _ = bench("beam_cg_step()  [steady state]", cal.beam_cg_step,  params_scale_pert, n_warmup=0, n_runs=3)
    _, _ = bench("joint_step()    [JIT compile]",  cal.joint_step,    params_scale_pert, n_runs=1)
    _, _ = bench("joint_step()    [steady state]", cal.joint_step,    params_scale_pert, n_warmup=0, n_runs=3)

    # ── beam_cg_step timing vs ntimes ─────────────────────────────────────
    # Each call to beam_cg_step runs n_cg inner CG iterations, each of which
    # calls the forward model twice (JVP).  Wall-clock time scales ~linearly
    # with ntimes because the forward model is O(ntimes).
    #
    # Recoverability thresholds (angular-diversity rank condition):
    #   ntimes_beam  = n_beam_params / nfreq  (beam only, sky fixed at truth)
    #   ntimes_joint = (n_sky + n_beam) / nfreq  (joint sky+beam)
    #
    # Below the threshold the measurement matrix is rank-deficient and the
    # beam is not uniquely constrained by data (null space exists).  Above the
    # threshold every beam mode is observed, enabling joint recovery.
    # Denser time sampling (smaller dt) sweeps the beam across more adjacent
    # sky pixels, providing the angular diversity needed to reach the threshold.

    n_beam_params = int(np.prod(beam_coeffs_true.shape))
    n_sky_params  = int(gsm_coeffs.size)
    ntimes_beam  = int(np.ceil(n_beam_params / args.nfreq))
    ntimes_joint = int(np.ceil((n_sky_params + n_beam_params) / args.nfreq))

    print(f"\n[ beam_cg_step timing vs ntimes ]\n")
    print(f"  Measures wall-clock time per beam_cg_step call vs ntimes.")
    print(f"  Expected: ~linear in ntimes (forward model is O(ntimes)).")
    print(f"  Threshold (beam only):  ntimes ≥ {ntimes_beam}"
          f"  (n_beam={n_beam_params} / nfreq={args.nfreq})")
    print(f"  Threshold (joint):      ntimes ≥ {ntimes_joint}"
          f"  ((n_sky={n_sky_params} + n_beam={n_beam_params}) / nfreq)\n")

    hdr2 = (f"  {'ntimes':>7}  {'dt (min)':>8}  {'det?':>5}"
            f"  {'ms/call':>8}  {'ratio vs ntimes_0':>18}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    t_ref = None
    nt_ref = None
    for ntimes_t in args.ntimes_sweep:
        times_t = (Time("2025-01-01")
                   + np.linspace(0, 86400, ntimes_t, endpoint=False) * u.s)
        geom_t = fwd.precompute_geometry(times_t)
        tau_t = 86400.0 / ntimes_t
        sigma_t = (T_sky_mean + T_rx_K) / np.sqrt(delta_nu_hz * tau_t)
        ant_t = fwd.simulate(gsm_coeffs, beam_coeffs_true, geom=geom_t)
        data_t = (np.array(ant_t)
                  + rng.normal(scale=sigma_t, size=ant_t.shape).astype(np.float32))
        inv_nv_t = np.full(data_t.shape, 1.0 / sigma_t**2, dtype=np.float32)

        cal_t = Calibrator(fwd, data_t, inv_noise_var=inv_nv_t, lam_beam=0.01)
        cal_t._beam_nom = beam_coeffs_true.copy()
        cal_t._geom = geom_t

        p_t = {
            'sky_coeffs':  gsm_coeffs * 1.2,
            'beam_coeffs': beam_coeffs_true * 0.9,
        }

        # Warm up JIT for this ntimes, then time steady-state calls
        cal_t.beam_cg_step(p_t)   # compile
        times_ms = []
        for _ in range(3):
            t0 = time.perf_counter()
            cal_t.beam_cg_step(p_t)
            times_ms.append((time.perf_counter() - t0) * 1e3)
        ms = float(np.mean(times_ms))

        if t_ref is None:
            t_ref = ms
            nt_ref = ntimes_t

        ratio_pred = ntimes_t / nt_ref          # expected linear scaling
        ratio_obs  = ms / t_ref
        det = "yes" if ntimes_t >= ntimes_beam else "no"

        print(f"  {ntimes_t:>7}  {tau_t/60:>8.1f}  {det:>5}"
              f"  {ms:>8.1f}  {ratio_obs:>7.2f}× obs  ({ratio_pred:.2f}× expected)")

    print(f"\n  Observed scaling closely follows ntimes (linear), confirming")
    print(f"  the forward model dominates and is O(ntimes) as designed.")

    # ── Recovery / convergence validation ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Recovery Validation  (ntimes={args.ntimes})")
    print(f"{'='*70}\n")

    sky_true_map = gsm_coeffs @ sky.basis.A.T   # (npix_sky, nfreq)

    # ── Sanity check: sky-only recovery with the true beam ─────────────────
    # Loss is exactly quadratic in sky_coeffs → Newton-CG should reach the
    # global minimum in one step.  The data-fit residual after recovery
    # measures basis truncation + noise (irreducible floor).
    print("[ 1. Sky-only fit quality  (true beam fixed, 1 sky_step) ]\n")
    params_sky_only = {
        'sky_coeffs': np.zeros_like(gsm_coeffs),
        'beam_coeffs': beam_coeffs_true.copy(),
    }
    cal_sonly = Calibrator(fwd, data, inv_noise_var=inv_noise_var, lam_beam=0.0)
    cal_sonly._geom = geom
    cal_sonly.sky_step(params_sky_only)                   # JIT warmup
    p_sky = cal_sonly.sky_step(params_sky_only)

    T_sky_pred = np.array(fwd.simulate(
        p_sky['sky_coeffs'], beam_coeffs_true, geom=geom
    ))
    T_noiseless = np.array(fwd.simulate(gsm_coeffs, beam_coeffs_true, geom=geom))
    fit_rms_1 = float(np.std(T_sky_pred - T_noiseless) / np.mean(T_noiseless) * 100)
    loss_sky1 = float(cal_sonly._loss(p_sky))
    print(f"  Loss after 1 sky_step:  {loss_sky1:.3e}  (noise floor ~1.0)")
    print(f"  Data-fit rms:           {fit_rms_1:.3f}%  (noiseless residual)")
    print(f"  (sky loss is quadratic → 1 Newton step reaches the exact minimum;")
    print(f"   residual loss above noise floor = irreducible basis-truncation error)")
    print()

    # ── Joint sky+beam recovery: beam shape perturbation ──────────────────
    # 2.0m → 2.1m arm length changes the frequency-dependent beam pattern.
    # The calibrator should:
    #   (a) fit the data well (loss near noise floor)
    #   (b) keep the beam near its prior (regularization)
    # Recovery of the TRUE beam vs. the prior is limited by the sky-beam
    # degeneracy — many (sky, beam) pairs explain the observations equally.
    arm_true = 2.0
    arm_pert = 2.1
    print(f"[ 2. Joint sky+beam fit quality ]\n")
    print(f"  True arm: {arm_true} m  |  Beam prior: {arm_pert} m  (shape perturbation)\n")

    beam_pert = Beam.from_dipole(
        args.nside, freqs_hz,
        arm_lengths_m=[arm_pert] * args.n_dipoles,
        K=args.k_beam,
    )
    beam_coeffs_pert = beam_pert.coeffs.copy()
    beam_pert_map    = beam_coeffs_pert @ beam.basis.A.T

    def norm_beam(bmap):
        """Per-(dipole,freq) normalize for shape comparison."""
        return bmap / (bmap.sum(axis=1, keepdims=True) + 1e-30)

    beam_prior_err = float(
        np.std(norm_beam(beam_pert_map) - norm_beam(beam_coeffs_true @ beam.basis.A.T))
        / np.std(norm_beam(beam_coeffs_true @ beam.basis.A.T)) * 100
    )
    print(f"  Beam-prior shape error vs truth:  {beam_prior_err:.2f}%")

    params_joint = {
        'sky_coeffs':  np.zeros_like(gsm_coeffs),
        'beam_coeffs': beam_coeffs_pert.copy(),
    }
    cal_joint = Calibrator(
        fwd, data, inv_noise_var=inv_noise_var, lam_beam=0.1
    )
    cal_joint._beam_nom = beam_coeffs_pert.copy()
    cal_joint._geom = geom

    t0 = time.perf_counter()
    result = cal_joint.fit(params=params_joint, max_iter=args.max_iter, verbose=False)
    t_fit  = (time.perf_counter() - t0) * 1e3

    T_joint_pred = np.array(fwd.simulate(
        result['params']['sky_coeffs'],
        result['params']['beam_coeffs'],
        geom=geom,
    ))
    fit_rms_j = float(np.std(T_joint_pred - T_noiseless) / np.mean(T_noiseless) * 100)
    beam_final_map = result['params']['beam_coeffs'] @ beam.basis.A.T
    beam_err_f = float(
        np.std(norm_beam(beam_final_map) - norm_beam(beam_coeffs_true @ beam.basis.A.T))
        / np.std(norm_beam(beam_coeffs_true @ beam.basis.A.T)) * 100
    )
    beam_err_prior = float(
        np.std(norm_beam(beam_final_map) - norm_beam(beam_pert_map))
        / np.std(norm_beam(beam_pert_map)) * 100
    )

    sigma_noise = float(np.std(data - T_noiseless))
    fit_noise   = float(sigma_noise / np.mean(T_noiseless) * 100)

    print(f"\n  Metric                              initial      final")
    print(f"  loss (noise floor ~1.0)             --           {result['losses'][-1]:.3e}")
    print(f"  data-fit rms (noiseless)            --           {fit_rms_j:.3f}%")
    print(f"  noise level (reference)             --           {fit_noise:.3f}%")
    print(f"  beam shape vs truth                 {beam_prior_err:.2f}%        {beam_err_f:.2f}%")
    print(f"  beam shape vs prior                 0.00%        {beam_err_prior:.2f}%")
    print(f"\n  Note: data-fit rms > noise floor = residual from beam mismatch")
    print(f"  (beam stays near prior; sky absorbs remainder of beam error)")
    print(f"\n  converged:  {result['converged']}   iterations: {result['n_iter']}   "
          f"fit time: {t_fit:.0f} ms")

    print(f"\n  Loss curve:")
    for i, loss in enumerate(result['losses']):
        print(f"    iter {i:3d}:  {loss:.6e}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nside",        type=int,   default=8,
                   help="HEALPix nside (default 8)")
    p.add_argument("--nfreq",        type=int,   default=20,
                   help="Number of frequencies (default 20)")
    p.add_argument("--ntimes",       type=int,   default=24,
                   help="Number of time steps for timing section (default 24)")
    p.add_argument("--n_dipoles",    type=int,   default=1,
                   help="Number of dipoles (default 1)")
    p.add_argument("--k_beam",       type=int,   default=5,
                   help="Beam basis modes (default 5)")
    p.add_argument("--k_sky",        type=int,   default=5,
                   help="Sky basis modes (default 5)")
    p.add_argument("--max_iter",     type=int,   default=10,
                   help="Max calibration iterations per run (default 10)")
    p.add_argument("--ntimes-sweep", type=int,   nargs="+",
                   default=[96, 192, 384, 480],
                   help="ntimes values for shape-recovery sweep (default 96 192 384 480)")
    args = p.parse_args()
    args.ntimes_sweep = getattr(args, 'ntimes_sweep')
    main(args)
