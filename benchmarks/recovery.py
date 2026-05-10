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

    # Perturbed starting point (20% sky, 10% beam)
    params_start = {
        'sky_coeffs':  gsm_coeffs * 1.2,
        'beam_coeffs': beam_coeffs_true * 0.9,
    }

    cal = Calibrator(fwd, data, inv_noise_var=inv_noise_var, lam_beam=0.01)
    cal._beam_nom = beam_coeffs_true.copy()
    cal._geom = geom

    print()
    _, _ = bench("sky_step()  [JIT compile]", cal.sky_step, params_start, n_runs=1)
    _, _ = bench("sky_step()  [steady state]", cal.sky_step, params_start, n_warmup=0, n_runs=3)
    _, _ = bench("beam_step() [JIT compile]", cal.beam_step, params_start, n_runs=1)
    _, _ = bench("beam_step() [steady state]", cal.beam_step, params_start, n_warmup=0, n_runs=3)

    # ── Convergence ───────────────────────────────────────────────────────
    print(f"\n[ Convergence  (perturbation: +20% sky, -10% beam) ]\n")
    header = (f"  {'Iter':>4}  {'Loss':>12}  {'ΔLoss%':>8}"
              f"  {'Sky RMS err':>12}  {'Beam RMS err':>12}  {'ms/iter':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    params = dict(params_start)
    loss_prev = float(cal._loss(params))
    sky_err = float(np.std(params['sky_coeffs'] - gsm_coeffs))
    beam_err = float(np.std(params['beam_coeffs'] - beam_coeffs_true))
    print(f"  {'init':>4}  {loss_prev:12.4e}  {'---':>8}"
          f"  {sky_err:12.4e}  {beam_err:12.4e}  {'---':>8}")

    t_fit_start = time.perf_counter()
    n_iter_done = 0
    for i in range(args.max_iter):
        t_iter = time.perf_counter()
        params = cal.sky_step(params)
        params = cal.beam_step(params)
        _block(params['sky_coeffs'])
        ms_iter = (time.perf_counter() - t_iter) * 1e3

        loss_new = float(cal._loss(params))
        sky_err = float(np.std(params['sky_coeffs'] - gsm_coeffs))
        beam_err = float(np.std(params['beam_coeffs'] - beam_coeffs_true))
        pct_change = 100.0 * (loss_prev - loss_new) / abs(loss_prev) if loss_prev != 0 else 0.0
        print(f"  {i+1:>4}  {loss_new:12.4e}  {pct_change:+8.2f}%"
              f"  {sky_err:12.4e}  {beam_err:12.4e}  {ms_iter:8.0f}")
        n_iter_done = i + 1
        if abs(pct_change) < 1e-4:
            print(f"  (converged)")
            break
        loss_prev = loss_new

    t_fit_total = time.perf_counter() - t_fit_start
    ms_per_iter = t_fit_total / n_iter_done * 1e3
    print(f"\n  Total: {t_fit_total:.2f} s over {n_iter_done} iterations"
          f"  ({ms_per_iter:.0f} ms/iter)")

    final_beam_err = float(np.std(params['beam_coeffs'] - beam_coeffs_true))
    init_beam_err = float(np.std(params_start['beam_coeffs'] - beam_coeffs_true))
    if abs(final_beam_err - init_beam_err) / init_beam_err < 0.01:
        print(f"\n  NOTE: beam_step made no progress ({final_beam_err:.4e} unchanged).")
        print(f"        After sky convergence, the sky compensates for beam error,")
        print(f"        so any beam move increases the data loss. Consider:")
        print(f"          - Higher lam_beam (stronger regularization pull toward beam_nom)")
        print(f"          - Larger lr for beam_step to escape the local minimum")
        print(f"          - Simultaneous (not alternating) sky/beam gradient steps")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nside",      type=int, default=8,  help="HEALPix nside (default 8)")
    p.add_argument("--nfreq",      type=int, default=20, help="Number of frequencies (default 20)")
    p.add_argument("--ntimes",     type=int, default=24, help="Number of time steps (default 24)")
    p.add_argument("--n_dipoles",  type=int, default=1,  help="Number of dipoles (default 1)")
    p.add_argument("--k_beam",     type=int, default=5,  help="Beam basis modes (default 5)")
    p.add_argument("--k_sky",      type=int, default=5,  help="Sky basis modes (default 5)")
    p.add_argument("--max_iter",   type=int, default=15, help="Max calibration iterations (default 15)")
    main(p.parse_args())
