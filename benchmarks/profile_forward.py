"""
Profile the current ForwardModel pipeline with synthetic inputs.

This benchmark is intentionally dependency-light: it avoids pygdsm and optional
line profilers so it can run in any development environment with the package's
core dependencies installed.

Examples
--------
    python benchmarks/profile_forward.py
    python benchmarks/profile_forward.py --mode lunar --ntimes 256 --nside-sky 16
    python benchmarks/profile_forward.py --profile-precompute
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import astropy.units as u
import healpy
import jax
import numpy as np
from astropy.time import Time

from eigsep_sim import Beam, EarthSurface, ForwardModel, LunarOrbit, Sky


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def fmt_time(seconds):
    if seconds >= 1:
        return f"{seconds:.3f} s"
    return f"{seconds * 1e3:.1f} ms"


def bench(label, fn, n_runs=5, n_warmup=0):
    for _ in range(n_warmup):
        block(fn())
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        block(result)
        times.append(time.perf_counter() - t0)
    best = min(times)
    mean = float(np.mean(times))
    print(f"  {label:<42s} best {fmt_time(best):>9s}  mean {fmt_time(mean):>9s}")
    return result, best


def geom_nbytes(geom):
    total = 0
    for value in geom.values():
        if hasattr(value, "nbytes"):
            total += value.nbytes
    return total


def build_case(args):
    rng = np.random.default_rng(args.seed)
    freqs_hz = np.linspace(args.fmin_mhz, args.fmax_mhz, args.nfreq) * 1e6

    if args.mode == "earth":
        observer = EarthSurface(lat=39.2, lon=-113.4, height=1600.0)
        t0 = Time("2025-06-21T04:00:00")
        times = t0 + np.linspace(0, args.hours * 3600, args.ntimes) * u.s
    else:
        observer = LunarOrbit(
            altitude=args.lunar_alt_km * 1e3,
            rot_orbit_vec=[0.0, 0.0, 1.0],
            rot_spin_vec=[0.0, 0.0, 1.0],
            spin_period=args.spin_period_s,
        )
        times = observer.t0 + np.linspace(0, args.hours * 3600, args.ntimes) * u.s

    beam = Beam.from_dipole(
        args.nside_beam,
        freqs_hz,
        arm_lengths_m=3.0,
        K=args.k_beam,
    )

    npix_sky = healpy.nside2npix(args.nside_sky)
    sky_map = 1000.0 + 100.0 * rng.standard_normal((npix_sky, args.nfreq))
    sky = Sky.from_map(args.nside_sky, freqs_hz, sky_map, n_modes=args.k_sky)
    sky_coeffs = sky.basis.project(sky_map).astype(np.float32)

    fwd = ForwardModel(observer, beam, sky)
    return fwd, sky_coeffs, beam.coeffs, times


def print_profile(label, fn, n_stats):
    pr = cProfile.Profile()
    pr.enable()
    result = fn()
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(n_stats)
    print(f"\n[{label} cProfile]")
    print(buf.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["earth", "lunar"], default="earth")
    parser.add_argument("--nside-sky", type=int, default=16)
    parser.add_argument("--nside-beam", type=int, default=32)
    parser.add_argument("--nfreq", type=int, default=16)
    parser.add_argument("--ntimes", type=int, default=256)
    parser.add_argument("--k-sky", type=int, default=3)
    parser.add_argument("--k-beam", type=int, default=4)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--fmin-mhz", type=float, default=50.0)
    parser.add_argument("--fmax-mhz", type=float, default=150.0)
    parser.add_argument("--lunar-alt-km", type=float, default=100.0)
    parser.add_argument("--spin-period-s", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-precompute", action="store_true")
    parser.add_argument("--profile-stats", type=int, default=20)
    args = parser.parse_args()

    fwd, sky_coeffs, beam_coeffs, times = build_case(args)

    print("=" * 78)
    print(f"ForwardModel profile [{args.mode}] on {jax.default_backend()}")
    print(
        f"sky nside={fwd.sky.nside} npix={fwd.sky.npix}, "
        f"beam nside={fwd.beam.nside} npix={fwd.beam.npix}, "
        f"ntimes={args.ntimes}, nfreq={args.nfreq}"
    )
    print("=" * 78)

    if args.profile_precompute:
        geom = print_profile(
            "precompute_geometry(times)",
            lambda: fwd.precompute_geometry(times=times),
            args.profile_stats,
        )
    else:
        geom, _ = bench(
            "precompute_geometry(times)",
            lambda: fwd.precompute_geometry(times=times),
            n_runs=3,
        )

    print(f"  geom returned arrays                 {geom_nbytes(geom) / 1024**2:.1f} MB")

    sky_mask = fwd.build_sky_mask(times=times)
    geom_mask, _ = bench(
        "precompute_geometry(times, sky_mask)",
        lambda: fwd.precompute_geometry(times=times, sky_mask=sky_mask),
        n_runs=3,
    )
    print(
        f"  sky_mask keeps {int(sky_mask.sum())}/{sky_mask.size} pixels "
        f"({100 * sky_mask.mean():.1f}%)"
    )
    print(
        f"  masked geom returned arrays          "
        f"{geom_nbytes(geom_mask) / 1024**2:.1f} MB"
    )

    print("\n[simulate]")
    _, compile_t = bench(
        "simulate full [compile+run]",
        lambda: fwd.simulate(sky_coeffs, beam_coeffs, geom=geom, T_gnd=300.0),
        n_runs=1,
    )
    _, full_t = bench(
        "simulate full [steady]",
        lambda: fwd.simulate(sky_coeffs, beam_coeffs, geom=geom, T_gnd=300.0),
        n_runs=5,
    )
    _, compile_mask_t = bench(
        "simulate sky_mask [compile+run]",
        lambda: fwd.simulate(
            sky_coeffs, beam_coeffs, geom=geom_mask, T_gnd=300.0
        ),
        n_runs=1,
    )
    _, mask_t = bench(
        "simulate sky_mask [steady]",
        lambda: fwd.simulate(
            sky_coeffs, beam_coeffs, geom=geom_mask, T_gnd=300.0
        ),
        n_runs=5,
    )

    if full_t > 0:
        print(f"\n  steady sky_mask speedup              {full_t / mask_t:.2f}x")
    print(
        f"  compile overhead full/masked         "
        f"{fmt_time(max(compile_t - full_t, 0.0))} / "
        f"{fmt_time(max(compile_mask_t - mask_t, 0.0))}"
    )


if __name__ == "__main__":
    main()
