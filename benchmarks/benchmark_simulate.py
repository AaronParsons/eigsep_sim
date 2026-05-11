"""
ForwardModel benchmark — JAX integration with body_rots and transmitters.

Measures wall-clock time at Canyon Demo scale:
  nside_beam=64, nside_sky=16, N_FREQ=20, N_AZ×N_ALT = 36×36 = 1296 pointings.

Sections
--------
1. precompute_geometry   — rots path, with/without body_rots, with/without TX
2. simulate (JIT kernel) — warmup (first call) vs. steady-state; 0/1/3 TX sources
3. End-to-end            — precompute + simulate for the Canyon Demo scenario

Usage
-----
    python benchmarks/benchmark_simulate.py
"""

import time
import numpy as np
import healpy
from astropy.time import Time

from eigsep_sim import Sky, Beam, ForwardModel
from eigsep_sim.observer import EarthSurface
from eigsep_sim.beam import BEAM_NPZ
from eigsep_sim.basis import BeamBasis


# ── Helpers ───────────────────────────────────────────────────────────────────

def hms(seconds):
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    return f"{seconds * 1e3:.1f} ms"


def bench(label, fn, n_repeat=3, warmup=0):
    """Run fn() n_repeat times (after warmup calls) and report min wall time."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:<55} {hms(best):>10}  (best of {n_repeat})")
    return best


# ── Build objects (not timed) ─────────────────────────────────────────────────

print("Building objects …")

LAT, LON, HEIGHT_M = 39.2, -113.4, 1600.0
FREQ_MIN, FREQ_MAX, N_FREQ = 50e6, 200e6, 20
NSIDE_SKY, N_SKY_MODES = 16, 3
K_BEAM = 5
N_AZ, N_ALT = 36, 36

freqs_hz  = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
az_rad    = np.linspace(0, 2 * np.pi, N_AZ,  endpoint=False)
alt_rad   = np.linspace(0, 2 * np.pi, N_ALT, endpoint=False)
n_scan    = N_AZ * N_ALT      # 1296

observer = EarthSurface(lat=LAT, lon=LON, height=HEIGHT_M)
observer.set_time(Time("2025-06-21T04:00:00"))

basis, coeffs = BeamBasis.from_beam_file(BEAM_NPZ, freqs_hz, K_BEAM)
npix_beam  = coeffs.shape[1]
nside_beam = healpy.npix2nside(npix_beam)
beam = Beam(nside_beam, freqs_hz, basis, coeffs,
            u_body=np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

sky        = Sky.from_gsm(NSIDE_SKY, freqs_hz, n_modes=N_SKY_MODES)
sky_coeffs = sky.init_coeffs()
npix_sky   = sky.npix

R_gal2top = observer.rot_gal2top().astype(np.float32)
rots_scan = [R_gal2top] * n_scan

body_rots_scan = [Beam.top2body(az, alt)
                  for az in az_rad for alt in alt_rad]

TX_DIR   = np.array([0.0, 0.0, -1.0])     # nadir
TX_FREQS = freqs_hz[::4]
TX_POW   = 1e6 * np.ones_like(TX_FREQS)

# Three ForwardModels: no TX, 1 TX, 3 TX
fwd_0tx = ForwardModel(observer, beam, sky)
fwd_1tx = ForwardModel(observer, beam, sky,
                       transmitters=[(TX_DIR, TX_FREQS, TX_POW)])
fwd_3tx = ForwardModel(observer, beam, sky,
                       transmitters=[(TX_DIR,                 TX_FREQS, TX_POW),
                                     (np.array([1., 0., 0.]), TX_FREQS, TX_POW * 0.1),
                                     (np.array([0., 1., 0.]), TX_FREQS, TX_POW * 0.1)])

print(f"  sky: nside={NSIDE_SKY}, npix={npix_sky}, modes={N_SKY_MODES}")
print(f"  beam: nside={nside_beam}, npix={npix_beam}, modes={K_BEAM}")
print(f"  scan: {N_AZ}×{N_ALT}={n_scan} pointings, {N_FREQ} freqs")
print()

# ── 1. precompute_geometry ────────────────────────────────────────────────────

print("=" * 70)
print("1. precompute_geometry  (1296 pointings at Canyon Demo scale)")
print("=" * 70)

bench("rots only (no body_rots, no TX)",
      lambda: fwd_0tx.precompute_geometry(rots=rots_scan))

bench("rots + body_rots (no TX)",
      lambda: fwd_0tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan))

bench("rots + body_rots + 1 TX",
      lambda: fwd_1tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan))

bench("rots + body_rots + 3 TX",
      lambda: fwd_3tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan))

print()

# ── 2. simulate (JIT kernel) ──────────────────────────────────────────────────

print("=" * 70)
print("2. simulate — JIT warmup and steady-state  (1296 time steps)")
print("=" * 70)

# Pre-compute all geom objects once (not under benchmark)
geom_0tx = fwd_0tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
geom_1tx = fwd_1tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
geom_3tx = fwd_3tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)

# Force JIT compilation with a one-step geom so the 1296-step warmup is fair
geom_1step_0tx = fwd_0tx.precompute_geometry(rots=[R_gal2top], body_rots=[body_rots_scan[0]])
geom_1step_1tx = fwd_1tx.precompute_geometry(rots=[R_gal2top], body_rots=[body_rots_scan[0]])
geom_1step_3tx = fwd_3tx.precompute_geometry(rots=[R_gal2top], body_rots=[body_rots_scan[0]])

print("  [triggering JIT compilation with 1-step geom …]")
for fwd, geom_1 in [(fwd_0tx, geom_1step_0tx),
                    (fwd_1tx, geom_1step_1tx),
                    (fwd_3tx, geom_1step_3tx)]:
    _ = np.array(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom_1))
print()

# JIT warmup (first full-size call — XLA may re-specialize for new shapes)
print("  First call (potential XLA re-specialization for 1296-step input):")
t_warm_0 = bench("0 TX, first simulate(1296 steps)",
                 lambda: np.array(fwd_0tx.simulate(sky_coeffs, beam.coeffs, geom=geom_0tx)),
                 n_repeat=1)
t_warm_1 = bench("1 TX, first simulate(1296 steps)",
                 lambda: np.array(fwd_1tx.simulate(sky_coeffs, beam.coeffs, geom=geom_1tx)),
                 n_repeat=1)
t_warm_3 = bench("3 TX, first simulate(1296 steps)",
                 lambda: np.array(fwd_3tx.simulate(sky_coeffs, beam.coeffs, geom=geom_3tx)),
                 n_repeat=1)
print()

print("  Steady-state (subsequent calls, best of 5):")
t_ss_0 = bench("0 TX",
               lambda: np.array(fwd_0tx.simulate(sky_coeffs, beam.coeffs, geom=geom_0tx)),
               n_repeat=5)
t_ss_1 = bench("1 TX",
               lambda: np.array(fwd_1tx.simulate(sky_coeffs, beam.coeffs, geom=geom_1tx)),
               n_repeat=5)
t_ss_3 = bench("3 TX",
               lambda: np.array(fwd_3tx.simulate(sky_coeffs, beam.coeffs, geom=geom_3tx)),
               n_repeat=5)

if t_ss_0 > 0:
    print(f"\n  TX overhead vs 0-TX baseline:")
    print(f"    1 TX: {(t_ss_1 - t_ss_0) / t_ss_0 * 100:+.1f}%")
    print(f"    3 TX: {(t_ss_3 - t_ss_0) / t_ss_0 * 100:+.1f}%")
print()

# ── 3. End-to-end ─────────────────────────────────────────────────────────────

print("=" * 70)
print("3. End-to-end  (precompute_geometry + simulate, 1296 pointings)")
print("=" * 70)

def e2e_0tx():
    g = fwd_0tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
    return np.array(fwd_0tx.simulate(sky_coeffs, beam.coeffs, geom=g))

def e2e_1tx():
    g = fwd_1tx.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
    return np.array(fwd_1tx.simulate(sky_coeffs, beam.coeffs, geom=g))

bench("0 TX  (sky only)", e2e_0tx, n_repeat=3)
bench("1 TX  (sky + nadir TX)", e2e_1tx, n_repeat=3)

print()
print("Done.")
