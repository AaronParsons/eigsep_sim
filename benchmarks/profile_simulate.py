"""
ForwardModel profiling — identifies optimization targets.

Runs three profiling passes at Canyon Demo scale
(nside_beam=64, nside_sky=16, 20 freqs, 1296 pointings):

  1. cProfile on precompute_geometry  — function-call cost breakdown
  2. line_profiler on precompute_geometry + _build_sim_fn internals
  3. JAX profiler on simulate          — XLA kernel breakdown (Chrome trace)

Usage
-----
    python benchmarks/profile_simulate.py
    # Then open benchmarks/jax_profile/ in chrome://tracing
"""

import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np
import healpy
from astropy.time import Time
try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None

import jax
import jax.profiler

from eigsep_sim import Sky, Beam, ForwardModel
from eigsep_sim.observer import EarthSurface
from eigsep_sim.beam import BEAM_NPZ
from eigsep_sim.basis import BeamBasis


# ── Build objects (not profiled) ──────────────────────────────────────────────

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
rots_scan      = [R_gal2top] * n_scan
body_rots_scan = [Beam.top2body(az, alt)
                  for az in az_rad for alt in alt_rad]

TX_DIR   = np.array([0.0, 0.0, -1.0])
TX_FREQS = freqs_hz[::4]
TX_POW   = 1e6 * np.ones_like(TX_FREQS)

fwd = ForwardModel(observer, beam, sky,
                   transmitters=[(TX_DIR, TX_FREQS, TX_POW)])

# Warm up JIT with one step so profiling measures steady-state
geom_1 = fwd.precompute_geometry(rots=[R_gal2top], body_rots=[body_rots_scan[0]])
_ = np.array(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom_1))
# Second call to ensure XLA has compiled for 1-step
_ = np.array(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom_1))

print(f"  sky npix={npix_sky}  beam npix={npix_beam}  scan={n_scan}  freqs={N_FREQ}")
print()


# ── 1. cProfile — precompute_geometry ─────────────────────────────────────────

print("=" * 70)
print("1. cProfile — precompute_geometry (1296 pointings, with body_rots + 1 TX)")
print("=" * 70)

pr = cProfile.Profile()
pr.enable()
geom = fwd.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
pr.disable()

buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
ps.print_stats(25)
print(buf.getvalue())


# ── 2. line_profiler — precompute_geometry ────────────────────────────────────

print("=" * 70)
print("2. line_profiler — precompute_geometry")
print("=" * 70)

if LineProfiler is None:
    print("  line_profiler is not installed; skipping this section.\n")
else:
    from eigsep_sim.simulate import ForwardModel as _FwdCls

    lp = LineProfiler()
    lp.add_function(_FwdCls.precompute_geometry)
    lp_wrapper = lp(_FwdCls.precompute_geometry)

    # Call via wrapper (line_profiler instruments the source function directly)
    lp.enable_by_count()
    geom = fwd.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
    lp.disable_by_count()

    buf2 = io.StringIO()
    lp.print_stats(stream=buf2)
    print(buf2.getvalue())


# ── 3. JAX profiler — simulate ────────────────────────────────────────────────

print("=" * 70)
print("3. JAX profiler — simulate (1296 steps, 1 TX)")
print("=" * 70)

TRACE_DIR = "benchmarks/jax_profile"

# Pre-compute geom so profiling covers only the JAX kernel
geom_full = fwd.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)

# One untraced call to let XLA re-specialize for 1296-step input shape
_ = np.array(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom_full))

print(f"  Capturing XLA trace → {TRACE_DIR}/")
with jax.profiler.trace(TRACE_DIR, create_perfetto_trace=True):
    result = np.array(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom_full))
    result.block_until_ready() if hasattr(result, 'block_until_ready') else None

print(f"  Output shape: {result.shape}  (ntimes, n_dipoles, nfreq)")
print(f"  Trace written to {TRACE_DIR}/")
print(f"  Open in chrome://tracing or https://ui.perfetto.dev")
print()


# ── 4. Memory — precompute_geometry ──────────────────────────────────────────

print("=" * 70)
print("4. Peak memory — precompute_geometry (1296 pointings)")
print("=" * 70)

tracemalloc.start()
geom_mem = fwd.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Compute sizes of the returned JAX arrays
def mb(arr):
    return arr.nbytes / 1024**2

print(f"  Python heap peak during precompute_geometry: {peak / 1024**2:.1f} MB")
print()
print("  Geom array sizes:")
geom_keys = [
    'rots_jax',
    'body_rots_jax',
    'terrain_masks_jax',
    'terrain_emissions_jax',
    'default_emission_masks_jax',
    'unresolved_emission_jax',
    'unresolved_default_emission_jax',
    'crds_gal_jax',
    'tx_crds_jax',
]
total = 0.0
for key in geom_keys:
    arr = geom_mem[key]
    size_mb = mb(arr)
    total += size_mb
    print(f"    {key:<17s} {str(tuple(arr.shape)):30s}  {size_mb:.1f} MB")
if 'sky_indices_jax' in geom_mem:
    arr = geom_mem['sky_indices_jax']
    size_mb = mb(arr)
    total += size_mb
    print(f"    {'sky_indices_jax':<17s} {str(tuple(arr.shape)):30s}  {size_mb:.1f} MB")
print(f"    {'TOTAL':30s}  {total:.1f} MB")
print()
print("Done.")
