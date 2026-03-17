"""
Benchmark: beam-interpolation strategies in eigsep_sim.

Measures wall-time for several implementations of the beam-weighted sky
integral (the core of _beam_sum) and compares them with the baseline
lax.scan implementation used in Simulator.sim().

Methods benchmarked
-------------------
1. baseline       – current _beam_sum with jax.lax.scan
2. vmap           – jax.vmap over orientations
3. precomp_den    – precompute denominator (rotation-invariant for full sky)
4. low_nside      – use nside=16 beam (coarser, but much smaller map)
5. sh_rotate_alm  – SH decomposition + healpy.rotate_alm per orientation
6. sh_fft_spin    – SH + FFT for pure z-spin sweep (all orientations at once)

Usage
-----
    python benchmarks/beam_interp.py

JAX device and XLA flags can be set before running, e.g.:
    JAX_PLATFORM_NAME=cpu python benchmarks/beam_interp.py
"""

from __future__ import annotations

import time
from functools import partial

import healpy
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

# Must be imported before JIT-compiled helpers so healjax is initialised.
from eigsep_sim.healpix import float_dtype, interpolate_map
from eigsep_sim.beam import short_dipole_beam

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

NSIDE_SKY = 64
NSIDE_BEAM = 64
NSIDE_BEAM_LO = 16
NFREQ = 64
N_ORIENT = 32       # number of beam orientations
N_REPEAT = 5        # timing repeats (min is reported)
LMAX_SH = 2 * NSIDE_SKY  # standard band-limit for SH decomposition

freqs = np.linspace(50e6, 200e6, NFREQ)
npix_sky = healpy.nside2npix(NSIDE_SKY)
npix_beam = healpy.nside2npix(NSIDE_BEAM)

rng = np.random.default_rng(42)

# Synthetic sky and beam maps (float32 like the real sim)
sky_np = rng.uniform(1.0, 100.0, (npix_sky, NFREQ)).astype(np.float32)
beam_np = short_dipole_beam(freqs, NSIDE_BEAM, dipole_axis=(0, 0, 1))
beam_lo_np = short_dipole_beam(freqs, NSIDE_BEAM_LO, dipole_axis=(0, 0, 1))

# Topocentric pixel unit vectors (3, npix_sky)
crds_top_np = np.stack(
    healpy.pix2vec(NSIDE_SKY, np.arange(npix_sky)), axis=0
).astype(np.float32)

# Random rotation matrices, shape (N_ORIENT, 3, 3)
rot_ms_np = Rotation.random(N_ORIENT, random_state=rng).as_matrix().astype(np.float32)

# JAX arrays
sky_jax = jnp.asarray(sky_np, dtype=float_dtype)
beam_jax = jnp.asarray(beam_np, dtype=float_dtype)
beam_lo_jax = jnp.asarray(beam_lo_np, dtype=float_dtype)
crds_top_jax = jnp.asarray(crds_top_np, dtype=float_dtype)
rot_ms_jax = jnp.asarray(rot_ms_np, dtype=float_dtype)


# ---------------------------------------------------------------------------
# Method 1 – baseline: lax.scan (mirrors _beam_sum in sim.py)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def beam_sum_scan(beam_nside, beam_map, sky, crds, rot_ms):
    """Return (num, den) using lax.scan over orientations."""
    def body(_, R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ crds))
        return None, (jnp.sum(wgt * sky, axis=0), jnp.sum(wgt, axis=0))
    _, (num, den) = jax.lax.scan(body, None, rot_ms)
    return num, den


# ---------------------------------------------------------------------------
# Method 2 – vmap over orientations
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def beam_sum_vmap(beam_nside, beam_map, sky, crds, rot_ms):
    """Return (num, den) using jax.vmap over orientations."""
    def single(R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ crds))
        return jnp.sum(wgt * sky, axis=0), jnp.sum(wgt, axis=0)
    return jax.vmap(single)(rot_ms)


# ---------------------------------------------------------------------------
# Method 3 – precomputed denominator
#
# The denominator sum_pix B(R @ crds_pix) is the integral of the beam over
# the sky pixel centres.  For a full-sphere sky at the same nside as the
# beam this equals sum(beam_map, axis=0) — independent of rotation R.
# Precomputing it saves one jnp.sum per orientation inside the scan loop.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def beam_sum_precomp_den(beam_nside, beam_map, sky, crds, rot_ms):
    """Return (num, den) where den is precomputed (rotation-invariant)."""
    # Precompute once — shape (nfreq,)
    den_fixed = jnp.sum(beam_map, axis=0) * (npix_sky / npix_beam)

    def body(_, R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ crds))
        return None, jnp.sum(wgt * sky, axis=0)

    _, num = jax.lax.scan(body, None, rot_ms)
    den = jnp.broadcast_to(den_fixed[None, :], num.shape)
    return num, den


# ---------------------------------------------------------------------------
# Method 4 – low-nside beam (nside=16 instead of 64)
# ---------------------------------------------------------------------------
# Benchmarked via beam_sum_scan with NSIDE_BEAM_LO and beam_lo_jax.


# ---------------------------------------------------------------------------
# Method 5 – SH inner product with healpy.rotate_alm
#
# For each orientation R the sky alm are rotated in place using Wigner
# D-matrices, then the inner product with the (static) beam alm gives T_ant.
# Cost: O(lmax^3) per orientation for the Wigner D recursion.
# ---------------------------------------------------------------------------

def _sh_inner_product(alm_a: np.ndarray, alm_b: np.ndarray, lmax: int) -> float:
    """
    <A, B> = sum_{l,m} a_lm conj(b_lm) summed over ALL m (including m < 0).

    Healpy stores only m >= 0; for real maps a_{l,-m} = (-1)^m conj(a_{lm}),
    so the contribution of m > 0 pairs is 2 Re(a_lm conj(b_lm)).
    """
    _, m_arr = healpy.Alm.getlm(lmax)
    wgt = np.where(m_arr == 0, 1.0, 2.0)
    return float(np.sum(wgt * np.real(alm_a * np.conj(alm_b))))


def bench_sh_rotate_alm(
    beam_alm: np.ndarray,
    sky_alm: np.ndarray,
    lmax: int,
    rot_ms: np.ndarray,
    beam_solid_angle: float,
) -> np.ndarray:
    """
    Single-frequency T_ant via SH inner product for each rotation matrix.

    Parameters
    ----------
    beam_alm, sky_alm : (n_alm,) complex128
        SH coefficients of beam and sky (single frequency).
    lmax : int
    rot_ms : (N, 3, 3) float
        Rotation matrices (topocentric → beam frame).
    beam_solid_angle : float
        Denominator: integral(B d_Omega) = sum_pix B * (4pi/npix).

    Returns
    -------
    T_ant : (N,) float
    """
    T_ant = np.empty(len(rot_ms))
    for i, R in enumerate(rot_ms):
        # Rotate sky into the beam frame using R (i.e. R^{-1} applied to sky)
        # healpy.rotate_alm modifies in-place and returns None
        sky_alm_rot = sky_alm.copy()
        healpy.rotate_alm(sky_alm_rot, matrix=R)
        T_ant[i] = _sh_inner_product(beam_alm, sky_alm_rot, lmax) / beam_solid_angle
    return T_ant


# ---------------------------------------------------------------------------
# Method 6 – SH + FFT for a pure z-spin sweep
#
# For rotations R_z(phi) about the z-axis, the Wigner D-matrix is diagonal:
#   D^l_{mm'}(R_z(phi)) = delta_{mm'} e^{-im phi}
#
# Therefore the beam-weighted integral becomes:
#   T_ant(phi) = (1/den) * sum_{l,m} b_lm conj(t_lm) e^{-im phi}
#             = (1/den) * DFT[C][k]   at  phi_k = 2pi k / N
# where C_m = sum_l b_lm conj(t_lm).
#
# This gives ALL N_phi orientations via a single FFT — O(N log N) vs O(N npix)
# for the pixel-domain approach.  Precomputing C_m (once) costs O(n_alm).
# ---------------------------------------------------------------------------

def _compute_C_m(beam_alm: np.ndarray, sky_alm: np.ndarray, lmax: int) -> np.ndarray:
    """
    Compute C_m = sum_{l>=m} b_lm conj(t_lm) for m = 0 .. lmax.

    Parameters
    ----------
    beam_alm, sky_alm : (n_alm,) complex128

    Returns
    -------
    C_pos : (lmax+1,) complex128  — modes m = 0 .. lmax
    """
    _, m_arr = healpy.Alm.getlm(lmax)
    product = beam_alm * np.conj(sky_alm)
    C_pos = np.zeros(lmax + 1, dtype=np.complex128)
    np.add.at(C_pos, m_arr, product)
    return C_pos


def sh_fft_spin_sweep(
    C_pos: np.ndarray,
    N_phi: int,
    beam_solid_angle: float,
) -> np.ndarray:
    """
    Evaluate T_ant(phi_k) for k = 0 .. N_phi-1 (phi_k = 2pi k / N_phi)
    via a single FFT, given the precomputed coupling coefficients C_pos.

    Parameters
    ----------
    C_pos : (lmax+1,) complex128
        Precomputed coupling modes C_m = sum_l b_lm conj(t_lm) for m >= 0.
    N_phi : int
        Number of equally-spaced spin angles.
    beam_solid_angle : float
        Normalisation denominator = integral(B d_Omega).

    Returns
    -------
    T_ant : (N_phi,) float64
    """
    mmax = len(C_pos) - 1

    # Build DFT spectrum: C_{-m} = conj(C_m) for m > 0
    spectrum = np.zeros(N_phi, dtype=np.complex128)
    spectrum[0] = C_pos[0]
    # Positive-m modes at indices 1..mmax (or up to N_phi//2)
    m_hi = min(mmax, N_phi // 2)
    spectrum[1:m_hi + 1] = C_pos[1:m_hi + 1]
    # Negative-m modes at indices N_phi-m (wrap around)
    spectrum[N_phi - m_hi:N_phi] = np.conj(C_pos[m_hi:0:-1])

    # T_ant(phi_k) = (1/den) sum_m C_m e^{-im phi_k}
    #              = (1/den) * N * IFFT[spectrum]
    T_ant = np.fft.ifft(spectrum).real * N_phi / beam_solid_angle
    return T_ant


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _block(result):
    """Ensure JAX computations have completed."""
    if isinstance(result, (tuple, list)):
        for r in result:
            if hasattr(r, 'block_until_ready'):
                r.block_until_ready()
    elif hasattr(result, 'block_until_ready'):
        result.block_until_ready()


def timeit(fn, label: str, n: int = N_REPEAT):
    """
    Run *fn* n times and report min/mean wall-clock time.

    Returns
    -------
    result : last result from fn()
    t_min  : minimum wall time [seconds]
    """
    # Warm-up (triggers JIT if applicable)
    result = fn()
    _block(result)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        _block(result)
        times.append(time.perf_counter() - t0)

    t_min = min(times)
    t_mean = float(np.mean(times))
    print(f"  {label:<40s}  min={t_min * 1e3:8.2f} ms  mean={t_mean * 1e3:8.2f} ms")
    return result, t_min


# ---------------------------------------------------------------------------
# Precompute SH coefficients (done once, outside timing loops)
# ---------------------------------------------------------------------------

def _precompute_sh(lmax: int, freq_idx: int = 0):
    """Return (beam_alm, sky_alm, C_pos, beam_solid_angle) for one frequency."""
    bm1 = beam_np[:, freq_idx].astype(np.float64)
    sk1 = sky_np[:, freq_idx].astype(np.float64)
    beam_alm = healpy.map2alm(bm1, lmax=lmax, use_pixel_weights=False)
    sky_alm = healpy.map2alm(sk1, lmax=lmax, use_pixel_weights=False)
    C_pos = _compute_C_m(beam_alm, sky_alm, lmax)
    # Beam solid angle = integral B d_Omega ≈ sum_pix B * (4pi/npix)
    beam_solid_angle = float(np.sum(bm1) * (4.0 * np.pi / npix_beam))
    return beam_alm, sky_alm, C_pos, beam_solid_angle


# ---------------------------------------------------------------------------
# Sanity check: SH FFT result should match pixel-domain scan (single freq)
# ---------------------------------------------------------------------------

def _sanity_check(lmax: int = LMAX_SH, freq_idx: int = 0):
    """Verify SH-FFT spin sweep against pixel-domain scan for a z-rotation."""
    # Generate N_ORIENT z-rotations
    phis = np.linspace(0, 2 * np.pi, N_ORIENT, endpoint=False)
    rot_z = np.stack([
        np.array([[np.cos(p), -np.sin(p), 0],
                  [np.sin(p),  np.cos(p), 0],
                  [0,          0,         1]], dtype=np.float32)
        for p in phis
    ])
    rot_z_jax = jnp.asarray(rot_z, dtype=float_dtype)

    # Pixel-domain result (single frequency)
    sky1_jax = sky_jax[:, freq_idx:freq_idx + 1]
    num, den = beam_sum_scan(NSIDE_BEAM, beam_jax[:, freq_idx:freq_idx + 1],
                             sky1_jax, crds_top_jax, rot_z_jax)
    T_pix = np.asarray(num / den)[:, 0]  # (N_ORIENT,)

    # SH-FFT result
    beam_alm, sky_alm, C_pos, beam_sol = _precompute_sh(lmax, freq_idx)
    T_fft = sh_fft_spin_sweep(C_pos, N_ORIENT, beam_sol)

    rms_err = float(np.sqrt(np.mean((T_pix - T_fft) ** 2)))
    rel_err = rms_err / float(np.mean(np.abs(T_pix)))
    print(f"\n  [sanity] SH-FFT vs pixel-domain (lmax={lmax}): "
          f"RMS={rms_err:.4f} K  relative={rel_err:.4f}")
    if rel_err > 0.05:
        print("  WARNING: relative error > 5% — consider increasing lmax.")
    else:
        print("  OK: SH-FFT agrees with pixel-domain within 5%.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print(f"  eigsep_sim beam-interpolation benchmark")
    print(f"{'=' * 60}")
    print(f"  JAX devices   : {jax.devices()}")
    print(f"  nside_sky     : {NSIDE_SKY}  ({npix_sky} pixels)")
    print(f"  nside_beam    : {NSIDE_BEAM}  ({npix_beam} pixels)")
    print(f"  nside_beam_lo : {NSIDE_BEAM_LO}  ({healpy.nside2npix(NSIDE_BEAM_LO)} pixels)")
    print(f"  nfreq         : {NFREQ}")
    print(f"  n_orient      : {N_ORIENT}")
    print(f"  lmax_sh       : {LMAX_SH}")
    print(f"  repeats       : {N_REPEAT}")
    print()

    # ------------------------------------------------------------------
    # Pre-warm JAX JIT for all pixel-domain methods
    # ------------------------------------------------------------------
    print("  JIT warm-up … ", end='', flush=True)
    for _ in range(2):
        _block(beam_sum_scan(NSIDE_BEAM, beam_jax, sky_jax, crds_top_jax, rot_ms_jax))
        _block(beam_sum_vmap(NSIDE_BEAM, beam_jax, sky_jax, crds_top_jax, rot_ms_jax))
        _block(beam_sum_precomp_den(NSIDE_BEAM, beam_jax, sky_jax, crds_top_jax, rot_ms_jax))
        _block(beam_sum_scan(NSIDE_BEAM_LO, beam_lo_jax, sky_jax, crds_top_jax, rot_ms_jax))
    print("done\n")

    # ------------------------------------------------------------------
    # Pixel-domain methods (all frequencies)
    # ------------------------------------------------------------------
    print("  Pixel-domain methods  (nfreq=64, n_orient=32)")
    print("  " + "-" * 56)

    _, t_scan = timeit(
        lambda: beam_sum_scan(
            NSIDE_BEAM, beam_jax, sky_jax, crds_top_jax, rot_ms_jax),
        label="1. baseline  (lax.scan, nside=64)",
    )
    _, t_vmap = timeit(
        lambda: beam_sum_vmap(
            NSIDE_BEAM, beam_jax, sky_jax, crds_top_jax, rot_ms_jax),
        label="2. vmap      (nside=64)",
    )
    _, t_pden = timeit(
        lambda: beam_sum_precomp_den(
            NSIDE_BEAM, beam_jax, sky_jax, crds_top_jax, rot_ms_jax),
        label="3. precomp-den (lax.scan, nside=64)",
    )
    _, t_lo = timeit(
        lambda: beam_sum_scan(
            NSIDE_BEAM_LO, beam_lo_jax, sky_jax, crds_top_jax, rot_ms_jax),
        label="4. low-nside (lax.scan, nside=16)",
    )

    # ------------------------------------------------------------------
    # SH methods (single frequency — SH decomposition is 1-D)
    # ------------------------------------------------------------------
    print()
    print("  SH methods  (single frequency, n_orient=32)")
    print("  " + "-" * 56)
    print("  Precomputing SH coefficients … ", end='', flush=True)
    t_sh_pre0 = time.perf_counter()
    beam_alm, sky_alm, C_pos, beam_sol = _precompute_sh(LMAX_SH)
    t_sh_pre = time.perf_counter() - t_sh_pre0
    print(f"done  ({t_sh_pre * 1e3:.1f} ms)")

    _, t_sh_rot = timeit(
        lambda: bench_sh_rotate_alm(
            beam_alm, sky_alm, LMAX_SH, rot_ms_np, beam_sol),
        label="5. SH+rotate_alm (per orientation)",
    )
    _, t_fft = timeit(
        lambda: sh_fft_spin_sweep(C_pos, N_ORIENT, beam_sol),
        label="6. SH+FFT        (spin sweep, all at once)",
    )

    # For a fair comparison with the FFT approach, also time C_m precomputation
    _, t_cpos = timeit(
        lambda: _compute_C_m(beam_alm, sky_alm, LMAX_SH),
        label="   └─ C_m precompute (amortised over many t)",
    )

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    _sanity_check(LMAX_SH)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Speedup vs baseline (min time)")
    print(f"{'=' * 60}")
    results = [
        ("vmap (nside=64)",         t_scan / t_vmap),
        ("precomp-den (nside=64)",  t_scan / t_pden),
        ("low-nside (nside=16)",    t_scan / t_lo),
        ("SH+rotate_alm (1 freq)",  t_scan / t_sh_rot),
        ("SH+FFT spin (1 freq)",    t_scan / t_fft),
        ("SH+FFT incl C_m precomp", t_scan / (t_fft + t_cpos)),
    ]
    for label, speedup in results:
        bar = '█' * max(1, min(50, int(speedup * 2)))
        print(f"  {label:<35s}  {speedup:9.2f}x  {bar}")

    print()
    print("Notes:")
    print("  • SH methods are single-frequency; scale nfreq manually.")
    print("  • SH+rotate_alm costs O(lmax^3) per orientation.")
    print("  • SH+FFT spin sweep costs O(n_alm) precompute + O(N log N) per freq.")
    print("  • For a full nfreq=64 spin sweep, multiply SH+FFT time by 64.")
    print()
