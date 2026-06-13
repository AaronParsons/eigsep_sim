"""Profile one fast-cg iteration at the user's slow configuration.

Times the primitive operations (simulate, loss, grad, HVP) and the
composite steps (sky_step, beam_cg_step) to locate where the ~190 s/iter
goes at n_freq=2, and tests whether clipping the spectral basis ranks to
n_freq (removing rank deficiency) restores fast CG convergence.
"""

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import astropy.units as u
from astropy.time import Time

from eigsep_sim import (
    EarthSurface,
    Beam,
    Sky,
    NullTerrain,
    ForwardModel,
    Calibrator,
    DTYPE_R_NPY,
)

NSIDE = 8
N_TIMES = 36
N_AZ, N_ALT = 12, 10
N_ORIENT = N_AZ * N_ALT


def build(n_freq, n_modes, K):
    freqs_hz = np.linspace(55e6, 150e6, n_freq)
    obs = EarthSurface(lat=39.2, lon=-113.4)
    beam = Beam.from_dipole(
        nside=NSIDE,
        freqs_hz=freqs_hz,
        arm_lengths_m=2.0,
        u_body=np.eye(3, dtype=DTYPE_R_NPY)[:1],
        K=K,
    )
    sky = Sky.from_gsm(NSIDE, freqs_hz, n_modes=n_modes, include_flat=True)
    gsm = sky.init_coeffs()
    fwd = ForwardModel(obs, beam, sky, terrain=NullTerrain())
    base_times = (
        Time("2025-01-01")
        + np.linspace(0, 86400, N_TIMES, endpoint=False) * u.s
    )
    base_rots = obs.rot_gal2top_stack(base_times)
    rots = np.repeat(base_rots, N_ORIENT, axis=0)
    az = np.linspace(0, 2 * np.pi, N_AZ, endpoint=False)
    alt = np.linspace(0, np.pi / 2, N_ALT)
    orient = np.stack([Beam.top2body(a, h) for h in alt for a in az])
    body_rots = np.tile(orient, (N_TIMES, 1, 1))
    geom = fwd.precompute_geometry(rots=rots, body_rots=body_rots)
    truth = np.asarray(fwd.simulate(gsm, beam.coeffs, geom=geom))
    sigma = np.abs(truth).mean() * 1e-4
    rng = np.random.default_rng(0)
    data = (truth + rng.normal(scale=sigma, size=truth.shape)).reshape(
        -1, n_freq
    )
    inv = np.full_like(data, 1.0 / sigma**2)
    cal = Calibrator(
        fwd, data, inv_noise_var=inv, lam_beam=0.01, lam_beam_harmonic=0.0
    )
    prng = np.random.default_rng(1)
    ini = {
        "sky_coeffs": gsm
        * prng.uniform(0.9, 1.1, size=gsm.shape).astype(gsm.dtype),
        "beam_coeffs": beam.coeffs
        * prng.uniform(0.9, 1.1, size=beam.coeffs.shape).astype(
            beam.coeffs.dtype
        ),
    }
    cal._beam_nom = ini["beam_coeffs"].copy()
    cal._resolve_geom(geom=geom)
    return cal, ini, geom


def t(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    tic = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - tic) / n


def profile(label, n_freq, n_modes, K):
    print(f"\n=== {label}: n_freq={n_freq} n_modes={n_modes} K={K} ===")
    cal, ini, geom = build(n_freq, n_modes, K)
    fwd = cal.fwd
    sky_j = jnp.asarray(ini["sky_coeffs"])
    beam_j = jnp.asarray(ini["beam_coeffs"])

    def sim():
        return jax.block_until_ready(fwd.simulate(sky_j, beam_j, geom=geom))

    print(f"simulate            : {t(sim):8.3f}s")

    def loss():
        return float(cal._loss(ini))

    print(f"loss                : {t(loss):8.3f}s")

    def loss_sky(s):
        return cal._loss({"sky_coeffs": s, "beam_coeffs": beam_j})

    grad_fn = jax.grad(loss_sky)

    def grad():
        return jax.block_until_ready(grad_fn(sky_j))

    print(f"grad (sky)          : {t(grad):8.3f}s")
    g0 = grad_fn(sky_j)

    def hvp():
        return jax.block_until_ready(jax.jvp(grad_fn, (sky_j,), (g0,))[1])

    print(f"hvp (sky)           : {t(hvp):8.3f}s")

    tic = time.perf_counter()
    p1 = cal.sky_step(ini)
    print(f"sky_step (n_cg=50)  : {time.perf_counter() - tic:8.3f}s")
    tic = time.perf_counter()
    p2 = cal.beam_cg_step(p1, n_cg=10, cg_tol=1e-3)
    print(f"beam_cg_step (10)   : {time.perf_counter() - tic:8.3f}s")
    loss_after = float(cal._loss(p2))
    print(
        f"loss before/after   : {float(cal._loss(ini)):.3e} -> "
        f"{loss_after:.3e}"
    )


if __name__ == "__main__":
    n_freq = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    # User config: spectral ranks exceed n_freq (rank-deficient).
    profile("user config", n_freq, n_modes=5, K=5)
    # Clipped ranks: well-posed conditional problems.
    profile(
        "clipped ranks", n_freq, n_modes=min(5, n_freq - 1), K=min(5, n_freq)
    )
