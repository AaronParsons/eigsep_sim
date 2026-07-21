"""Joint sky+beam recovery for the canyon by alternating exact solves (ALS).

Each conditional problem is linear/quadratic: sky given beam (exact Cholesky),
beam given sky (exact Cholesky, with the TX term anchoring scale + meridian).
Alternating these is block coordinate descent on the bilinear problem; the TX
term removes the sky x beam scale degeneracy so it converges to the truth.
"""

import time

import numpy as np

from canyon_tx_lib import (
    build_canyon, make_ops, sky_solve, beam_solve, chi2,
    map_err_sky, map_err_beam,
)


def make_data(cfg, ops, snr_frac=1e-3, seed=0):
    ref = np.asarray(ops["simulate"](cfg["sky_coeffs"], cfg["beam_coeffs"]))
    sigma = snr_frac * np.sqrt(np.mean(ref**2))
    rng = np.random.default_rng(seed)
    data = ref + rng.normal(scale=sigma, size=ref.shape)
    inv = np.full_like(data, 1.0 / sigma**2)
    return data, inv, sigma


def solve_als(cfg, ops, data, inv, n_rounds=6, perturb=0.1, seed=1,
              use_beam=True, lam_sky=0.0, lam_beam=0.0, beam_nom=None,
              verbose=True):
    """Regularized alternating exact solves from a perturbed start.

    lam_sky / lam_beam are absolute Tikhonov strengths (Gram-eigenvalue units);
    the beam ridge pulls toward beam_nom (the nominal/measured beam). The TX
    term anchors absolute beam scale, so there is no residual scale gauge.
    """
    prng = np.random.default_rng(seed)
    sc0, bc0 = cfg["sky_coeffs"], cfg["beam_coeffs"]
    if beam_nom is None:
        beam_nom = bc0  # nominal = truth here (recovery test perturbs from it)
    beam = bc0 * prng.uniform(1 - perturb, 1 + perturb, size=bc0.shape).astype(bc0.dtype)
    sky = sc0 * prng.uniform(1 - perturb, 1 + perturb, size=sc0.shape).astype(sc0.dtype)
    # Exclude the TX channels from the sky estimate: the bright tone there
    # otherwise contaminates the sky (it is a beam/TX term, not sky). The beam
    # step keeps all channels (TX folded into its operator).
    inv_sky = np.asarray(inv).copy()
    inv_sky[:, np.asarray(cfg["tx_mask"])] = 0.0
    hist = []
    for r in range(n_rounds):
        t0 = time.perf_counter()
        sky = sky_solve(ops, cfg, data, inv_sky, beam, lam=lam_sky)
        if use_beam:
            beam = beam_solve(ops, cfg, data, inv, sky, lam=lam_beam,
                              beam_nom=beam_nom)
        dt = time.perf_counter() - t0
        c = chi2(ops, data, inv, sky, beam)
        es = map_err_sky(cfg, sky)
        eb = map_err_beam(cfg, beam)
        hist.append(dict(round=r, chi2=c, sky_err=es, beam_err=eb, dt=dt))
        if verbose:
            print(f"  round {r}: chi2 {c:8.3f}  sky_err {es:.4f}  "
                  f"beam_err {eb:.4f}  dt {dt:.1f}s", flush=True)
    return sky, beam, hist


def main():
    cfg = build_canyon(nside_sky=8, nside_beam=8, n_freq=24,
                       n_times=8, n_az=12, n_alt=8)
    ops = make_ops(cfg)
    print(f"orientations={cfg['n_orient']} visible_pix={int(cfg['sky_mask'].sum())} "
          f"n_freq={cfg['n_freq']} TXchan={list(np.where(cfg['tx_mask'])[0])}")
    data, inv, sigma = make_data(cfg, ops, snr_frac=1e-3)
    print(f"sigma={sigma:.3g}  chi2 at truth={chi2(ops, data, inv, cfg['sky_coeffs'], cfg['beam_coeffs']):.3f}")

    print("\nALS joint (sky+beam, TX on):")
    solve_als(cfg, ops, data, inv, n_rounds=6, use_beam=True)

    print("\nSky-only (beam frozen at perturbed init):")
    solve_als(cfg, ops, data, inv, n_rounds=3, use_beam=False)


if __name__ == "__main__":
    main()
