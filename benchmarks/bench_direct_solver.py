"""Full-fit comparison: exact direct sky solve vs truncated-CG sky solve.

The sky conditional Hessian is ill-conditioned (kappa ~ 1e8 from sky
coverage), so truncated CG never reaches the conditional minimum and
fast-cg leans on many outer iterations.  This builds the per-frequency
Gram Hessian and Cholesky-solves the sky block exactly each outer
iteration, reusing the existing beam fast path, and compares loss vs
cumulative wall time against solver='fast-cg' at n_freq=20.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from eigsep_sim.const import DTYPE_R_NPY

from bench_fastcg_profile import build


def exact_sky_solve(cal, ops, params):
    """Return params with the exact conditional sky minimizer."""
    fwd = cal.fwd
    A = fwd._sky_basis_A_jax  # (F, M)
    beam_j = jnp.asarray(params["beam_coeffs"], dtype=jnp.float32)
    sky_j = jnp.asarray(params["sky_coeffs"], dtype=jnp.float32)
    P, M = int(sky_j.shape[0]), int(sky_j.shape[1])

    g_op = ops["build_g"](beam_j)  # (D, F, T, P)
    const = fwd.simulate(jnp.zeros_like(sky_j), beam_j, geom=cal._geom)
    obs = cal._matched_observations(const.shape)
    data_eff = jnp.transpose(
        jnp.asarray(obs["data"]) - const, (1, 2, 0)
    )  # (D, F, T)
    inv_var = jnp.transpose(jnp.asarray(obs["inv_noise_var"]), (1, 2, 0))
    n_data = data_eff.size
    nf = int(A.shape[0])

    resid = jnp.einsum("dftp,fp->dft", g_op, (sky_j @ A.T).T) - data_eff
    b = -(2.0 / n_data) * (
        jnp.einsum("dftp,dft->fp", g_op, inv_var * resid).T @ A
    )

    G = jnp.transpose(g_op, (1, 0, 2, 3)).reshape(nf, -1, P)  # (F, DT, P)
    w = jnp.transpose(inv_var, (1, 0, 2)).reshape(nf, -1)
    Bf = jnp.einsum("fnp,fnq->fpq", G * w[..., None], G)  # (F, P, P)
    H = (2.0 / n_data) * jnp.einsum("fm,fn,fpq->pmqn", A, A, Bf)
    H = np.asarray(H, dtype=np.float64).reshape(P * M, P * M)
    H = 0.5 * (H + H.T)
    if cal._lam_sky > 0:
        H[np.diag_indices_from(H)] += 2.0 * cal._lam_sky / (P * M)
    H[np.diag_indices_from(H)] += 1e-8 * np.trace(H) / (P * M)

    L = np.linalg.cholesky(H)
    delta = jax.scipy.linalg.cho_solve(
        (jnp.asarray(L), True), b.ravel().astype(jnp.float64)
    ).reshape(P, M)
    out = dict(params)
    out["sky_coeffs"] = np.asarray(
        sky_j + jnp.asarray(delta, dtype=jnp.float32), dtype=DTYPE_R_NPY
    )
    return out


def fit_direct(cal, ops, params, max_iter=8):
    params = {k: np.asarray(v).copy() for k, v in params.items()}
    losses, times = [], []
    t_start = time.perf_counter()
    for it in range(max_iter):
        params = exact_sky_solve(cal, ops, params)
        params = cal.beam_cg_step(params, n_cg=10, cg_tol=1e-3)
        params = cal._project_scale_degeneracy(params)
        losses.append(float(cal._loss(params)))
        times.append(time.perf_counter() - t_start)
        print(
            f"  direct iter {it}: loss {losses[-1]:.4e}  "
            f"t {times[-1]:6.1f}s",
            flush=True,
        )
    return params, losses, times


def main():
    nf = 20
    cal, ini, geom = build(nf, 5, 5)
    ops = cal._ensure_linear_ops()

    print("=== exact direct sky solve + fast beam ===", flush=True)
    p0 = {k: v.copy() for k, v in ini.items()}
    # warm up jit (build_g, beam path)
    exact_sky_solve(cal, ops, p0)
    cal.beam_cg_step(p0, n_cg=10, cg_tol=1e-3)
    _, losses_d, times_d = fit_direct(cal, ops, p0, max_iter=8)

    print("\n=== current fast-cg ===", flush=True)
    t0 = time.perf_counter()
    res = cal.fit(
        params={k: v.copy() for k, v in ini.items()},
        geom=geom,
        max_iter=10,
        tol=0.0,
        verbose=False,
        solver="fast-cg",
    )
    tt = np.cumsum([t["wall_time"] for t in res["telemetry"]])
    for it, (loss, t) in enumerate(zip(res["losses"], tt)):
        print(f"  fast-cg iter {it}: loss {loss:.4e}  t {t:6.1f}s")
    print(f"\nfast-cg total {time.perf_counter() - t0:.1f}s")
    print(f"direct  reached {losses_d[-1]:.4e} at {times_d[-1]:.1f}s")
    print(f"fast-cg reached {res['losses'][-1]:.4e} at {tt[-1]:.1f}s")


if __name__ == "__main__":
    main()
