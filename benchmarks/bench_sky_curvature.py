"""Characterize the conditional sky curvature and test direct/preconditioned
alternatives to the truncated-CG conditional solve in fast-cg.

The forward model decouples across frequency in map space, so the sky
conditional Hessian is H = (2/N) sum_f (a_f a_f^T) (x) B_f, where
B_f[p,p'] = sum_{t,d} w G[d,f,t,p] G[d,f,t,p'] is the per-frequency pixel
Gram and a_f = A_sky[f, :].  This script, at the notebook scale
(n_freq=20), compares at a fixed point:

  * plain CG (current fast-cg conditional solve),
  * an exact direct solve (form H via the B_f, Cholesky),
  * a pixel-block (M x M) Jacobi-preconditioned CG.

It reports wall time, loss reduction, and error vs the exact Newton step.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from bench_fastcg_profile import build


def main():
    nf = 20
    cal, ini, geom = build(nf, 5, 5)
    fwd = cal.fwd
    ops = cal._ensure_linear_ops()

    A_np = np.asarray(fwd._sky_basis_A_jax)  # (F, M)
    sv = np.linalg.svd(A_np, compute_uv=False)
    print(
        f"A_sky {A_np.shape}  singular values "
        f"{np.array2string(sv, precision=3)}  cond {sv[0] / sv[-1]:.2e}"
    )

    beam_j = jnp.asarray(ini["beam_coeffs"])
    sky_j = jnp.asarray(ini["sky_coeffs"])
    A = jnp.asarray(A_np)
    P, M = int(sky_j.shape[0]), int(sky_j.shape[1])

    g_op = ops["build_g"](beam_j)  # (D, F, T, P)
    const = fwd.simulate(jnp.zeros_like(sky_j), beam_j, geom=geom)
    obs = cal._matched_observations(const.shape)
    data_eff = jnp.transpose(
        jnp.asarray(obs["data"]) - const, (1, 2, 0)
    )  # (D, F, T)
    inv_var = jnp.transpose(jnp.asarray(obs["inv_noise_var"]), (1, 2, 0))
    n_data = data_eff.size

    def fwd_op(v):  # (P, M) -> (D, F, T)
        return jnp.einsum("dftp,fp->dft", g_op, (v @ A.T).T)

    def adj_op(u):  # (D, F, T) -> (P, M)
        return jnp.einsum("dftp,dft->fp", g_op, u).T @ A

    resid = fwd_op(sky_j) - data_eff
    b = -(2.0 / n_data) * adj_op(inv_var * resid)  # (P, M)

    def loss_of(delta):
        return float(
            cal._loss(
                {
                    "sky_coeffs": np.asarray(sky_j + delta),
                    "beam_coeffs": ini["beam_coeffs"],
                }
            )
        )

    loss0 = float(cal._loss(ini))
    print(f"loss at point: {loss0:.6e}\n")

    # --- per-frequency Grams B_f -------------------------------------------
    t0 = time.perf_counter()
    G = jnp.transpose(g_op, (1, 0, 2, 3)).reshape(nf, -1, P)  # (F, DT, P)
    w = jnp.transpose(inv_var, (1, 0, 2)).reshape(nf, -1)  # (F, DT)
    Bf = jnp.einsum("fnp,fnq->fpq", G * w[..., None], G)  # (F, P, P)
    Bf.block_until_ready()
    t_bf = time.perf_counter() - t0

    # --- exact direct solve (float64 reference) ----------------------------
    t0 = time.perf_counter()
    H = (2.0 / n_data) * jnp.einsum("fm,fn,fpq->pmqn", A, A, Bf)
    H = np.asarray(H, dtype=np.float64).reshape(P * M, P * M)
    H = 0.5 * (H + H.T)
    ridge = 1e-8 * np.trace(H) / (P * M)
    H[np.diag_indices_from(H)] += ridge
    L = np.linalg.cholesky(H)
    delta_exact = jax.scipy.linalg.cho_solve(
        (jnp.asarray(L), True), b.ravel().astype(jnp.float64)
    ).reshape(P, M)
    delta_exact = jnp.asarray(delta_exact, dtype=sky_j.dtype)
    t_direct = time.perf_counter() - t0
    eig = np.linalg.eigvalsh(H)
    eig = eig[eig > ridge * 10]
    print(
        f"H spectrum: cond {eig[-1] / eig[0]:.2e}  "
        f"({len(eig)} non-null of {P * M})"
    )
    print(
        f"EXACT      form_Bf {t_bf:5.2f}s  H+chol {t_direct:5.2f}s  "
        f"loss {loss0:.4e} -> {loss_of(delta_exact):.4e}\n"
    )

    # --- plain CG ----------------------------------------------------------
    for nit in (10, 25, 50):
        d = ops["sky_cg_solve"](g_op, sky_j, data_eff, inv_var, 1e-10, nit)
        d.block_until_ready()
        t0 = time.perf_counter()
        d = ops["sky_cg_solve"](g_op, sky_j, data_eff, inv_var, 1e-10, nit)
        d.block_until_ready()
        dt = time.perf_counter() - t0
        err = float(
            jnp.linalg.norm(d - delta_exact) / jnp.linalg.norm(delta_exact)
        )
        print(
            f"CG    n={nit:2d}   {dt:5.2f}s  loss {loss_of(d):.4e}  "
            f"err_vs_exact {err:.2e}"
        )

    # --- pixel-block (M x M) Jacobi preconditioned CG ----------------------
    Bdiag = jnp.einsum("fpp->fp", Bf)  # (F, P)
    Mp = (2.0 / n_data) * jnp.einsum("fm,fn,fp->pmn", A, A, Bdiag)  # (P,M,M)
    floor = 1e-3 * float(jnp.max(jnp.trace(Mp, axis1=1, axis2=2))) / M
    Minv = jnp.linalg.inv(Mp + floor * jnp.eye(M))  # (P, M, M)

    def precond(v):
        return jnp.einsum("pmn,pn->pm", Minv, v.reshape(P, M)).ravel()

    def hmv(v):
        h = (2.0 / n_data) * adj_op(inv_var * fwd_op(v.reshape(P, M)))
        return h.ravel() + 1e-10 * v

    from functools import partial

    @partial(jax.jit, static_argnums=(1,))
    def pcg(rhs, nit):
        x, _ = jax.scipy.sparse.linalg.cg(
            hmv, rhs, maxiter=nit, tol=1e-3, M=precond
        )
        return x

    for nit in (3, 5, 10):
        d = pcg(b.ravel(), nit)
        d.block_until_ready()
        t0 = time.perf_counter()
        d = pcg(b.ravel(), nit)
        d.block_until_ready()
        dt = time.perf_counter() - t0
        d = d.reshape(P, M)
        err = float(
            jnp.linalg.norm(d - delta_exact) / jnp.linalg.norm(delta_exact)
        )
        print(
            f"PCG   n={nit:2d}   {dt:5.2f}s  loss {loss_of(d):.4e}  "
            f"err_vs_exact {err:.2e}"
        )


if __name__ == "__main__":
    main()
