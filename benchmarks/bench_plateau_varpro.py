"""Prototype variable-projection (VarPro) solver for the bilinear problem.

The forward model is bilinear: linear in sky_coeffs at fixed beam_coeffs and
vice versa. VarPro eliminates the sky exactly at each step (the conditional
sky problem is quadratic) and takes a Newton-CG step on the reduced beam
objective phi(b) = min_s L(s, b), whose Hessian is the Schur complement
H_bb - H_bs H_ss^{-1} H_sb, applied matrix-free with nested CG.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from eigsep_sim.const import DTYPE_R_NPY


def varpro_fit(
    cal,
    params,
    max_outer=15,
    sky_cg=200,
    outer_cg=20,
    inner_cg=30,
    tol=0.0,
    verbose=True,
):
    """Run VarPro iterations; returns dict like Calibrator.fit."""
    params = {k: np.asarray(v).copy() for k, v in params.items()}
    losses, telemetry = [], []

    def loss_sb(s, b):
        return cal._loss({"sky_coeffs": s, "beam_coeffs": b})

    grad_s_fn = jax.grad(loss_sb, argnums=0)
    grad_b_fn = jax.grad(loss_sb, argnums=1)

    def solve_sky(params):
        return cal.sky_step(params, n_cg=sky_cg, step_size=1.0)

    converged = False
    for it in range(max_outer):
        tic = time.perf_counter()
        # 1. Exact conditional sky solve.
        params = solve_sky(params)
        s = jnp.asarray(params["sky_coeffs"])
        b = jnp.asarray(params["beam_coeffs"])
        loss_before = float(loss_sb(s, b))

        # 2. Reduced gradient = partial beam gradient at the conditional
        #    sky optimum (envelope theorem).
        gb = grad_b_fn(s, b)

        # 3. Schur-complement HVP with nested CG for H_ss^{-1}.
        def h_ss(w):
            return jax.jvp(lambda s_: grad_s_fn(s_, b), (s,), (w,))[1]

        def h_sb(v):
            return jax.jvp(lambda b_: grad_s_fn(s, b_), (b,), (v,))[1]

        def h_bs(w):
            return jax.jvp(lambda s_: grad_b_fn(s_, b), (s,), (w,))[1]

        def h_bb(v):
            return jax.jvp(lambda b_: grad_b_fn(s, b_), (b,), (v,))[1]

        def hvp_reduced(v_flat):
            v = v_flat.reshape(b.shape)
            rhs = h_sb(v)
            w, _ = jax.scipy.sparse.linalg.cg(
                h_ss, rhs, maxiter=inner_cg, tol=1e-2
            )
            out = h_bb(v) - h_bs(w)
            return out.ravel()

        delta, _ = jax.scipy.sparse.linalg.cg(
            hvp_reduced, -gb.ravel(), maxiter=outer_cg, tol=1e-2
        )

        # 4. Line search on the beam step, re-solving sky each trial.
        step = 1.0
        accepted = False
        for _ in range(8):
            trial = dict(params)
            trial["beam_coeffs"] = np.asarray(
                (b.ravel() + step * delta).reshape(b.shape),
                dtype=DTYPE_R_NPY,
            )
            trial = solve_sky(trial)
            loss_trial = float(cal._loss(trial))
            if loss_trial < loss_before:
                params = cal._project_scale_degeneracy(trial)
                loss = loss_trial
                accepted = True
                break
            step *= 0.25
        if not accepted:
            loss = loss_before

        dt = time.perf_counter() - tic
        losses.append(loss)
        telemetry.append(
            {
                "iteration": it,
                "wall_time": dt,
                "loss": loss,
                "step_type": f"varpro:{step if accepted else 0}",
            }
        )
        if verbose:
            print(
                f"varpro iter {it:3d}: loss = {loss:.6e}  "
                f"step = {step if accepted else 0:.3g}  dt = {dt:.2f}s",
                flush=True,
            )
        if it > 0 and abs(losses[-2] - loss) / (abs(losses[-2]) + 1e-30) < tol:
            converged = True
            break
        if not accepted:
            break

    return {
        "params": params,
        "losses": losses,
        "telemetry": telemetry,
        "converged": converged,
        "n_iter": len(losses),
        "solver": "varpro",
    }


if __name__ == "__main__":
    import json

    from bench_plateau_setup import (
        build_problem,
        make_calibrator,
        accuracy_metrics,
    )

    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "smooth"
    setup = build_problem(perturb_mode=mode)
    cal = make_calibrator(setup)
    cal._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=setup["geom"])

    tic = time.perf_counter()
    result = varpro_fit(
        cal, {k: v.copy() for k, v in setup["prms_ini"].items()}
    )
    total = time.perf_counter() - tic
    acc = accuracy_metrics(setup, result["params"])
    out = {
        "label": "varpro",
        "losses": [float(x) for x in result["losses"]],
        "times_s": list(
            np.cumsum([t["wall_time"] for t in result["telemetry"]])
        ),
        "total_s": total,
        "final_loss": float(result["losses"][-1]),
        "final_data_chi2": float(cal.data_loss(result["params"])),
        **acc,
    }
    print(out["final_loss"], out["final_data_chi2"], acc)
    with open(f"results/plateau_varpro_{mode}.json", "w") as f:
        json.dump(out, f, indent=1)
