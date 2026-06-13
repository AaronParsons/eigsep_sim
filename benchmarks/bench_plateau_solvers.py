"""Race Calibrator solvers on the notebook recovery problem.

For each solver configuration, runs a fresh fit from the same perturbed
init and records the loss trajectory against cumulative wall time, the
final data chi2, and truth-recovery accuracy. Results go to
benchmarks/results/plateau_solvers_<mode>.json.
"""

import json
import sys
import time

import numpy as np

from bench_plateau_setup import (
    build_problem,
    make_calibrator,
    accuracy_metrics,
)


def run_fit_solver(setup, label, fit_kwargs, cal_kwargs=None, max_iter=40):
    cal = make_calibrator(setup, **(cal_kwargs or {}))
    # Match the notebook: explicit params means _beam_nom is the init beam.
    cal._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=setup["geom"])
    params = {k: v.copy() for k, v in setup["prms_ini"].items()}

    print(f"--- running {label} ---", flush=True)
    tic = time.perf_counter()
    result = cal.fit(
        params=params,
        geom=setup["geom"],
        max_iter=max_iter,
        tol=0.0,
        verbose=True,
        **fit_kwargs,
    )
    total = time.perf_counter() - tic
    times = np.cumsum([t["wall_time"] for t in result["telemetry"]])
    entry = summarize(
        setup, cal, label, result["params"], result["losses"], times, total
    )
    return entry


def run_lbfgs(setup, label, maxiter):
    cal = make_calibrator(setup)
    cal._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=setup["geom"])
    params = {k: v.copy() for k, v in setup["prms_ini"].items()}
    tic = time.perf_counter()
    result = cal.fit_lbfgs(params, maxiter=maxiter)
    total = time.perf_counter() - tic
    losses = [float(result["losses"][-1])]
    return summarize(
        setup, cal, label, result["params"], losses, [total], total
    )


def summarize(setup, cal, label, params, losses, times, total):
    acc = accuracy_metrics(setup, params)
    entry = {
        "label": label,
        "losses": [float(x) for x in losses],
        "times_s": [float(x) for x in times],
        "total_s": float(total),
        "final_loss": float(losses[-1]),
        "final_data_chi2": float(cal.data_loss(params)),
        **acc,
    }
    print(
        f"{label:28s} loss={entry['final_loss']:12.4e} "
        f"data_chi2={entry['final_data_chi2']:10.4e} "
        f"sky_rms={acc['sky_frac_rms']:8.4f} "
        f"beam_rms={acc['beam_frac_rms']:8.4f} "
        f"t={total:7.1f}s",
        flush=True,
    )
    return entry


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smooth"
    setup = build_problem(perturb_mode=mode)

    # Warm up JIT so the first benchmarked solver is not charged compile time.
    cal0 = make_calibrator(setup)
    cal0._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    cal0._resolve_geom(geom=setup["geom"])
    cal0.fit(
        params={k: v.copy() for k, v in setup["prms_ini"].items()},
        geom=setup["geom"],
        max_iter=1,
        verbose=False,
    )

    print(f"perturb_mode={mode}")
    print(f"loss at init : {cal0._loss(setup['prms_ini']):.6e}")
    cal0._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    print(f"loss at truth: {cal0._loss(setup['prms_tru']):.6e}")

    results = []
    # max_iter per solver targets roughly equal (~2-3 min) wall budgets;
    # telemetry timestamps make trajectories comparable regardless.
    variants = [
        (
            "sched-damp100 (notebook)",
            {"solver": "adaptive-scheduled", "lambda_damp": 100.0},
            None,
            40,
        ),
        (
            "sched-damp1e-2",
            {"solver": "adaptive-scheduled", "lambda_damp": 1e-2},
            None,
            40,
        ),
        (
            "afp-damp1e-2",
            {"solver": "adaptive-fixed-point", "lambda_damp": 1e-2},
            None,
            25,
        ),
        ("cg (ALS+AA)", {"solver": "cg"}, None, 8),
        ("fast-cg", {"solver": "fast-cg"}, None, 12),
        ("joint-newton-cg", {"solver": "joint"}, None, 6),
    ]
    for label, fit_kwargs, cal_kwargs, max_iter in variants:
        results.append(
            run_fit_solver(
                setup, label, fit_kwargs, cal_kwargs, max_iter=max_iter
            )
        )

    results.append(run_lbfgs(setup, "lbfgs-200", maxiter=200))

    out = f"results/plateau_solvers_{mode}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
