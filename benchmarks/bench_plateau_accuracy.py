"""Accuracy vs lam_beam_harmonic for the fast-cg solver.

With exact conditional solves the optimizer reaches the regularized
minimum; this sweep finds the harmonic penalty strength at which that
minimum coincides with the injected truth (data chi2/datum -> ~1,
sky/beam fractional errors small).
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


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smooth"
    setup = build_problem(perturb_mode=mode)

    results = []
    for lam_h in [0.0, 1e2, 1e3]:
        cal = make_calibrator(setup, lam_beam_harmonic=lam_h)
        cal._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
        cal._resolve_geom(geom=setup["geom"])
        params = {k: v.copy() for k, v in setup["prms_ini"].items()}
        tic = time.perf_counter()
        result = cal.fit(
            params=params,
            geom=setup["geom"],
            max_iter=10,
            tol=1e-4,
            verbose=True,
            solver="fast-cg",
        )
        total = time.perf_counter() - tic
        acc = accuracy_metrics(setup, result["params"])
        entry = {
            "lam_beam_harmonic": lam_h,
            "final_loss": float(result["losses"][-1]),
            "final_data_chi2": float(cal.data_loss(result["params"])),
            "losses": [float(x) for x in result["losses"]],
            "times_s": list(
                np.cumsum([float(t["wall_time"]) for t in result["telemetry"]])
            ),
            "total_s": total,
            **acc,
        }
        results.append(entry)
        print(
            f"lam_h={lam_h:8.1e} loss={entry['final_loss']:10.4e} "
            f"data_chi2={entry['final_data_chi2']:10.4e} "
            f"sky_rms={acc['sky_frac_rms']:8.4f} "
            f"beam_rms={acc['beam_frac_rms']:8.4f} t={total:6.1f}s",
            flush=True,
        )

    out = f"results/plateau_accuracy_{mode}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
