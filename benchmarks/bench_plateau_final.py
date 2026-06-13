"""Final solution check: fast-cg to convergence + notebook-style accuracy.

Runs the recommended solver on the smooth-perturbation problem, saves the
fitted parameters, and evaluates accuracy after removing the per-frequency
sky/beam scale gauge exactly as the notebook does (RecoverySolution +
ScaleDegeneracy), alongside raw L2 relative errors.
"""

import json
import sys
import time

import numpy as np

from eigsep_sim.recovery import RecoverySolution, ScaleDegeneracy

from bench_plateau_setup import build_problem, make_calibrator


def notebook_metrics(setup, params):
    sky = setup["sky"]
    beam = setup["beam"]
    sky_tru = setup["prms_tru"]["sky_coeffs"] @ sky.basis.A.T
    beam_tru = setup["prms_tru"]["beam_coeffs"] @ beam.basis.A.T
    sky_fit = np.asarray(params["sky_coeffs"]) @ sky.basis.A.T
    beam_fit = np.asarray(params["beam_coeffs"]) @ beam.basis.A.T

    degens = [ScaleDegeneracy({"sky": 1.0, "beam": -1.0}, group_axes=(-1,))]
    ref = {"sky": sky_tru, "beam": beam_tru}
    fixed = (
        RecoverySolution({"sky": sky_fit, "beam": beam_fit}, degens)
        .remove_degen(ref, inplace=False)
        .params
    )
    sky_fix, beam_fix = fixed["sky"], fixed["beam"]

    def l2rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    def fracrms(a, b):
        return float(np.sqrt(np.mean(((a - b) / np.abs(b)) ** 2)))

    return {
        "sky_l2rel": l2rel(sky_fix, sky_tru),
        "beam_l2rel": l2rel(beam_fix, beam_tru),
        "sky_frac_rms": fracrms(sky_fix, sky_tru),
        "beam_frac_rms": fracrms(beam_fix, beam_tru),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smooth"
    lam_h = float(sys.argv[2]) if len(sys.argv) > 2 else 1e2
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    setup = build_problem(perturb_mode=mode)
    cal = make_calibrator(setup, lam_beam_harmonic=lam_h)
    cal._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=setup["geom"])

    print(
        "metrics at init:",
        notebook_metrics(setup, setup["prms_ini"]),
        flush=True,
    )

    params = {k: v.copy() for k, v in setup["prms_ini"].items()}
    tic = time.perf_counter()
    result = cal.fit(
        params=params,
        geom=setup["geom"],
        max_iter=max_iter,
        tol=1e-3,
        verbose=True,
        solver="fast-cg",
    )
    total = time.perf_counter() - tic

    metrics = notebook_metrics(setup, result["params"])
    data_chi2 = float(cal.data_loss(result["params"]))
    print(
        f"\nfinal loss={result['losses'][-1]:.4e} data_chi2={data_chi2:.4e}"
        f" t={total:.1f}s",
        flush=True,
    )
    print("metrics at fit :", metrics, flush=True)

    np.savez(
        f"results/plateau_final_{mode}.npz",
        sky_coeffs=result["params"]["sky_coeffs"],
        beam_coeffs=result["params"]["beam_coeffs"],
        sky_tru=setup["prms_tru"]["sky_coeffs"],
        beam_tru=setup["prms_tru"]["beam_coeffs"],
        sky_ini=setup["prms_ini"]["sky_coeffs"],
        beam_ini=setup["prms_ini"]["beam_coeffs"],
        losses=np.asarray(result["losses"]),
        times=np.cumsum([t["wall_time"] for t in result["telemetry"]]),
    )
    with open(f"results/plateau_final_{mode}.json", "w") as f:
        json.dump(
            {
                "lam_beam_harmonic": lam_h,
                "max_iter": max_iter,
                "final_loss": float(result["losses"][-1]),
                "data_chi2": data_chi2,
                "total_s": total,
                **metrics,
            },
            f,
            indent=1,
        )
    print("saved results/plateau_final_%s.{npz,json}" % mode, flush=True)


if __name__ == "__main__":
    main()
