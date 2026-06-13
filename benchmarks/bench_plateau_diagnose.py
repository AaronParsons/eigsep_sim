"""Diagnose the loss landscape at init vs truth for the notebook fit.

Decomposes the Calibrator loss into data chi2, beam ridge, and harmonic
penalty terms to determine whether the observed plateau is an optimizer
limitation or a regularizer-induced floor.
"""

import numpy as np

from bench_plateau_setup import build_problem, make_calibrator


def loss_components(cal, params):
    data = cal.data_loss(params)
    ridge = 0.0
    if cal._lam_beam > 0 and cal._beam_nom is not None:
        diff = np.asarray(params["beam_coeffs"]) - cal._beam_nom
        ridge = cal._lam_beam * float(np.mean(diff**2))
    harmonic = cal._lam_beam_harmonic * cal._beam_harmonic_penalty(
        params["beam_coeffs"]
    )
    return {
        "data": data,
        "beam_ridge": ridge,
        "beam_harmonic": harmonic,
        "total": data + ridge + harmonic,
    }


def main():
    setup = build_problem()
    cal = make_calibrator(setup)
    # Mimic the notebook: fit() called with explicit params sets
    # _beam_nom to the perturbed initial beam.
    cal._beam_nom = np.asarray(setup["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=setup["geom"])

    print("noise floor check: chi2/datum at truth should be ~1")
    for label, params in [
        ("init", setup["prms_ini"]),
        ("truth", setup["prms_tru"]),
    ]:
        comps = loss_components(cal, params)
        print(f"\n--- {label} ---")
        for k, v in comps.items():
            print(f"  {k:14s}: {v:.6e}")

    # Same decomposition with beam_nom set to the TRUTH beam, the intended
    # use when the nominal beam model is accurate.
    cal2 = make_calibrator(setup)
    cal2._beam_nom = np.asarray(setup["prms_tru"]["beam_coeffs"]).copy()
    cal2._resolve_geom(geom=setup["geom"])
    print("\n=== with beam_nom = truth beam ===")
    for label, params in [
        ("init", setup["prms_ini"]),
        ("truth", setup["prms_tru"]),
    ]:
        comps = loss_components(cal2, params)
        print(f"\n--- {label} ---")
        for k, v in comps.items():
            print(f"  {k:14s}: {v:.6e}")


if __name__ == "__main__":
    main()
