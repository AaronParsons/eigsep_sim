"""Benchmark the EIGSEP_Recovery_v001_NewFramework notebook core.

The default configuration mirrors the current notebook. It writes a JSON
summary with fit telemetry so solver changes can be compared without relying
on notebook state.

Examples
--------
    python benchmarks/recovery_new_framework.py
    python benchmarks/recovery_new_framework.py --sky synthetic --solver all \
        --nside-sky 4 --nside-beam 4 --nfreq 6 --ntimes 4 --naz 2 --nalt 2 \
        --k-sky 2 --k-beam 2 --max-iter 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import astropy.units as u
import healpy
import jax
import numpy as np
from astropy.time import Time

from eigsep_sim import (
    Beam,
    Calibrator,
    EarthSurface,
    ForwardModel,
    NullTerrain,
    Sky,
)
from eigsep_sim.recovery import RecoverySolution, ScaleDegeneracy, relative_rms


def elapsed(label, fn):
    start = time.perf_counter()
    result = fn()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    seconds = time.perf_counter() - start
    print(f"  {label:<34s} {seconds:8.3f} s")
    return result, seconds


def synthetic_sky(nside, freqs_hz, n_modes):
    x, y, z = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
    spectral_index = -2.5 + 0.15 * x - 0.10 * y
    sky_map = (180.0 + 45.0 * z[:, None] + 15.0 * x[:, None]) * (
        freqs_hz[None, :] / 100e6
    ) ** spectral_index[:, None]
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=n_modes)
    return sky, sky.basis.project(sky_map)


def make_model(args):
    freqs_hz = np.linspace(55e6, 150e6, args.nfreq)
    axes_body = np.eye(3, dtype=np.float32)[: args.n_beam_pols]
    beam = Beam.from_dipole(
        args.nside_beam,
        freqs_hz,
        arm_lengths_m=2.0,
        u_body=axes_body,
        K=args.k_beam,
    )
    if args.sky == "gsm":
        sky = Sky.from_gsm(
            args.nside_sky, freqs_hz, n_modes=args.k_sky, include_flat=True
        )
        sky_coeffs = sky.init_coeffs()
    else:
        sky, sky_coeffs = synthetic_sky(args.nside_sky, freqs_hz, args.k_sky)
    observer = EarthSurface(lat=39.2, lon=-113.4)
    return ForwardModel(observer, beam, sky, terrain=NullTerrain()), sky_coeffs


def make_geometry(fwd, args):
    az_rad = np.linspace(0.0, 2.0 * np.pi, args.naz, endpoint=False)
    alt_rad = np.linspace(0.0, 0.5 * np.pi, args.nalt)
    orient_rots = np.stack(
        [Beam.top2body(az, alt) for alt in alt_rad for az in az_rad]
    )
    base_times = (
        Time("2025-01-01")
        + np.linspace(0.0, 86400.0, args.ntimes, endpoint=False) * u.s
    )
    base_rots = fwd.observer.rot_gal2top_stack(base_times)
    rots = np.repeat(base_rots, len(orient_rots), axis=0)
    body_rots = np.tile(orient_rots, (args.ntimes, 1, 1))
    return fwd.precompute_geometry(rots=rots, body_rots=body_rots)


def sampled_beam_weights(geom, beam_coeffs, beam_basis_A):
    """Return per-sample integrated beam weights in simulator units."""
    beam_maps = beam_coeffs @ beam_basis_A.T
    pixels = np.asarray(geom["beam_px_jax"])
    weights = np.asarray(geom["beam_wgts_jax"])
    ntimes = pixels.shape[0]
    n_dipoles, _, nfreq = beam_maps.shape
    beam_weights = np.zeros((ntimes, n_dipoles, nfreq), dtype=float)
    for freq_index in range(nfreq):
        for dipole_index in range(n_dipoles):
            beam_map = beam_maps[dipole_index, :, freq_index]
            beam_weights[:, dipole_index, freq_index] = sum(
                (
                    beam_map[pixels[:, neighbor_index, :]]
                    * weights[:, neighbor_index, :]
                ).sum(axis=1)
                for neighbor_index in range(4)
            )
    return beam_weights


def solver_options(name, args):
    if name == "adaptive-fixed-point":
        return {
            "solver": "adaptive-fixed-point",
            "lambda_damp": args.lambda_damp,
        }
    if name == "adaptive-scheduled":
        return {
            "solver": "adaptive-scheduled",
            "lambda_damp": args.lambda_damp,
            "schedule_max_every": {
                "sky": args.schedule_sky_max_every,
                "beam": args.schedule_beam_max_every,
                "joint": args.schedule_joint_max_every,
            },
            "schedule_eff_alpha": args.schedule_eff_alpha,
            "schedule_step_gain_factor": args.schedule_step_gain_factor,
        }
    if name == "hybrid-lbfgs":
        return {
            "solver": "hybrid-lbfgs",
            "lambda_damp": args.lambda_damp,
        }
    if name == "alternating":
        return {"solver": "alternating"}
    if name == "fast-cg":
        return {
            "solver": "fast-cg",
            "beam_cg_niter": args.beam_cg_niter,
            "beam_cg_tol": args.beam_cg_tol,
        }
    if name == "cg":
        return {
            "solver": "cg",
            "beam_cg_niter": args.beam_cg_niter,
            "beam_cg_tol": args.beam_cg_tol,
        }
    if name == "joint":
        return {"solver": "joint"}
    raise ValueError(f"unknown solver {name}")


def run_solver(
    name, fwd, geom, data, inv_noise_var, params_ini, maps_true, args
):
    calibrator = Calibrator(
        fwd,
        data,
        inv_noise_var=inv_noise_var,
        m_anderson=args.m_anderson,
        lam_beam=args.lam_beam,
        lam_sky=0.0,
    )
    calibrator._geom = geom
    calibrator._beam_nom = fwd.beam.coeffs.copy()
    params = {key: value.copy() for key, value in params_ini.items()}
    result, seconds = elapsed(
        f"fit [{name}]",
        lambda: calibrator.fit(
            params=params,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=False,
            **solver_options(name, args),
        ),
    )
    sky_map = result["params"]["sky_coeffs"] @ fwd.sky.basis.A.T
    beam_map = result["params"]["beam_coeffs"] @ fwd.beam.basis.A.T
    projected = RecoverySolution(
        {"sky": sky_map, "beam": beam_map},
        [ScaleDegeneracy({"sky": 1.0, "beam": -1.0}, group_axes=(-1,))],
    ).remove_degen({"sky": maps_true[0], "beam": maps_true[1]}, inplace=False)
    errors = {
        "sky_relative_rms": relative_rms(
            projected.params["sky"], maps_true[0]
        ),
        "beam_relative_rms": relative_rms(
            projected.params["beam"], maps_true[1]
        ),
    }
    telemetry = result.get("telemetry", [])
    if telemetry:
        last = telemetry[-1]
        dchi2_total = (
            telemetry[0].get("loss", result["losses"][0]) - last["loss"]
        )
        telemetry_summary = {
            "last_step_type": last.get("step_type"),
            "last_delta_chi2_per_sec": last.get("delta_chi2_per_sec"),
            "total_delta_chi2": dchi2_total,
            "median_delta_chi2_per_sec": float(
                np.nanmedian(
                    [t.get("delta_chi2_per_sec", np.nan) for t in telemetry]
                )
            ),
            "beam_roughness_initial": telemetry[0].get("beam_roughness"),
            "beam_roughness_final": last.get("beam_roughness"),
            "beam_shape_update_rms_final": last.get("beam_shape_update_rms"),
            "beam_scale_update_rms_final": last.get("beam_scale_update_rms"),
            "joint_beam_shape_update_rms_final": last.get(
                "joint_beam_shape_update_rms"
            ),
            "joint_beam_scale_update_rms_final": last.get(
                "joint_beam_scale_update_rms"
            ),
            "step_type_counts": {
                step: sum(1 for t in telemetry if t.get("step_type") == step)
                for step in sorted({t.get("step_type") for t in telemetry})
            },
        }
    else:
        telemetry_summary = {}
    print(
        f"    iterations={result['n_iter']} converged={result['converged']} "
        f"loss={result['losses'][-1]:.4e} "
        f"sky_err={100 * errors['sky_relative_rms']:.3f}% "
        f"beam_err={100 * errors['beam_relative_rms']:.3f}% "
        f"last_step={telemetry_summary.get('last_step_type')}"
    )
    return seconds, result, errors, telemetry_summary


def main(args):
    if args.n_beam_pols not in (1, 2):
        raise ValueError("n_beam_pols must be 1 or 2")
    print(f"Recovery NewFramework benchmark [{jax.default_backend()}]")
    print(
        f"  sky={args.sky} nside_sky={args.nside_sky} "
        f"nside_beam={args.nside_beam} nfreq={args.nfreq} "
        f"ntimes={args.ntimes} orientations={args.naz * args.nalt} "
        f"beam_pols={args.n_beam_pols}"
    )
    fwd, sky_coeffs = make_model(args)
    geom, _ = elapsed("precompute_geometry", lambda: make_geometry(fwd, args))
    beam_coeffs = fwd.beam.coeffs.copy()
    antenna_temp, _ = elapsed(
        "simulate [JIT compile]",
        lambda: fwd.simulate(sky_coeffs, beam_coeffs, geom=geom),
    )
    _, _ = elapsed(
        "simulate [steady state]",
        lambda: fwd.simulate(sky_coeffs, beam_coeffs, geom=geom),
    )

    sky_map_true = sky_coeffs @ fwd.sky.basis.A.T
    beam_map_true = beam_coeffs @ fwd.beam.basis.A.T
    delta_nu_hz = float(np.diff(fwd.beam.freqs_hz).mean())
    tau_s = 86400.0 / (args.ntimes * args.naz * args.nalt)
    beam_weight = sampled_beam_weights(geom, beam_coeffs, fwd.beam.basis.A)
    sigma_noise = (
        np.abs(np.asarray(antenna_temp)) + args.t_rx_k * np.abs(beam_weight)
    ) / np.sqrt(delta_nu_hz * tau_s)
    rng = np.random.default_rng(args.seed)
    data = np.asarray(antenna_temp) + rng.normal(
        scale=sigma_noise, size=antenna_temp.shape
    )
    params_ini = {
        "sky_coeffs": sky_coeffs
        * rng.uniform(0.9, 1.1, size=sky_coeffs.shape),
        "beam_coeffs": beam_coeffs
        * rng.uniform(0.9, 1.1, size=beam_coeffs.shape),
    }
    solvers = (
        [
            "adaptive-scheduled",
            "adaptive-fixed-point",
            "fast-cg",
            "cg",
            "joint",
            "alternating",
        ]
        if args.solver == "all"
        else [args.solver]
    )
    summaries = []
    for solver in solvers:
        seconds, result, errors, telemetry_summary = run_solver(
            solver,
            fwd,
            geom,
            data,
            1.0 / sigma_noise.reshape(-1, args.nfreq) ** 2,
            params_ini,
            (sky_map_true, beam_map_true),
            args,
        )
        summaries.append(
            {
                "solver": solver,
                "seconds": seconds,
                "n_iter": result["n_iter"],
                "converged": bool(result["converged"]),
                "loss_initial": (
                    float(result["losses"][0]) if result["losses"] else None
                ),
                "loss_final": (
                    float(result["losses"][-1]) if result["losses"] else None
                ),
                "sky_relative_rms": float(errors["sky_relative_rms"]),
                "beam_relative_rms": float(errors["beam_relative_rms"]),
                "telemetry_summary": telemetry_summary,
                "telemetry": result.get("telemetry", []),
            }
        )

    output = {
        "config": vars(args),
        "backend": jax.default_backend(),
        "summaries": summaries,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=float))
    print(f"Wrote telemetry: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sky", choices=["gsm", "synthetic"], default="gsm")
    parser.add_argument(
        "--solver",
        choices=[
            "adaptive-fixed-point",
            "hybrid-lbfgs",
            "adaptive-scheduled",
            "alternating",
            "fast-cg",
            "cg",
            "joint",
            "all",
        ],
        default="adaptive-fixed-point",
    )
    parser.add_argument("--nside-sky", type=int, default=8)
    parser.add_argument("--nside-beam", type=int, default=8)
    parser.add_argument("--nfreq", type=int, default=20)
    parser.add_argument("--ntimes", type=int, default=36)
    parser.add_argument("--naz", type=int, default=6)
    parser.add_argument("--nalt", type=int, default=5)
    parser.add_argument("--n-beam-pols", type=int, default=1)
    parser.add_argument("--k-sky", type=int, default=5)
    parser.add_argument("--k-beam", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--lam-beam", type=float, default=0.01)
    parser.add_argument("--m-anderson", type=int, default=5)
    parser.add_argument("--lambda-damp", type=float, default=1e-1)
    parser.add_argument("--schedule-sky-max-every", type=int, default=5)
    parser.add_argument("--schedule-beam-max-every", type=int, default=2)
    parser.add_argument("--schedule-joint-max-every", type=int, default=4)
    parser.add_argument("--schedule-eff-alpha", type=float, default=0.3)
    parser.add_argument("--schedule-step-gain-factor", type=float, default=2.0)
    parser.add_argument("--beam-cg-niter", type=int, default=8)
    parser.add_argument("--beam-cg-tol", type=float, default=1e-2)
    parser.add_argument("--t-rx-k", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="benchmarks/results/recovery_new_framework_latest.json",
    )
    main(parser.parse_args())
