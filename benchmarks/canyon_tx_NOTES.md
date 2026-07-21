# Canyon TX-anchored joint sky+beam recovery — research notes

Goal: use a ground transmitter directly below the antenna (nadir), transmitting
in every 8th channel with near-perfect isolation in adjacent channels, together
with terrain masking and many antenna orientations, to solve sky AND beam at
higher resolution than the sky-only / degenerate joint problem allows.

## Key geometry fact (verified 2026-06-12)

`Beam.top2body(az, alt) = R_x(-alt) @ R_z(-az)`. The nadir direction `[0,0,-1]`
lies on the topocentric z-axis, which `R_z(-az)` leaves fixed, so the TX
direction in the BODY frame is `[0, -sin(alt), -cos(alt)]` — **independent of
azimuth**. The transmitter therefore samples the beam along a single 1-D great
circle (the body y-z meridian), parameterized only by the tilt `alt`.

Implication: TX provides
  1. an ABSOLUTE beam measurement (sky-independent) -> fixes the overall
     beam scale and breaks the multiplicative sky x beam degeneracy that
     limited v001;
  2. the beam radial profile along the meridian, at the TX frequencies
     (every 8th channel) -> constrains the K spectral beam modes there;
  3. NOT direct off-meridian / azimuthal beam structure — that still comes
     from the bilinear sky integrals as the masked sky rotates through the beam.
Tilt diversity (range of `alt`) sets how much of the beam meridian is covered.

## Forward-model structure

Per orientation t, channel f, single dipole:
  T[t,f] = sum_p (beam@sky)[t,p,f] * mask[t,p] * sky_recon[p,f]      (sky, bilinear)
         + sum_p (beam@sky)[t,p,f] * terrain_emit[t,p,f]            (terrain, lin in beam)
         + T_gnd * (beam@blocked)                                   (ground, lin in beam)
         + beam@nadir[t,f] * P_tx[f]                                (TX, lin in beam, sky-indep)
All terms are beam-weighted; only the sky term is bilinear. Given the beam,
the sky problem is linear (data - offset, offset = simulate(sky=0, beam)).
Given the sky, the whole prediction is linear in beam_coeffs (build W including
sky+terrain+ground+TX), so the beam solve is also an exact quadratic — but it is
only well-posed because TX anchors scale + meridian.

## Plan

1. Prototype alternating exact solves (sky direct Cholesky; beam direct Cholesky
   with TX in the beam operator). Validate that TX breaks the joint degeneracy
   (beam recovered with TX, not without).
2. Explore resolution: vary nside_sky, nside_beam (or a harmonic beam basis),
   n_orientations, tilt diversity, terrain on/off, TX on/off. Measure
   conditioning (kappa), recovery error, runtime.
3. Interval hierarchy: coarse-to-fine; multi-resolution beam.
4. Integrate good approach into src (TX-aware ops + beam direct solve) and build
   a notebook. Tests.

## Results log

### 2026-06-12/13 prototype (canyon_tx_lib.py, validated vs simulate to 1e-15)

- Operators (sky design G, beam operator W including TX) reproduce
  ForwardModel.simulate to machine precision.
- **float64 is required** in the conditional Gram/solve: TX power makes the
  forward span a huge dynamic range; float32 accumulation loses the beam-from-
  sky correction. Solves now accumulate in numpy float64. TX power for the
  prototype set to 1e3 (the physically relevant quantity is TX SNR, not the
  absolute level; 1e6 also blows float32 in the forward `ref` itself).
- **Single sidereal time is sky-rank-deficient**: noiseless sky solve fit data
  to 4e-6 but recovered coeffs only to 29% — the fixed terrain mask + smooth
  beam give correlated orientation integrals. Adding sidereal time sampling
  (orientation scan nested in n_times) rotates the galactic sky through the
  fixed topocentric horizon: visible pixels 317->517 and sky map err 0.29->0.082
  (nside_sky=8, 8 times x 16 az x 8 alt).
- **TX gives perfect conditional beam recovery**: beam map err 0.347 (single
  time) -> 0.000 (multi-time), chi2 ~ noise floor. TX breaks the sky x beam
  scale degeneracy (TX = beam x P_tx is sky-independent, so scaling beam changes
  the TX prediction).
- Open: sky conditional chi2 ~6 (map err good) — investigate (rank tail / SNR /
  channel weighting). Next: ALS joint solve (both perturbed), then resolution
  scaling vs n_times, nside, tilt diversity, TX on/off.

### Resolution characterization (snr_frac=1e-3, nside_sky=nside_beam=8, canyon_tx_resolution.py)

The conditional Gram is ill-posed; effective resolution = #modes above the
noise (Wiener/Tikhonov oracle). Key numbers:

SKY (given true beam), unknowns ~2068:
- baseline (terrain, 8 times): 760 modes, cond 1.2e12, oracle map_err 0.45
- no terrain:                  396 modes, cond 2.8e30 (!) — terrain breaks degeneracy
- 1 sidereal time:             241 modes, cond 4.3e29
- 16 sidereal times:          1157 modes, cond 3.7e11
=> sky resolution scales with sidereal-time sampling AND requires terrain.

BEAM (given true sky), unknowns 3840:
- TX on:  3045 modes, map_err 0.268
- TX off: 2579 modes, map_err 0.311
=> Given a KNOWN sky, the bilinear integrals already constrain most of the
   beam; TX adds the absolute scale + meridian (+466 modes). TX's decisive
   value is in the JOINT problem (removing the sky x beam scale/shape gauge),
   which the regularized ALS (canyon_tx_joint.py) is set up to show.

Regularization: relative Tikhonov lam vs top Gram eig; oracle lam_sky ~1e-8,
lam_beam ~1e-7 at this SNR. Solves use eigen-basis (one eigh per conditional
solve) so lambda is cheap to apply/sweep.

### Scaling sweep (canyon_tx_scaling.py, snr=1e-3, nside 8/8)

SKY modes vs sidereal times (given true beam):
  nt:    1    2    4    8   16   24
  modes:231  282  555  729 1157 1251   (out of ~2070)
  map_err:.543 .479 .455 .447 .392 .368
=> resolution keeps growing with sidereal-time sampling (the interval
   hierarchy); diminishing returns past ~16 times at this nside/SNR.

BEAM modes (given true sky): TX on vs off nearly equal (a known sky already
constrains the beam); both reach 3840/3840 by 16 times. TX's value is in the
JOINT problem, not the conditional-given-true-sky problem.

### Joint regularized ALS (canyon_tx_joint.py, both perturbed 10%, snr=1e-3)

TX-on: converges monotonically — sky_err 1.10 -> 0.62, beam_err 0.094 -> 0.084
over 8 rounds, chi2 -> 0.71. The naive (unregularized) ALS diverged; Tikhonov
(eigen-basis Wiener) stabilizes it and the TX anchor fixes the scale gauge.
TX-off: sky_err 1.10 -> 0.62, beam_err 0.094 -> 0.084 — IDENTICAL to TX-on.

### Scale degeneracy: terrain, not TX, is the absolute reference (canyon_tx_degeneracy.py / canyon_tx_power.py)

delta-chi2(+/-10% scale), normalized to chi2(s=1):
- no terrain, no TX: 0.00  (exactly degenerate — the v001 sky x beam scale gauge)
- terrain only:      1906   (known ground emission T_gnd through the beam = absolute reference)
- terrain + TX(1e3): 1911   (+0.3%)
TX-only (no terrain) scale constraint vs power: 1e2->0, 1e3->0, 1e4->3, 1e5->279
(grows ~ power^2; needs TX >> noise to matter). terrain+TX: 1906/1911/1967/2922.
=> At moderate TX power TERRAIN dominates the absolute-scale constraint; the
   joint TX-on and TX-off recoveries are identical. The transmitter's distinct
   value is the extra beam SHAPE modes along the 1-D meridian (a known point
   source) — terrain's diffuse uniform ground only constrains the beam integral.

NOTE: canyon_tx_tgnd.py (T_gnd-uncertainty robustness) was inconclusive — it
varied make_ops t_gnd, which only affects the ~zero observer-occlusion term,
not the terrain object's baked-in emission. Not claimed.

## CORRECTIONS from Aaron (2026-06-13) — addressed

1. **TX coverage is 2-D, not 1-D.** My 1-D "meridian" was an artifact of the
   orientation composition (top2body = R_x(-alt)R_z(-az), az INNER about the
   vertical with nadir on that axis). The physical hanging/tumbling antenna has
   azimuth as the OUTER rotation of an already-tilted body: top2body =
   R_z(-az)R_x(-alt). Then nadir sweeps a 2-D cone. Verified: tumble scan
   touches 330/768 beam pixels (43% of sphere, the lower hemisphere) vs 30/768
   (4%) for the alt-az convention. build_canyon now has scan="tumble" (default)
   / "altaz". This makes the TX a genuine 2-D beam-mapping probe.
2. **TX sits ON the terrain -> independent beam probe.** A bright, known point
   source in the (blocked) lower hemisphere; distinct from the diffuse terrain
   ground emission, so it adds independent beam constraints / redundancy beyond
   the scale reference.
3. **Adjacent-channel differencing isolates the TX.** The sky+terrain is smooth
   in frequency; a LOCAL (windowed) low-order fit to neighbouring TX-free
   channels, evaluated at the TX channel and subtracted, leaves the pure TX
   tone (= beam x P_tx, sky-independent). isolate_tx() does this. Accuracy vs
   TX power (tumble, every-4th channel): P=1e3 -> 73% (TX below the smooth
   curvature), 1e4 -> 7.4%, 1e5 -> 0.8%, 1e6 -> 0.17%. So a bright calibration
   tone is cleanly separable; beam_from_tx() then solves the beam from the TX
   alone (sky-independent, absolute). Standalone per-pixel beam-from-TX over the
   covered backlobe saturates ~0.49 (interpolation/deconvolution + low beam
   values there); the value is as an independent anchor in the joint solve.

## TX value with 2-D coverage (canyon_tx_value.py, tumble scan, P=1e5)

CLEAN metric — beam modes (given true sky, above-noise): TX-on **1626**/3840 vs
TX-off **1332** — with 2-D coverage the TX adds ~290 independent beam modes (vs
~17 in the old 1-D scan). This is the unconfounded measure of TX value and it is
clearly positive.

The joint sky_err comparison (TX-on 0.614 vs TX-off 0.491) is a NOISE-MODEL
ARTIFACT, not contamination: make_data sets sigma = snr_frac * RMS(all channels),
so the bright TX (1e5, comparable to the antenna temperature) inflates the global
RMS and over-noises the science channels in the TX-on case. Excluding TX channels
from the sky step (value2.log) gave identical results, confirming it is the
global-sigma scaling, not leakage. A per-channel/radiometer noise model would
make the comparison fair. Recommended architecture: per-channel noise, science
channels for the sky, isolated-TX (isolate_tx/beam_from_tx) as an independent
absolute beam constraint — partially prototyped; full integration is follow-up.

## CONCLUSIONS (the level at which the canyon system can be solved)

1. Terrain is central: masks sky (resolution via time) + ground emission breaks
   the scale degeneracy.
2. Sky resolution scales with sidereal-time sampling (interval hierarchy):
   ~230 modes (1 time) -> ~1250 (24 times) at nside_sky=8, snr 1e-3.
3. Beam is fully solvable given the sky with enough time/tilt; TX adds
   incremental meridian-shape modes.
4. Regularized eigen-basis (Wiener) ALS jointly recovers both (naive ALS
   diverges). Effective resolution = #Gram modes above noise.
5. Notebook: notebooks/EIGSEP_Recovery_v003_CanyonTX.ipynb (built from
   build_v003_notebook.py).
