"""
EIGSEP Canyon Demo
==================
Simulate antenna temperature for EIGSEP: one bowtie antenna suspended 100 m
above Marjum Canyon floor, scanning in azimuth and altitude, with a CW
transmitter placed directly below the antenna on the canyon floor.

Geography: Marjum Canyon, Millard County, Utah  (39.2°N, 113.4°W, ~1600 m)

Az/alt convention (matching EIGSEP_Simulations.ipynb)
------------------------------------------------------
At az=0, alt=0 the body frame coincides with the topocentric frame
(x = east, y = north, z = up).

  Az  — rotation around ẑ (vertical), counterclockwise.
  Alt — rotation around x̂ (east axis); 0 = no tilt, π/2 = antenna sideways.

Per-step top→body rotation (see Beam.top2body):
    R_top2body = R_x(−alt) @ R_z(−az)

Both az and alt span [0, 2π).

Transmitter physics
-------------------
Nadir in topocentric = [0, 0, −1].  In body frame:
    R_top2body @ [0,0,−1] = R_x(−alt) @ [0,0,−1] = [0, −sin(alt), −cos(alt)]

For the bowtie (symmetric around x̂), the coupling to nadir is dominated
by the alt angle; the az dependence comes from chromatic beam asymmetry.

Usage
-----
    python notebooks/EIGSEP_Canyon_Demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import healpy

from eigsep_sim import Sky, Beam, ForwardModel
from eigsep_sim.observer import EarthSurface
from eigsep_sim.beam import BEAM_NPZ
from eigsep_sim.basis import BeamBasis

# ── Configuration ─────────────────────────────────────────────────────────────

LAT      = 39.2     # Marjum Canyon, Millard County, Utah [deg N]
LON      = -113.4   # [deg E]  (matches EIGSEP_Simulations.ipynb)
HEIGHT_M = 1600.0   # observer height above ellipsoid [m]  (floor ~1500 m + 100 m)

FREQ_MIN = 50e6     # [Hz]
FREQ_MAX = 200e6
N_FREQ   = 20

NSIDE_SKY   = 16    # HEALPix resolution for sky (npix = 3072)
N_SKY_MODES = 3     # GSM spectral modes
K_BEAM      = 5     # beam spectral modes (SVD truncation of bowtie)

T_GND = 300.0       # ground temperature for terrain-blocked pixels [K]
T_TX  = 1e6         # transmitter effective temperature [K]  (tunable)
TX_DIR_TOP = np.array([0.0, 0.0, -1.0])   # nadir in topocentric frame


OBS_TIME = Time("2025-06-21T04:00:00")    # summer solstice, ~midnight local

N_AZ  = 36          # azimuth steps, 0–2π
N_ALT = 36          # altitude steps, 0–2π  (matches v002/v003 notebooks)

freqs_hz  = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
freqs_mhz = freqs_hz / 1e6
az_rad    = np.linspace(0, 2 * np.pi, N_AZ,  endpoint=False)
alt_rad   = np.linspace(0, 2 * np.pi, N_ALT, endpoint=False)
TX_FREQS  = freqs_hz[::4]
TX_POW    = T_TX * np.ones_like(TX_FREQS)


# ── Build observer ─────────────────────────────────────────────────────────────

print("Building observer …")
observer = EarthSurface(lat=LAT, lon=LON, height=HEIGHT_M)
observer.set_time(OBS_TIME)

# ── Build beam from measured bowtie pattern ────────────────────────────────────
# Load the eigsep_bowtie_v000.npz beam and compress it into K spectral modes.
# The NPZ stores bm as (nfreq_file, npix); load_beam_file interpolates to freqs_hz.

print("Loading bowtie beam and building spectral basis …")
basis, coeffs = BeamBasis.from_beam_file(BEAM_NPZ, freqs_hz, K_BEAM)
npix_beam  = coeffs.shape[1]
nside_beam = healpy.npix2nside(npix_beam)
beam = Beam(nside_beam, freqs_hz, basis, coeffs,
            u_body=np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
print(f"  nside_beam={nside_beam}, npix_beam={npix_beam}, K_BEAM={K_BEAM}")

# ── Build sky ─────────────────────────────────────────────────────────────────

print("Building GSM sky …")
sky        = Sky.from_gsm(NSIDE_SKY, freqs_hz, n_modes=N_SKY_MODES)
sky_coeffs = sky.init_coeffs()

fwd = ForwardModel(observer, beam, sky,
                   transmitters=[(TX_DIR_TOP, TX_FREQS, TX_POW)])

# ── Compute geometry for the az/alt scan ──────────────────────────────────────
# Compute gal→top rotation once; replicate for all n_scan pointings.
# body_rots encodes the per-pointing antenna orientation.

R_gal2top = observer.rot_gal2top().astype(np.float32)

body_rots_scan = []
for az in az_rad:
    for alt in alt_rad:
        body_rots_scan.append(Beam.top2body(az, alt))

n_scan = N_AZ * N_ALT
rots_scan = [R_gal2top] * n_scan    # fixed sky rotation at OBS_TIME

print(f"Precomputing geometry for {n_scan} (az, alt) pointings …")
geom = fwd.precompute_geometry(rots=rots_scan, body_rots=body_rots_scan)

# ── Simulate sky antenna temperature ──────────────────────────────────────────

print("Simulating sky + transmitter Tant …")
T_total_raw = np.array(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom, T_gnd=T_GND))
# shape: (n_scan, 1, N_FREQ) → (N_AZ, N_ALT, N_FREQ)
T_total = T_total_raw[:, 0, :].reshape(N_AZ, N_ALT, N_FREQ)

# ── Plots ──────────────────────────────────────────────────────────────────────

f_mid = N_FREQ // 2
az_deg  = np.rad2deg(az_rad)
alt_deg = np.rad2deg(alt_rad)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    f"EIGSEP Canyon Demo — Marjum Canyon (39.2°N, 113.4°W, {HEIGHT_M:.0f} m)\n"
    f"Bowtie beam (nside={nside_beam}), {N_AZ}×{N_ALT} az/alt, "
    f"f = {FREQ_MIN/1e6:.0f}–{FREQ_MAX/1e6:.0f} MHz",
    fontsize=12,
)

# 1. Sky Tant vs azimuth (alt=0, centre frequency)
ax = axes[0, 0]
ax.plot(az_deg, T_total[:, 0, f_mid])
ax.set_xlabel("Azimuth [deg]")
ax.set_ylabel("Tant [K]")
ax.set_title(f"Total Tant vs azimuth  (alt=0, {freqs_mhz[f_mid]:.0f} MHz)")
ax.grid(alpha=0.3)

# 2. Total Tant waterfall — az vs alt, centre frequency
ax = axes[0, 1]
im = ax.pcolormesh(az_deg, alt_deg, T_total[:, :, f_mid].T,
                   shading="nearest", cmap="viridis")
plt.colorbar(im, ax=ax, label="Tant [K]")
ax.set_xlabel("Azimuth [deg]")
ax.set_ylabel("Altitude [deg]")
ax.set_title(f"Total Tant waterfall  ({freqs_mhz[f_mid]:.0f} MHz)")

# 3. Total Tant spectra — fixed alt=0, several az
ax = axes[0, 2]
for i_az in range(0, N_AZ, 4):
    ax.semilogy(freqs_mhz, T_total[i_az, 0, :],
                alpha=0.6, label=f"az={az_deg[i_az]:.0f}°")
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Tant [K]")
ax.set_title("Total Tant spectra  (alt=0, various az)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# 4. Total Tant vs azimuth (several alt angles, centre frequency)
ax = axes[1, 0]
for i_alt in range(0, N_ALT, 6):
    ax.plot(az_deg, T_total[:, i_alt, f_mid],
            label=f"alt={alt_deg[i_alt]:.0f}°")
ax.set_xlabel("Azimuth [deg]")
ax.set_ylabel("Tant [K]")
ax.set_title(f"Total Tant vs azimuth  ({freqs_mhz[f_mid]:.0f} MHz)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 5. Total Tant vs alt (fixed az=0, all frequencies)
ax = axes[1, 1]
for i_f in range(0, N_FREQ, 4):
    ax.plot(alt_deg, T_total[0, :, i_f],
            alpha=0.7, label=f"{freqs_mhz[i_f]:.0f} MHz")
ax.set_xlabel("Altitude [deg]")
ax.set_ylabel("Tant [K]")
ax.set_title("Total Tant vs alt  (az=0, all freqs)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# 6. Total Tant waterfall — log scale at centre frequency
ax = axes[1, 2]
im = ax.pcolormesh(az_deg, alt_deg, np.log10(T_total[:, :, f_mid].T + 1),
                   shading="nearest", cmap="plasma")
plt.colorbar(im, ax=ax, label="log10(Tant + 1) [K]")
ax.set_xlabel("Azimuth [deg]")
ax.set_ylabel("Altitude [deg]")
ax.set_title(f"Total Tant (log scale)  ({freqs_mhz[f_mid]:.0f} MHz)")

plt.tight_layout()
out = "notebooks/EIGSEP_Canyon_Demo.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
