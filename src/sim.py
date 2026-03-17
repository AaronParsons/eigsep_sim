"""
Global-signal radiometer simulation.

Architecture overview
---------------------
Sky sources live in the *galactic* frame (consistent with the GSM):

  sky_gal   : (npix, nfreq)  -- GSM + optional monopole, static HEALPix map
  fixed srcs : (N, 3) + (N, nfreq)  -- extragalactic catalog, galactic vecs, static
  SS srcs    : (M, 3) + (M, nfreq)  -- solar-system bodies, galactic vecs, per-timestep

Per timestep the galactic→topocentric rotation R = observer.rot_gal2top() is
computed *once in NumPy*.  The JAX kernel then applies beam-orientation
rotations (az/alt) in a lax.scan loop, interpolating the beam at each
topocentric pixel/source direction and forming the beam-weighted sum.

No sky resampling is done per timestep; the only rotation applied inside JAX
is the cheap (3, 3) @ (3, N) beam-orientation multiply.
"""

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.interpolate import interp1d
import healpy
from astropy.time import Time
from pygdsm import GlobalSkyModel16 as GSM16
import tqdm

from .healpix import HPM, float_dtype, interpolate_map
from .beam import Beam

try:
    import eigsep_terrain.utils as etu
    _HAS_TERRAIN = True
except ImportError:
    _HAS_TERRAIN = False

real_dtype = np.float32
complex_dtype = np.complex64

_DATA = os.path.join(os.path.dirname(__file__), 'data')
TERRAIN_NPZ = os.path.join(_DATA, 'horizon_models_v000.npz')
BANDPASS_NPZ = os.path.join(_DATA, 'bandpass.npz')
S11_NPZ = os.path.join(_DATA, 'S11_eigsep_bowtie_v000.npz')

DEFAULT_RESISTIVITY = 3e2  # Ohm m


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

class Terrain(HPM):
    """
    Horizon model stored as a HEALPix map in the topocentric frame.

    Pixels where the line-of-sight is open (above terrain) hold NaN.
    Pixels blocked by terrain hold the terrain reflectivity (0–1).

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].  Used for transmitter bookkeeping.
    height : float
        Antenna height above ground [m].  Selects the closest pre-computed
        model from the NPZ file.
    filename : str
        Path to the terrain NPZ file.
    resistivity_ohm_m : float
        Soil resistivity used for reflectivity calculations.
    transmitters : list of (vec, fqs, pwrs)
        RF transmitters at fixed topocentric unit vectors *vec* with powers
        *pwrs* [K equivalent] at frequencies *fqs* [Hz].
    """

    def __init__(self, freqs, height=114, filename=TERRAIN_NPZ,
                 resistivity_ohm_m=DEFAULT_RESISTIVITY, transmitters=()):
        with np.load(filename) as npz:
            nside_horizon = int(npz['nside'])
            heights = npz['heights']
            i = np.argmin(np.abs(heights - height))
            r = npz['r'][i].astype(real_dtype)
        HPM.__init__(self, nside_horizon, interp=False)
        self.freqs = np.asarray(freqs, dtype=real_dtype)
        self.set_map(r)
        self.resistivity_ohm_m = resistivity_ohm_m
        self.set_transmitters(transmitters)

    def set_transmitters(self, transmitters):
        n_tx = len(transmitters)
        self.tx_vecs_top = np.empty((3, n_tx), dtype=real_dtype)
        self.tx_flux = np.zeros((n_tx, self.freqs.size), dtype=real_dtype)
        for cnt, (vec, fqs, pwrs) in enumerate(transmitters):
            self.tx_vecs_top[:, cnt] = vec
            chs = np.searchsorted(self.freqs, fqs)
            self.tx_flux[cnt, chs] = pwrs

    def get_mask(self, crds_top):
        """
        Float mask (1.0 = open sky, 0.0 = terrain-blocked) at topocentric
        directions *crds_top* (3, N).

        Parameters
        ----------
        crds_top : ndarray, shape (3, N)

        Returns
        -------
        mask : ndarray, shape (N,), float32
        """
        tx, ty, tz = crds_top
        vals = self[tx, ty, tz]           # NaN where sky is open
        return np.isnan(np.asarray(vals)).astype(real_dtype)

    def reflectivity(self, freqs, eta0=1):
        if not _HAS_TERRAIN:
            raise ImportError("eigsep_terrain is required for reflectivity calculations")
        conductivity = etu.conductivity_from_resistivity(self.resistivity_ohm_m)
        eta = etu.permittivity_from_conductivity(conductivity, freqs)
        gamma = etu.reflection_coefficient(eta, eta0=eta0)
        return gamma.astype(real_dtype)


# ---------------------------------------------------------------------------
# Instrument loaders
# ---------------------------------------------------------------------------

def load_bandpass(freqs, filename=BANDPASS_NPZ):
    """Load and interpolate a bandpass model onto *freqs* [Hz]."""
    npz = np.load(filename)
    return interp1d(npz['freqs'], npz['bandpass'], kind='cubic',
                    fill_value=0, bounds_error=False)(freqs).astype(real_dtype)


def load_S11(freqs, filename=S11_NPZ, termination=None):
    """Load and interpolate an S11 model onto *freqs* [Hz]."""
    npz = np.load(filename)
    if termination is None:
        mdl = npz['S11']
    else:
        Z = npz['Z']
        mdl = (np.abs(Z - termination) / np.abs(Z + termination)) ** 2
    return interp1d(npz['freqs'], mdl, kind='cubic',
                    fill_value=0, bounds_error=False)(freqs).astype(real_dtype)


# ---------------------------------------------------------------------------
# JAX beam-integration kernels
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def _beam_sum(beam_nside, beam_map, sky_masked, crds_top, rot_ms):
    """
    Beam-weighted sum over the gridded (HEALPix) sky.

    Parameters
    ----------
    beam_nside : int  (static)
    beam_map   : (npix_beam, nfreq)
    sky_masked : (npix_sky,  nfreq)  sky already multiplied by horizon+terrain mask
    crds_top   : (3, npix_sky)       topocentric pixel unit vectors
    rot_ms     : (n_orient, 3, 3)    topocentric → beam-frame rotations

    Returns
    -------
    num : (n_orient, nfreq)   beam-weighted sky sum
    den : (n_orient, nfreq)   beam solid-angle sum (normalisation)
    """
    def body(_, R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ crds_top))
        return None, (jnp.sum(wgt * sky_masked, axis=0),
                      jnp.sum(wgt, axis=0))
    _, (num, den) = jax.lax.scan(body, None, rot_ms)
    return num, den


@partial(jax.jit, static_argnums=(0,))
def _src_sum(beam_nside, beam_map, src_vecs_top, src_flux, rot_ms):
    """
    Beam-weighted sum over discrete (point/compact) sources.

    Parameters
    ----------
    beam_nside   : int  (static)
    beam_map     : (npix_beam, nfreq)
    src_vecs_top : (3, Nsrc)          topocentric source unit vectors
    src_flux     : (Nsrc, nfreq)      effective brightness temperature [K/pixel]
    rot_ms       : (n_orient, 3, 3)

    Returns
    -------
    num : (n_orient, nfreq)   additional beam-weighted source contribution
    """
    def body(_, R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ src_vecs_top))
        return None, jnp.sum(wgt * src_flux, axis=0)
    _, num = jax.lax.scan(body, None, rot_ms)
    return num


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Global-signal radiometer simulator.

    Parameters
    ----------
    observer : Observer
        An EarthSurface, LunarSurface, or LunarOrbit instance.
    freqs : array_like
        Frequency array [Hz].
    beam : Beam
        Antenna beam pattern.
    catalog : SourceCatalog, optional
        Source catalog providing fixed sources and solar-system bodies.
        Fixed sources are treated as discrete directions (no pixelisation).
        Solar-system bodies are updated each timestep.
    terrain : Terrain, optional
        Horizon/terrain model.  If None, the horizon is the geometric
        half-space (z > 0 in topocentric frame) from observer.above_horizon().
    nside : int
        HEALPix resolution for the GSM sky map.
    gsm : bool
        Include the Global Sky Model 2016.
    monopole : array_like, shape (nfreq,), optional
        Isotropic monopole temperature [K] to add.
    """

    def __init__(self, observer, freqs, beam, catalog=None, terrain=None,
                 nside=64, gsm=True, monopole=None):
        self.observer = observer
        self.freqs = np.asarray(freqs, dtype=real_dtype)
        self.beam = beam
        self.catalog = catalog
        self.terrain = terrain
        self.nside = nside
        self.npix = healpy.nside2npix(nside)

        # Galactic pixel unit vectors — static throughout the simulation
        self._crds_gal = np.array(
            healpy.pix2vec(nside, np.arange(self.npix)), dtype=real_dtype
        )  # (3, npix)

        # GSM sky map in galactic frame — static
        self._sky_gal = np.zeros((self.npix, self.nfreqs), dtype=real_dtype)
        if gsm:
            self._load_gsm()
        if monopole is not None:
            self._sky_gal += np.asarray(monopole, dtype=real_dtype)[None, :]

        # Fixed (static galactic) discrete sources from catalog.
        # Shape (3, N_fixed) and (N_fixed, nfreq).
        # Separated from SS sources so they don't need ephemeris updates.
        self._fixed_vecs_gal = np.empty((3, 0), dtype=real_dtype)
        self._fixed_flux = np.empty((0, self.nfreqs), dtype=real_dtype)
        if catalog is not None and len(catalog._crd_gal) > 0:
            self._fixed_vecs_gal = catalog._crd_gal.T.astype(real_dtype)  # (3, N)
            self._fixed_flux = catalog._Tsrc.astype(real_dtype)           # (N, nfreq)

    # ------------------------------------------------------------------
    # Properties

    @property
    def nfreqs(self):
        return self.freqs.size

    # ------------------------------------------------------------------
    # Setup

    def _load_gsm(self):
        gsm = GSM16(freq_unit='Hz', data_unit='TRJ',
                    resolution='lo', include_cmb=True)
        gsm_data = gsm.generate(self.freqs).astype(real_dtype).T  # (npix_gsm, nfreq)
        gsm_nside = healpy.npix2nside(gsm_data.shape[0])
        if gsm_nside != self.nside:
            gsm_hpm = HPM(nside=gsm_nside, interp=True)
            gsm_hpm.set_map(gsm_data)
            x, y, z = self._crds_gal
            gsm_data = np.asarray(gsm_hpm[x, y, z])
        self._sky_gal += gsm_data

    # ------------------------------------------------------------------
    # Per-timestep helpers

    def _ss_vecs_and_flux(self):
        """
        Return current solar-system source vectors (galactic, (3, M)) and
        effective temperatures ((M, nfreq)) after calling catalog.update_positions().

        Effective temperature = disc_brightness_temperature × (disc_area / pixel_area),
        which gives K-per-pixel units consistent with the gridded sky map and
        fixed catalog sources.
        """
        if self.catalog is None or not self.catalog._ss_sources:
            return (np.empty((3, 0), dtype=real_dtype),
                    np.empty((0, self.nfreqs), dtype=real_dtype))
        pixel_area = 4.0 * np.pi / self.npix
        vecs, fluxes = [], []
        for src in self.catalog._ss_sources:
            if src.crd_gal is None:
                continue
            T = src.temperature(self.freqs).astype(real_dtype)
            r = src.angular_radius()
            T_eff = T * (np.pi * r * r / pixel_area) if r > 0 else T
            vecs.append(src.crd_gal)
            fluxes.append(T_eff)
        if not vecs:
            return (np.empty((3, 0), dtype=real_dtype),
                    np.empty((0, self.nfreqs), dtype=real_dtype))
        return (np.array(vecs, dtype=real_dtype).T,      # (3, M)
                np.array(fluxes, dtype=real_dtype))       # (M, nfreq)

    def _horizon_mask(self, crds_top):
        """
        Float mask (1.0 = visible, 0.0 = blocked) for all sky pixels.

        For surface observers the terrain mask (if provided) is combined with
        the geometric horizon.  For orbital observers only the occultation
        geometry is used (above_horizon already encodes this).

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix)

        Returns
        -------
        mask : ndarray, shape (npix,), float32
        """
        # Geometric horizon / occultation — works for all observer types
        geo_mask = self.observer.above_horizon(self.nside).astype(real_dtype)

        if self.terrain is not None:
            terrain_mask = self.terrain.get_mask(crds_top)
            return geo_mask * terrain_mask
        return geo_mask

    def _all_src_vecs_flux_top(self, R, ss_vecs_gal, ss_flux):
        """
        Assemble all discrete sources in the topocentric frame and apply a
        simple horizon check (topocentric z > 0).  Concatenates:
          - fixed catalog sources  (static galactic vecs → topocentric)
          - SS bodies              (pre-rotated to topocentric)
          - terrain transmitters   (already in topocentric)

        Returns
        -------
        src_vecs_top : (3, Nsrc)
        src_flux     : (Nsrc, nfreq)
        """
        parts_v, parts_f = [], []

        # Fixed extragalactic / stellar sources
        if self._fixed_vecs_gal.shape[1] > 0:
            parts_v.append(R @ self._fixed_vecs_gal)
            parts_f.append(self._fixed_flux)

        # Solar-system bodies
        if ss_vecs_gal.shape[1] > 0:
            parts_v.append(R @ ss_vecs_gal)
            parts_f.append(ss_flux)

        # Terrain transmitters (already topocentric)
        if self.terrain is not None and self.terrain.tx_vecs_top.shape[1] > 0:
            parts_v.append(self.terrain.tx_vecs_top)
            parts_f.append(self.terrain.tx_flux)

        if not parts_v:
            return (np.empty((3, 0), dtype=real_dtype),
                    np.empty((0, self.nfreqs), dtype=real_dtype))

        vecs = np.concatenate(parts_v, axis=1)    # (3, Nsrc)
        flux = np.concatenate(parts_f, axis=0)    # (Nsrc, nfreq)

        # Zero out sources below the geometric horizon
        above = (vecs[2] > 0).astype(real_dtype)  # topocentric z > 0
        flux = flux * above[:, None]

        return vecs, flux

    # ------------------------------------------------------------------
    # Main simulation loop

    def sim(self, times, azalts=None, Trx=50.0, bandpass=1.0, S11=0.0,
            chunk_size=16):
        """
        Run the simulation over *times* and antenna orientations *azalts*.

        Parameters
        ----------
        times : array_like of `~astropy.time.Time`
            Observation epochs.
        azalts : ndarray, shape (n_orient, 2) or (ntimes, n_orient, 2), optional
            Azimuth and altitude [radians] of the beam boresight for each
            orientation.  If 2-D, the same orientations are used at every
            timestep.  Defaults to a single zenith-pointing beam (az=alt=0).
        Trx : float
            Receiver noise temperature [K].
        bandpass : float or array_like, shape (nfreq,)
            Bandpass response (multiplicative).
        S11 : float or array_like, shape (nfreq,)
            Reflection coefficient (power).  S12 = 1 − S11.
        chunk_size : int
            Number of beam orientations processed per JAX lax.scan call.
            Tune to balance memory and JIT overhead.

        Returns
        -------
        vis : ndarray, shape (ntimes, n_orient, nfreq), float32
            Simulated antenna temperature [K] after bandpass and S11.
        """
        times = list(times) if not isinstance(times, (list, np.ndarray)) else times
        ntimes = len(times)
        S12 = 1.0 - np.asarray(S11, dtype=real_dtype)
        bandpass = np.asarray(bandpass, dtype=real_dtype)

        if azalts is None:
            azalts = np.zeros((1, 2), dtype=real_dtype)
        azalts = np.asarray(azalts, dtype=real_dtype)
        if azalts.ndim == 2:
            azalts = np.broadcast_to(azalts, (ntimes, *azalts.shape))
        n_orient = azalts.shape[1]

        vis = np.empty((ntimes, n_orient, self.nfreqs), dtype=real_dtype)

        # Pre-fetch static JAX arrays
        beam_map = jnp.asarray(self.beam.map, dtype=float_dtype)
        beam_nside = self.beam._nside
        crds_gal_jax = jnp.asarray(self._crds_gal, dtype=float_dtype)

        for ti, t in enumerate(tqdm.tqdm(times)):
            t = Time(t)

            # Update solar-system source positions
            if self.catalog is not None:
                self.catalog.update_positions(t)
            self.observer.set_time(t)

            # Galactic → topocentric rotation (cheap NumPy, done once per timestep)
            R = self.observer.rot_gal2top().astype(real_dtype)  # (3, 3)
            crds_top = R @ self._crds_gal                        # (3, npix)

            # Combined horizon + terrain mask → masked sky map
            mask = self._horizon_mask(crds_top)                  # (npix,)
            sky_masked = self._sky_gal * mask[:, None]           # (npix, nfreq)

            # Discrete sources in topocentric frame
            ss_vecs_gal, ss_flux = self._ss_vecs_and_flux()
            src_vecs_top, src_flux = self._all_src_vecs_flux_top(
                R, ss_vecs_gal, ss_flux
            )

            # Beam orientation rotation matrices for this timestep
            rot_ms = self.beam.get_rotation_matrices(
                azalts[ti, :, 0], azalts[ti, :, 1]
            )  # (n_orient, 3, 3)

            # Convert to JAX
            sky_jax = jnp.asarray(sky_masked, dtype=float_dtype)
            crds_top_jax = jnp.asarray(crds_top, dtype=float_dtype)
            rot_ms_jax = jnp.asarray(rot_ms, dtype=float_dtype)

            # Beam-weighted sum — chunked to control peak memory
            t_ant = np.zeros((n_orient, self.nfreqs), dtype=real_dtype)
            for i in range(0, n_orient, chunk_size):
                rm_chunk = rot_ms_jax[i:i + chunk_size]
                num, den = _beam_sum(beam_nside, beam_map, sky_jax,
                                     crds_top_jax, rm_chunk)
                if src_vecs_top.shape[1] > 0:
                    sv_jax = jnp.asarray(src_vecs_top, dtype=float_dtype)
                    sf_jax = jnp.asarray(src_flux, dtype=float_dtype)
                    num = num + _src_sum(beam_nside, beam_map,
                                         sv_jax, sf_jax, rm_chunk)
                t_ant[i:i + chunk_size] = np.asarray(num / den)

            vis[ti] = bandpass * (S12 * t_ant + Trx)

        return vis
