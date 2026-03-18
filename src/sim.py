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

# SH FFT spin-sweep helpers — imported lazily to avoid hard dependency on healpy.Alm
def _sh_coupling_modes(beam_alm, sky_alm, lmax):
    """
    C_m = sum_{l>=m} b_lm * conj(t_lm)  for m = 0 .. lmax.

    Parameters
    ----------
    beam_alm, sky_alm : (n_alm,) complex128
        Spherical-harmonic coefficients (healpy convention, m >= 0 stored).
    lmax : int

    Returns
    -------
    C_pos : (lmax+1,) complex128
    """
    _, m_arr = healpy.Alm.getlm(lmax)
    product = beam_alm * np.conj(sky_alm)
    C_pos = np.zeros(lmax + 1, dtype=np.complex128)
    np.add.at(C_pos, m_arr, product)
    return C_pos


def _sh_fft_spin(C_pos, N_phi, beam_solid_angle):
    """
    Evaluate T_ant(phi_k) for k=0..N_phi-1 (phi_k = 2pi k / N_phi) via FFT.

    For rotations R_z(phi) about the z-axis the Wigner D-matrix is diagonal,
    D^l_{mm'}(R_z(phi)) = delta_{mm'} e^{-im phi}, so the beam-weighted
    integral reduces to a DFT:

        T_ant(phi_k) = (1/Omega_B) sum_m C_m e^{-i m phi_k}

    All N_phi orientations are evaluated via a single FFT instead of N_phi
    separate pixel-domain sums.

    Parameters
    ----------
    C_pos : (lmax+1,) complex128
        Coupling modes from :func:`_sh_coupling_modes`.
    N_phi : int
        Number of equally-spaced spin angles.
    beam_solid_angle : float
        Beam normalisation: integral(B dΩ) = sum_pix(B) * (4pi/npix).

    Returns
    -------
    T_ant : (N_phi,) float64
    """
    mmax = len(C_pos) - 1
    spectrum = np.zeros(N_phi, dtype=np.complex128)
    spectrum[0] = C_pos[0]
    m_hi = min(mmax, N_phi // 2)
    spectrum[1:m_hi + 1] = C_pos[1:m_hi + 1]
    spectrum[N_phi - m_hi:N_phi] = np.conj(C_pos[m_hi:0:-1])
    return np.fft.ifft(spectrum).real * N_phi / beam_solid_angle

try:
    import eigsep_terrain.reflectivity as etr
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
        conductivity = etr.conductivity_from_resistivity(self.resistivity_ohm_m)
        eta = etr.permittivity_from_conductivity(conductivity, freqs)
        gamma = etr.reflection_coefficient(eta, eta0=eta0)
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

@partial(jax.jit, static_argnums=(0, 5))
def _beam_sum(beam_nside, beam_map, sky_masked, crds_top, rot_ms,
              npix_sky=None):
    """
    Beam-weighted sum over the gridded (HEALPix) sky.

    The denominator (beam solid angle) is rotation-invariant for a full-sphere
    sky and is precomputed once, halving the number of reductions per
    orientation vs. computing it inside the scan loop.

    Parameters
    ----------
    beam_nside : int  (static)
    beam_map   : (npix_beam, nfreq)
    sky_masked : (npix_sky,  nfreq)  sky already multiplied by horizon+terrain mask
    crds_top   : (3, npix_sky)       topocentric pixel unit vectors
    rot_ms     : (n_orient, 3, 3)    topocentric → beam-frame rotations
    npix_sky   : int or None  (static)
        Number of sky pixels.  If None, inferred from sky_masked.shape[0].
        Used to scale the beam sum to match the sky-pixel sampling density.

    Returns
    -------
    num : (n_orient, nfreq)   beam-weighted sky sum
    den : (n_orient, nfreq)   beam solid-angle (repeated for each orientation)
    """
    # Denominator: sum_pix B(R @ n_pix) is rotation-invariant for full-sphere
    # sky.  It equals (npix_sky/npix_beam) * sum(beam_map, axis=0).
    npix_beam = beam_map.shape[0]
    if npix_sky is None:
        npix_sky = sky_masked.shape[0]
    den_row = jnp.sum(beam_map, axis=0) * (npix_sky / npix_beam)  # (nfreq,)

    def body(_, R):
        wgt = interpolate_map(beam_nside, beam_map, *(R @ crds_top))
        return None, jnp.sum(wgt * sky_masked, axis=0)

    _, num = jax.lax.scan(body, None, rot_ms)                     # (n_orient, nfreq)
    den = jnp.broadcast_to(den_row[None, :], num.shape)
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
    T_gnd : float
        Temperature [K] assigned to horizon-blocked pixels in both
        :meth:`sim` and :meth:`sky_map`.  Default 300.0.
    """

    def __init__(self, observer, freqs, beam, catalog=None, terrain=None,
                 nside=64, gsm=True, monopole=None, T_gnd=300.0):
        self.observer = observer
        self.freqs = np.asarray(freqs, dtype=real_dtype)
        self.beam = beam
        self.catalog = catalog
        self.terrain = terrain
        self.T_gnd = float(T_gnd)
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
    # Public sky accessor

    def sky_map(self, frame='gal', time=None, catalog=True, nside=None,
                channels=None, T_gnd=None, beam_weighted=False):
        """
        Return the sky model as a HEALPix map in the requested coordinate frame.

        When *time* is provided the same horizon + terrain mask (and ground
        temperature) used by :meth:`sim` is applied in galactic pixel space
        before resampling, so the visualised sky is identical to the one used
        in the simulation.

        Parameters
        ----------
        frame : {'gal', 'eq', 'top'}
            Output coordinate frame:
            - ``'gal'`` — galactic (the native storage frame)
            - ``'eq'``  — equatorial / ICRS
            - ``'top'`` — topocentric (observer body frame at *time*)
        time : `~astropy.time.Time` or str, optional
            Observation epoch.  Required for ``frame='top'``.  When provided,
            the horizon mask is applied and solar-system source positions are
            updated (if *catalog* is True).
        catalog : bool
            If True (default), bake catalog sources (fixed + solar-system)
            into the returned map.
        nside : int, optional
            Output HEALPix resolution.  Defaults to ``self.nside``.
        channels : int, slice, or array_like of int, optional
            Frequency channel selection (e.g. ``0``, ``slice(10, 20)``,
            ``[0, 32, 64]``).  Applied before resampling.  Defaults to all
            channels.
        T_gnd : float or None
            Ground temperature [K] used to fill horizon-blocked pixels.
            Overrides ``self.T_gnd`` for this call.  Only relevant when
            *time* is provided (masking requires observer position).
        beam_weighted : bool
            If True, multiply each output pixel by the beam response at its
            topocentric direction (nearest-pixel lookup).  Requires *time*.

        Returns
        -------
        sky : ndarray, shape (npix, nchans) or (npix,) for integer *channels*
            Sky brightness temperature in the requested frame.
        """
        if nside is None:
            nside = self.nside
        npix_out = healpy.nside2npix(nside)

        # Galactic base map (GSM + monopole)
        sky_gal = self._sky_gal
        if catalog and self.catalog is not None:
            if time is not None:
                self.catalog.update_positions(Time(time))
            sky_gal = sky_gal + self.catalog.convert_to_healpix().astype(real_dtype)

        # Resolve observer rotation once (needed for masking and/or frame='top')
        R_gal2top = None
        if time is not None:
            self.observer.set_time(Time(time))
            R_gal2top = self.observer.rot_gal2top().astype(real_dtype)

        # Rotation from the output frame to galactic (for pull-resampling)
        if frame == 'gal':
            R_out2gal = np.eye(3, dtype=real_dtype)
        elif frame == 'eq':
            from ._observer import ICRS2GAL
            R_out2gal = ICRS2GAL.astype(real_dtype)
        elif frame == 'top':
            if time is None:
                raise ValueError("sky_map requires time= when frame='top'")
            R_out2gal = R_gal2top.T                            # top → gal
        else:
            raise ValueError(f"unknown frame {frame!r}; choose 'gal', 'eq', or 'top'")

        # Apply horizon + terrain mask in galactic pixel space — same as sim().
        # Skipped when time is unknown (static sky visualisation).
        if time is not None:
            crds_top_gal = R_gal2top @ self._crds_gal         # (3, npix)
            sky_gal = self._masked_sky_gal(crds_top_gal, sky_gal, T_gnd)

        # Output pixel unit vectors and galactic lookup indices
        crds_out = np.array(healpy.pix2vec(nside, np.arange(npix_out)), dtype=real_dtype)
        gal_pix  = healpy.vec2pix(self.nside, *(R_out2gal @ crds_out))

        # Pull-resample with early channel slicing (avoids full-bandwidth alloc)
        sky_gal_ch = sky_gal if channels is None else sky_gal[:, channels]
        sky_out = sky_gal_ch[gal_pix]

        # Beam weighting: nearest-pixel lookup in beam map (output-pixel operation)
        if beam_weighted:
            if time is None:
                raise ValueError("time= is required for beam weighting")
            if frame == 'top':
                crds_top_out = crds_out
            else:
                crds_top_out = R_gal2top @ (R_out2gal @ crds_out)
            beam_map_ch = self.beam.map if channels is None else self.beam.map[:, channels]
            beam_pix = healpy.vec2pix(self.beam._nside, *crds_top_out)
            sky_out  = sky_out * beam_map_ch[beam_pix]

        return sky_out

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

    def _masked_sky_gal(self, crds_top, sky_gal, T_gnd=None):
        """
        Apply the horizon + terrain mask to *sky_gal* in galactic pixel space.

        Blocked pixels are replaced with *T_gnd* (defaults to ``self.T_gnd``).
        This is the shared masking step used by both :meth:`sim` and
        :meth:`sky_map` to guarantee identical effective sky maps.

        Parameters
        ----------
        crds_top : ndarray, shape (3, npix)
            Topocentric unit vectors for each galactic pixel.
        sky_gal : ndarray, shape (npix, nfreq)
            Galactic sky map (any number of frequency channels).
        T_gnd : float or None
            Ground temperature [K] to assign to blocked pixels.
            If None, uses ``self.T_gnd``.

        Returns
        -------
        sky_masked : ndarray, shape (npix, nfreq)
        """
        if T_gnd is None:
            T_gnd = self.T_gnd
        mask = self._horizon_mask(crds_top)          # (npix,) float32
        return sky_gal * mask[:, None] + T_gnd * (1.0 - mask[:, None])

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

            # Combined horizon + terrain mask → masked sky map (T_gnd fills blocked pixels)
            sky_masked = self._masked_sky_gal(crds_top, self._sky_gal)  # (npix, nfreq)

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
                                     crds_top_jax, rm_chunk,
                                     npix_sky=self.npix)
                if src_vecs_top.shape[1] > 0:
                    sv_jax = jnp.asarray(src_vecs_top, dtype=float_dtype)
                    sf_jax = jnp.asarray(src_flux, dtype=float_dtype)
                    num = num + _src_sum(beam_nside, beam_map,
                                         sv_jax, sf_jax, rm_chunk)
                t_ant[i:i + chunk_size] = np.asarray(num / den)

            vis[ti] = bandpass * (S12 * t_ant + Trx)

        return vis

    # ------------------------------------------------------------------
    # SH + FFT simulation helpers
    # ------------------------------------------------------------------

    def _sky_topocentric(self, R_gal2top, sky_gal=None):
        """
        Pull-resample the galactic sky to standard topocentric HEALPix pixels
        and apply the horizon mask.

        Parameters
        ----------
        R_gal2top : (3, 3) float32
            Galactic → topocentric rotation matrix.
        sky_gal : (npix, nfreq) float32, optional
            Galactic sky map to resample.  Defaults to ``self._sky_gal``.
            Pass a pre-augmented map (e.g. with baked-in catalog sources) to
            include additional components without mutating instance state.

        Returns
        -------
        sky_top_masked : (npix, nfreq) float32
            Sky temperature in topocentric pixel ordering with below-horizon
            pixels zeroed.
        """
        if sky_gal is None:
            sky_gal = self._sky_gal
        # Standard topocentric pixel unit vectors (3, npix)
        crds_top_std = np.array(
            healpy.pix2vec(self.nside, np.arange(self.npix)), dtype=real_dtype
        )
        # For each topocentric pixel j, find the galactic pixel it came from
        gal_dirs = R_gal2top.T @ crds_top_std          # R_top2gal applied
        gal_pix = healpy.vec2pix(self.nside, *gal_dirs)
        sky_top = sky_gal[gal_pix]                     # (npix, nfreq)
        # Horizon mask: above_horizon is indexed by galactic pixel; remap to
        # topocentric by looking up the galactic pixel for each topocentric pixel.
        mask_gal = self.observer.above_horizon(self.nside).astype(real_dtype)
        return sky_top * mask_gal[gal_pix, None]        # (npix, nfreq)

    def _beam_sh(self, lmax):
        """
        Precompute SH coefficients and solid angles for the beam.

        Returns
        -------
        beam_alms : list of (n_alm,) complex128, length nfreq
        beam_solid_angles : list of float, length nfreq
        """
        beam_map_np = self.beam.map      # (npix_beam, nfreq)
        npix_beam = beam_map_np.shape[0]
        beam_alms, beam_solid_angles = [], []
        for fi in range(self.nfreqs):
            bm = beam_map_np[:, fi].astype(np.float64)
            beam_alms.append(healpy.map2alm(bm, lmax=lmax,
                                            use_pixel_weights=False))
            beam_solid_angles.append(
                float(np.sum(bm) * (4.0 * np.pi / npix_beam))
            )
        return beam_alms, beam_solid_angles

    def _moon_surface_emission(self, sky_gal, gamma, T_regolith=200.0):
        """
        Galactic-frame emission map for Moon-disk pixels (thermal + reflected sky).

        For each pixel on the Moon's disc (as seen from the spacecraft), the
        emission is::

            T_pix(ν) = (1 − γ(ν)) · T_regolith + γ(ν) · T_sky(r̂_reflected, ν)

        where *r̂_reflected* is the direction of the sky that mirrors into the
        spacecraft's line of sight after reflecting off the local surface normal.

        Requires the observer to implement ``spacecraft_position()`` and
        ``above_horizon()`` (i.e. a :class:`~eigsep_sim.lunar_orbit.LunarOrbit`).

        Parameters
        ----------
        sky_gal : (npix, nfreq) float32
            Galactic sky map used for reflected-sky lookups.
        gamma : (nfreq,) float32
            Frequency-dependent surface amplitude reflectivity (0–1).
        T_regolith : float
            Mean lunar surface temperature [K].

        Returns
        -------
        emission : (npix, nfreq) float32
            Galactic-frame map; nonzero only at Moon-disk pixels.
        """
        from .const import R_MOON as _R_MOON

        moon_mask = ~self.observer.above_horizon(self.nside)  # True = on Moon disk
        emission = np.zeros((self.npix, self.nfreqs), dtype=real_dtype)
        if not moon_mask.any():
            return emission

        moon_pix = np.where(moon_mask)[0]
        moon_normals = self._crds_gal[:, moon_pix]   # (3, n_moon) outward normals

        # Vector from each surface point toward the spacecraft
        pos = self.observer.spacecraft_position()     # (3,) metres from Moon centre
        surface_pts = moon_normals * _R_MOON          # (3, n_moon) metres
        v_to_sc = pos[:, None] - surface_pts          # (3, n_moon)
        v_to_sc /= np.linalg.norm(v_to_sc, axis=0)   # normalise

        # Reflected viewing direction: r = 2(v·n)n − v
        dot_vn = np.einsum('ij,ij->j', v_to_sc, moon_normals)   # (n_moon,)
        refl = 2 * dot_vn[None, :] * moon_normals - v_to_sc     # (3, n_moon)

        # Sky temperature in the reflected directions (galactic frame lookup)
        refl_pix = healpy.vec2pix(self.nside, *refl)
        T_sky_refl = sky_gal[refl_pix]               # (n_moon, nfreq)

        gamma = np.asarray(gamma, dtype=real_dtype)
        T_surface = (1.0 - gamma[None, :]) * T_regolith + gamma[None, :] * T_sky_refl
        emission[moon_pix] = T_surface
        return emission

    # ------------------------------------------------------------------
    # SH + FFT spin-sweep simulation (z-axis only)
    # ------------------------------------------------------------------

    def sim_spin(self, times, n_phi, Trx=50.0, bandpass=1.0,
                 S11=0.0, lmax=None):
        """
        Fast spin-sweep simulation using spherical harmonics + FFT.

        For a pure rotation about the topocentric z-axis the Wigner D-matrix
        is diagonal, reducing the beam-weighted integral for all N_phi spin
        angles to a single FFT per frequency per timestep.  This is orders of
        magnitude faster than the pixel-domain loop in :meth:`sim`.

        For az+alt scans (e.g. a canyon antenna that tilts and spins) use
        :meth:`sim_azalt_sh` instead.

        Parameters
        ----------
        times : array_like of `~astropy.time.Time`
            Observation epochs.
        n_phi : int
            Number of equally-spaced spin angles over [0, 2π).
        Trx : float
            Receiver noise temperature [K].
        bandpass : float or array_like, shape (nfreq,)
            Bandpass response (multiplicative).
        S11 : float or array_like, shape (nfreq,)
            Reflection coefficient (power).  S12 = 1 − S11.
        lmax : int or None
            SH band-limit.  Defaults to 2 * nside.

        Returns
        -------
        vis : ndarray, shape (ntimes, n_phi, nfreq), float32
        """
        if lmax is None:
            lmax = 2 * self.nside

        times = list(times) if not isinstance(times, (list, np.ndarray)) else times
        ntimes = len(times)
        S12 = 1.0 - np.asarray(S11, dtype=real_dtype)
        bandpass = np.asarray(bandpass, dtype=real_dtype)

        vis = np.empty((ntimes, n_phi, self.nfreqs), dtype=real_dtype)
        beam_alms, beam_solid_angles = self._beam_sh(lmax)

        for ti, t in enumerate(tqdm.tqdm(times)):
            t = Time(t)
            self.observer.set_time(t)
            R_gal2top = self.observer.rot_gal2top().astype(real_dtype)

            # Sky in topocentric HEALPix pixels (pull resample + mask)
            sky_top = self._sky_topocentric(R_gal2top)  # (npix, nfreq)

            t_ant = np.zeros((n_phi, self.nfreqs), dtype=real_dtype)
            for fi in range(self.nfreqs):
                sky_alm = healpy.map2alm(sky_top[:, fi].astype(np.float64),
                                         lmax=lmax, use_pixel_weights=False)
                C_pos = _sh_coupling_modes(beam_alms[fi], sky_alm, lmax)
                t_ant[:, fi] = _sh_fft_spin(
                    C_pos, n_phi, beam_solid_angles[fi]
                ).astype(real_dtype)

            vis[ti] = bandpass * (S12 * t_ant + Trx)

        return vis

    # ------------------------------------------------------------------
    # SH + FFT az+alt simulation (canyon antenna)
    # ------------------------------------------------------------------

    def sim_azalt_sh(self, times, alts_rad, n_phi, east_vec=None,
                     Trx=50.0, bandpass=1.0, S11=0.0, lmax=None):
        """
        Fast az+alt simulation using SH rotation + FFT.

        For rotations R = R_z(az) · R_east(alt) the beam-weighted integral
        reduces to one Wigner sky-rotation (O(lmax³)) followed by an FFT
        (O(n_phi log n_phi)) per altitude per frequency, instead of n_phi
        separate pixel-domain sums.

        **Algorithm** — per timestep
          1. Pull-resample galactic sky to topocentric HEALPix pixels, apply
             horizon mask.  Cost: O(npix).
          2. SH decompose the masked sky per frequency.  Cost: nfreq × map2alm.
          3. For each altitude: rotate sky alm by R_east(+alt) using Wigner
             D-matrices (healpy.rotate_alm), compute coupling modes C_m, FFT
             over azimuth.  Cost: N_alt × nfreq × O(lmax³).
          4. Total per timestep ≈ nfreq × (1 + N_alt) × map2alm, vs.
             N_alt × n_phi × npix × nfreq for the pixel-domain approach.

        **Accuracy** — the SH approximation is accurate for maps that are
        bandlimited at *lmax* = 2 × nside.  Sharp horizon features may
        require a larger *lmax* or prior smoothing.

        **Further speedup** (not yet implemented) — for each altitude the
        Wigner D-matrices are the same for all frequencies; applying them via
        precomputed O(lmax³) block matrices across all frequencies at once
        (batch BLAS) would reduce the per-altitude cost to O(lmax³) instead
        of nfreq × O(lmax³).

        Parameters
        ----------
        times : array_like of `~astropy.time.Time`
            Observation epochs.
        alts_rad : array_like, shape (N_alt,)
            Altitude angles [radians] of the beam boresight.
        n_phi : int
            Number of equally-spaced azimuth angles over [0, 2π).
        east_vec : array_like, shape (3,), optional
            Altitude rotation axis in the topocentric frame.  Defaults to
            [1, 0, 0] (topocentric east / x-axis).  Provide the exact axis
            for non-ideal suspension geometry.
        Trx : float
            Receiver noise temperature [K].
        bandpass : float or array_like, shape (nfreq,)
            Bandpass response (multiplicative).
        S11 : float or array_like, shape (nfreq,)
            Reflection coefficient (power).  S12 = 1 − S11.
        lmax : int or None
            SH band-limit.  Defaults to 2 * nside.

        Returns
        -------
        vis : ndarray, shape (ntimes, N_alt, n_phi, nfreq), float32
            Simulated antenna temperature [K] after bandpass and S11.
        """
        from .coord import rot_m as _rot_m

        if lmax is None:
            lmax = 2 * self.nside
        if east_vec is None:
            east_vec = np.array([1.0, 0.0, 0.0])
        east_vec = np.asarray(east_vec, dtype=np.float64)
        east_vec /= np.linalg.norm(east_vec)

        alts_rad = np.asarray(alts_rad, dtype=np.float64)
        N_alt = len(alts_rad)

        times = list(times) if not isinstance(times, (list, np.ndarray)) else times
        ntimes = len(times)
        S12 = 1.0 - np.asarray(S11, dtype=real_dtype)
        bandpass = np.asarray(bandpass, dtype=real_dtype)

        vis = np.empty((ntimes, N_alt, n_phi, self.nfreqs), dtype=real_dtype)

        # Precompute beam SH and rotation matrices for each altitude
        beam_alms, beam_solid_angles = self._beam_sh(lmax)
        # R_east(+alt) rotation matrices — one per altitude
        R_alts = [_rot_m(float(alt), east_vec) for alt in alts_rad]

        for ti, t in enumerate(tqdm.tqdm(times)):
            t = Time(t)
            self.observer.set_time(t)
            R_gal2top = self.observer.rot_gal2top().astype(real_dtype)

            # Sky in topocentric HEALPix pixels (pull resample + mask)
            sky_top = self._sky_topocentric(R_gal2top)  # (npix, nfreq)

            # SH decompose sky per frequency (done once per timestep)
            sky_alms = [
                healpy.map2alm(sky_top[:, fi].astype(np.float64),
                               lmax=lmax, use_pixel_weights=False)
                for fi in range(self.nfreqs)
            ]

            # For each altitude: rotate sky alm → coupling modes → FFT
            for ai, (alt, R_alt) in enumerate(zip(alts_rad, R_alts)):
                t_ant = np.zeros((n_phi, self.nfreqs), dtype=real_dtype)
                for fi in range(self.nfreqs):
                    # Rotate sky by R_east(+alt): sky_rot(n) = sky(R_east(-alt) n)
                    sky_alm_rot = sky_alms[fi].copy()
                    healpy.rotate_alm(sky_alm_rot, matrix=R_alt)
                    C_pos = _sh_coupling_modes(beam_alms[fi], sky_alm_rot, lmax)
                    t_ant[:, fi] = _sh_fft_spin(
                        C_pos, n_phi, beam_solid_angles[fi]
                    ).astype(real_dtype)
                vis[ti, ai] = bandpass * (S12 * t_ant + Trx)

        return vis

    # ------------------------------------------------------------------
    # Orbital sweep (LunarOrbit)
    # ------------------------------------------------------------------

    def sim_orbit_spin(self, n_orbit, n_phi, time=None, Trx=50.0,
                       bandpass=1.0, S11=0.0, lmax=None,
                       T_regolith=200.0, terrain_type=None):
        """
        Simulate over one full circular orbit with a spin sweep at each
        orbital position.

        Iterates over *n_orbit* equally-spaced orbital phases in [0, 2π)
        by calling ``observer.set_phases(th_orbit)`` at each step, then
        performs the SH+FFT spin sweep for the *n_phi* spin angles.

        **Occultation**: the observer's ``above_horizon`` mask (lunar
        occultation from :class:`~eigsep_sim.lunar_orbit.LunarOrbit`) is
        applied automatically at each orbital position, zeroing sky pixels
        blocked by the Moon.

        **Lunar surface emission**: if *resistivity_ohm_m* is given, blocked
        Moon-disk pixels are replaced with thermal emission plus reflected sky
        via :meth:`_moon_surface_emission`.

        **Catalog sources**: Sun, Earth, and fixed point sources are baked into
        the sky map once at *time* before the orbit loop begins.  Orbital
        periods (~2 h) are much shorter than the timescale on which
        solar-system positions change, so this snapshot is a good
        approximation.

        Requires the observer to implement ``set_phases(th_orbit, th_spin=0)``
        — see :class:`~eigsep_sim.lunar_orbit.LunarOrbit`.

        Parameters
        ----------
        n_orbit : int
            Number of equally-spaced orbital positions over [0, 2π).
        n_phi : int
            Number of equally-spaced spin angles over [0, 2π).
        time : `~astropy.time.Time` or str, optional
            Epoch at which solar-system source positions are fixed.  Required
            if a catalog with solar-system sources was supplied.
        Trx : float
            Receiver noise temperature [K].
        bandpass : float or array_like, shape (nfreq,)
            Bandpass response (multiplicative).
        S11 : float or array_like, shape (nfreq,)
            Reflection coefficient (power).  S12 = 1 − S11.
        lmax : int or None
            SH band-limit.  Defaults to 2 * nside.
        T_regolith : float
            Mean lunar surface temperature [K] for thermal emission.
            Only used when *terrain_type* is set.
        terrain_type : str or None
            Named terrain type from ``eigsep_terrain.reflectivity.TERRAIN_TYPES``
            (e.g. ``'lunar_regolith'``).  When given, Moon-disk pixels receive
            thermal emission + reflected sky weighted by the terrain's
            frequency-dependent reflectivity (accounting for both eps_r and
            resistivity).  When None, Moon-disk pixels remain zeroed.

        Returns
        -------
        vis : ndarray, shape (n_orbit, n_phi, nfreq), float32
            Simulated antenna temperature [K].
        """
        if not hasattr(self.observer, 'set_phases'):
            raise TypeError(
                f"{type(self.observer).__name__} does not implement set_phases; "
                "use LunarOrbit or another observer that supports orbital phases"
            )
        if lmax is None:
            lmax = 2 * self.nside

        S12 = 1.0 - np.asarray(S11, dtype=real_dtype)
        bandpass = np.asarray(bandpass, dtype=real_dtype)

        # Snapshot sky: GSM + monopole + catalog sources baked in at `time`
        sky_gal = self._sky_gal
        if self.catalog is not None:
            if time is not None:
                self.catalog.update_positions(Time(time))
            src_map = self.catalog.convert_to_healpix().astype(real_dtype)
            sky_gal = sky_gal + src_map

        # Lunar surface reflectivity (optional)
        gamma = None
        if terrain_type is not None:
            if not _HAS_TERRAIN:
                raise ImportError("eigsep_terrain is required for terrain reflectivity")
            gamma = np.abs(
                etr.terrain_reflection_coefficient(
                    terrain_type, self.freqs.astype(np.float64)
                )
            ).astype(real_dtype)

        # Topocentric pixel unit vectors — constant across orbital positions
        crds_top_std = np.array(
            healpy.pix2vec(self.nside, np.arange(self.npix)), dtype=real_dtype
        )  # (3, npix)

        vis = np.empty((n_orbit, n_phi, self.nfreqs), dtype=real_dtype)
        beam_alms, beam_solid_angles = self._beam_sh(lmax)
        th_orbits = np.linspace(0.0, 2 * np.pi, n_orbit, endpoint=False)

        for oi, th_orbit in enumerate(tqdm.tqdm(th_orbits)):
            self.observer.set_phases(th_orbit)
            R_gal2top = self.observer.rot_gal2top().astype(real_dtype)
            sky_top = self._sky_topocentric(R_gal2top, sky_gal=sky_gal)

            # Add Moon surface emission to previously-zeroed Moon-disk pixels
            if gamma is not None:
                moon_em_gal = self._moon_surface_emission(sky_gal, gamma, T_regolith)
                gal_pix = healpy.vec2pix(self.nside, *(R_gal2top.T @ crds_top_std))
                sky_top = sky_top + moon_em_gal[gal_pix]

            t_ant = np.zeros((n_phi, self.nfreqs), dtype=real_dtype)
            for fi in range(self.nfreqs):
                sky_alm = healpy.map2alm(
                    sky_top[:, fi].astype(np.float64),
                    lmax=lmax, use_pixel_weights=False,
                )
                C_pos = _sh_coupling_modes(beam_alms[fi], sky_alm, lmax)
                t_ant[:, fi] = _sh_fft_spin(
                    C_pos, n_phi, beam_solid_angles[fi]
                ).astype(real_dtype)

            vis[oi] = bandpass * (S12 * t_ant + Trx)

        return vis
