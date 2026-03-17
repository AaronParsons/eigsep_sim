"""
Radio source catalog and sky map utilities.

Provides two categories of sources:
  - Fixed sources: extragalactic or stellar, constant galactic-frame positions,
    added manually or queried from online catalogs (e.g. 3CR via Vizier).
  - Solar system sources: time-varying positions computed via astropy ephemeris
    (Sun, Moon, Earth, planets).  Positions are computed relative to a
    configurable observer body.

All HEALPix maps produced here are in the galactic coordinate frame, consistent
with the Global Sky Model used in SkyModel.
"""

import numpy as np
import healpy as hp
import astropy.units as u
from astropy.constants import c, k_B
from astropy.time import Time
from astropy.coordinates import (
    get_body_barycentric, SkyCoord, CartesianRepresentation, FK5
)
from astroquery.vizier import Vizier

from .const import R_SUN, R_MOON, R_EARTH


# ---------------------------------------------------------------------------
# Precision control

PRECISION = 1

if PRECISION == 1:
    real_dtype = np.float32
    complex_dtype = np.complex64
else:
    assert PRECISION == 2
    real_dtype = np.float64
    complex_dtype = np.complex128


# ---------------------------------------------------------------------------
# Coordinate helpers

def radec_to_eqvec(ra_rad, dec_rad):
    """Convert RA/Dec (radians) to ICRS Cartesian unit vector(s)."""
    th_rad = np.pi / 2 - dec_rad
    return hp.ang2vec(th_rad, ra_rad)


def eqvec_to_pix(nside, crd_eq, r_rad=None):
    """
    Map an ICRS unit vector to HEALPix pixel index/indices.

    Parameters
    ----------
    nside : int
    crd_eq : array_like, shape (..., 3)
        ICRS unit vector(s).
    r_rad : float, optional
        If given, return all pixels within this angular radius.
    """
    if r_rad is not None:
        return hp.query_disc(nside, crd_eq, r_rad, inclusive=True)
    return hp.vec2pix(nside, crd_eq[..., 0], crd_eq[..., 1], crd_eq[..., 2])


def skycoords_to_eqvec(crds, t_astropy):
    """Convert SkyCoord to ICRS unit vectors precessed to epoch t_astropy."""
    crds_now = crds.transform_to(FK5(equinox=t_astropy))
    ra = crds_now.ra.to(u.rad).value
    dec = crds_now.dec.to(u.rad).value
    return radec_to_eqvec(ra, dec)


def skycoords_to_galvec(skycoords):
    """Convert SkyCoord to galactic-frame unit vectors (shape (N, 3))."""
    gal = skycoords.galactic.cartesian
    xyz = np.column_stack([gal.x.value, gal.y.value, gal.z.value])
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def Jy2K_nside(nside, freqs_Hz, F_Jy=1.0):
    """Conversion factor from Jy to brightness temperature for an nside map."""
    F = F_Jy * u.Jy
    nu = freqs_Hz * u.Hz
    omega_pix = hp.nside2pixarea(nside)
    T_K = (c**2 / (2 * k_B * nu**2) * F / omega_pix).to(u.K)
    return T_K.value


# ---------------------------------------------------------------------------
# Disk-pixel overlap (from sun.py)

def disc_overlap_fraction(nside, crd, r_rad, k=3):
    """
    Area-weighted overlap of a uniform circular disc with HEALPix pixels.

    Parameters
    ----------
    nside : int
    crd : array_like, shape (3,)
        Unit vector toward the disc centre (any coordinate frame, must match
        the map frame).
    r_rad : float
        Disc angular radius, radians.
    k : int
        Sub-pixel refinement level (overlap estimated at nside * 2**k).

    Returns
    -------
    ipix_ring : ndarray of int
        Ring-scheme pixel indices that overlap the disc (frac > 0).
    frac : ndarray of float32
        Fractional overlap for each returned pixel, normalised to sum to 1.
    """
    ipix = hp.query_disc(nside, crd, r_rad, inclusive=True, nest=True)
    nside_hi = nside * (2 ** k)
    ipix_hi_disc = np.sort(
        hp.query_disc(nside_hi, crd, r_rad, inclusive=True, nest=True)
    )
    four_k = 4 ** k
    frac = np.empty(ipix.size, dtype=np.float32)
    for i, p in enumerate(ipix):
        start = p * four_k
        stop = start + four_k
        left = np.searchsorted(ipix_hi_disc, start, side='left')
        right = np.searchsorted(ipix_hi_disc, stop, side='left')
        frac[i] = (right - left) / float(four_k)
    frac /= np.sum(frac)
    ipix_ring = hp.nest2ring(nside, ipix[frac > 0])
    return ipix_ring, frac[frac > 0]


# ---------------------------------------------------------------------------
# Solar system source

# Default radio brightness temperatures and spectral parameters
_SS_DEFAULTS = {
    # name        : (radius_m, T0_K,  fq0_Hz,  alpha)
    'sun'         : (R_SUN,   1e6,   65e6,   0.3),
    'moon'        : (R_MOON,  220,   1e8,    0.0),
    'earth'       : (R_EARTH, 300,   1e8,    0.0),
    'mercury'     : (2.44e6,  600,   1e8,    0.0),
    'venus'       : (6.05e6,  300,   1e8,    0.0),
    'mars'        : (3.39e6,  210,   1e8,    0.0),
    'jupiter'     : (7.15e7,  150,   1e8,    0.0),
    'saturn'      : (6.03e7,  140,   1e8,    0.0),
    'uranus'      : (2.56e7,  80,    1e8,    0.0),
    'neptune'     : (2.48e7,  80,    1e8,    0.0),
}


class SolarSystemSource:
    """
    A solar system body whose galactic-frame direction is tracked via the
    astropy ephemeris.

    Parameters
    ----------
    name : str
        Astropy body name ('sun', 'moon', 'earth', 'mars', etc.).
    radius_m : float
        Physical radius in metres.  0 treats the body as a point source.
    flux_model : callable
        ``flux_model(freqs_Hz)`` → brightness temperature array [K].
    observer_body : str
        Name of the body the observer is near; positions are computed
        relative to this body ('earth' or 'moon').
    """

    def __init__(self, name, radius_m, flux_model, observer_body='earth'):
        self.name = name
        self.radius_m = float(radius_m)
        self.flux_model = flux_model
        self.observer_body = observer_body
        self.crd_gal = None     # galactic unit vector (3,), updated by update()
        self.distance_m = None  # distance in metres, updated by update()

    def update(self, t):
        """
        Compute the galactic-frame direction and distance to this body at
        time *t* as seen from *observer_body*.
        """
        t = Time(t)
        body_bary = get_body_barycentric(self.name, t)
        obs_bary = get_body_barycentric(self.observer_body, t)
        rel = body_bary - obs_bary        # CartesianRepresentation, AU
        self.distance_m = rel.norm().to(u.m).value
        sc = SkyCoord(CartesianRepresentation(rel.xyz), frame='icrs')
        xyz = sc.galactic.cartesian.xyz.value
        self.crd_gal = xyz / np.linalg.norm(xyz)

    def angular_radius(self):
        """Angular radius of the body's disc in radians (0 if point source)."""
        if self.radius_m == 0 or self.distance_m is None or self.distance_m == 0:
            return 0.0
        return np.arctan2(self.radius_m, self.distance_m)

    def temperature(self, freqs_Hz):
        """Brightness temperature spectrum [K] at *freqs_Hz*."""
        return np.asarray(self.flux_model(freqs_Hz))


# ---------------------------------------------------------------------------
# Vizier query

def query_vizier(max_results=200, S178MHz_cut=100):
    """
    Query the 3CR catalog from Vizier and return source positions and fluxes.

    Parameters
    ----------
    max_results : int
    S178MHz_cut : float
        Minimum flux density at 178 MHz [Jy].

    Returns
    -------
    coords : SkyCoord
    fluxes : ndarray
        Flux densities at 178 MHz [Jy].
    """
    Vizier.ROW_LIMIT = max_results
    v = Vizier(columns=["*", "+_r"])
    result = v.query_constraints(catalog="VIII/1A/3CR",
                                 S178MHz=f">{S178MHz_cut}")
    tbl = result[0]
    if ("_RA.icrs" in tbl.colnames) and ("_DE.icrs" in tbl.colnames):
        ra_icrs, dec_icrs = tbl["_RA.icrs"], tbl["_DE.icrs"]
        if np.issubdtype(ra_icrs.dtype, np.number):
            coords = SkyCoord(ra_icrs * u.deg, dec_icrs * u.deg, frame="icrs")
        else:
            coords = SkyCoord(ra_icrs, dec_icrs,
                              unit=(u.hourangle, u.deg), frame="icrs")
    elif ("RA1950" in tbl.colnames) and ("DE1950" in tbl.colnames):
        coords_fk4 = SkyCoord(tbl["RA1950"], tbl["DE1950"],
                               unit=(u.hourangle, u.deg),
                               frame="fk4", equinox="B1950")
        coords = coords_fk4.transform_to("icrs")
    else:
        raise ValueError("No usable RA/Dec columns found in table.")
    return coords, np.asarray(tbl['S178MHz'])


# ---------------------------------------------------------------------------
# Random source helpers

def random_Jy_fluxes(nsrcs, power_law_index=2.0, min_Jy_flux=2.0, seed=0):
    """Draw flux densities [Jy] from a power-law source count distribution."""
    np.random.seed(seed)
    return min_Jy_flux * (1 - np.random.uniform(size=nsrcs)) ** (
        -1 / (power_law_index - 1)
    )


def random_spectral_indices(nsrcs, spectral_index_range=(-2, 0), seed=0):
    """Draw spectral indices uniformly from *spectral_index_range*."""
    np.random.seed(seed)
    return np.random.uniform(
        spectral_index_range[0], spectral_index_range[1], size=(nsrcs, 1)
    )


def random_points_on_sphere(nsrcs, radec=False):
    """
    Generate *nsrcs* points uniformly distributed on the unit sphere.

    Returns
    -------
    If ``radec=False``: ndarray, shape (nsrcs, 3) — Cartesian unit vectors.
    If ``radec=True``: (ra, dec) tuple of arrays in radians.
    """
    phi = np.random.uniform(0, 2 * np.pi, nsrcs)
    cos_theta = np.random.uniform(-1, 1, nsrcs)
    theta = np.arccos(cos_theta)
    if radec:
        return phi, np.pi / 2 - theta
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack((x, y, z), axis=1)


# ---------------------------------------------------------------------------
# Source catalog

class SourceCatalog:
    """
    Catalog of radio sources for constructing galactic HEALPix sky maps.

    Sources are divided into two groups:

    **Fixed sources** — extragalactic or stellar, with constant positions in
    the galactic frame.  Added via :meth:`add_sources`, :meth:`add_vizier_3c`,
    or :meth:`add_random_sources`.

    **Solar system sources** — Sun, Moon, Earth, and the planets, with
    time-varying galactic-frame positions computed via the astropy ephemeris.
    Added via :meth:`add_sun`, :meth:`add_moon`, :meth:`add_earth`,
    :meth:`add_planet`, or :meth:`add_solar_system_source`.  Call
    :meth:`update_positions` before :meth:`convert_to_healpix` to refresh
    their directions.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the output map.
    freqs_Hz : array_like
        Frequency array [Hz].
    observer_body : str
        The solar system body the observer is located on or near.
        Solar system source positions are computed relative to this body.
        Use ``'earth'`` for ground-based observers and ``'moon'`` for lunar
        surface or lunar orbit observers.
    dtype : str or dtype
        Floating-point dtype for map arrays.
    """

    def __init__(self, nside, freqs_Hz, observer_body='earth', dtype='float32'):
        self.nside = nside
        self.freqs = np.asarray(freqs_Hz, dtype=dtype)
        self.observer_body = observer_body
        self.dtype = dtype
        self._ss_sources = []
        self._crd_gal = np.empty((0, 3), dtype=dtype)   # (N, 3) galactic unit vectors
        self._Tsrc = np.empty((0, len(self.freqs)), dtype=dtype)  # (N, nfreq) [K]

    # ------------------------------------------------------------------
    # Solar system sources

    def add_solar_system_source(self, name, radius_m=0, flux_model=None):
        """
        Add an arbitrary solar system body.

        Parameters
        ----------
        name : str
            Astropy body name.
        radius_m : float
            Physical radius [m].  0 = point source.
        flux_model : callable, optional
            ``flux_model(freqs_Hz)`` → temperature [K].  Defaults to the
            built-in model for known bodies, or zero otherwise.
        """
        if flux_model is None:
            params = _SS_DEFAULTS.get(name.lower())
            if params is not None:
                _, T0, fq0, alpha = params
                flux_model = lambda f, T0=T0, fq0=fq0, a=alpha: (
                    T0 * (np.asarray(f) / fq0) ** a
                )
            else:
                flux_model = lambda f: np.zeros(len(np.asarray(f)))
        if radius_m == 0:
            params = _SS_DEFAULTS.get(name.lower())
            if params is not None:
                radius_m = params[0]
        self._ss_sources.append(
            SolarSystemSource(name, radius_m, flux_model, self.observer_body)
        )

    def add_sun(self, T0=1e6, fq0=65e6, spectral_index=0.3):
        """Add the Sun with a radio power-law brightness temperature model."""
        self.add_solar_system_source(
            'sun', R_SUN,
            lambda f, T0=T0, fq0=fq0, a=spectral_index: T0 * (np.asarray(f) / fq0) ** a
        )

    def add_moon(self, T_mean=220):
        """Add the Moon as a uniform-temperature disc."""
        self.add_solar_system_source(
            'moon', R_MOON,
            lambda f, T=T_mean: T * np.ones(len(np.asarray(f)))
        )

    def add_earth(self, T_mean=300, T_fm_rms=500):
        """
        Add Earth as a disc with thermal emission plus FM-band contamination.

        Parameters
        ----------
        T_mean : float
            Mean brightness temperature [K].
        T_fm_rms : float
            RMS of additive Gaussian noise in the FM band (88–109 MHz) [K].
        """
        def earth_flux(f, T_mean=T_mean, T_fm_rms=T_fm_rms):
            f = np.asarray(f)
            T = T_mean * np.ones_like(f)
            fm = (f > 88e6) & (f < 109e6)
            T[fm] += np.random.randn(int(fm.sum())) * T_fm_rms
            return T.clip(0)
        self.add_solar_system_source('earth', R_EARTH, earth_flux)

    def add_planet(self, name):
        """
        Add a planet using built-in default radius and temperature.

        Parameters
        ----------
        name : str
            One of 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
            'uranus', 'neptune'.
        """
        self.add_solar_system_source(name)

    def add_planets(self):
        """Add all eight planets using built-in defaults."""
        for name in ('mercury', 'venus', 'mars', 'jupiter',
                     'saturn', 'uranus', 'neptune'):
            self.add_planet(name)

    # ------------------------------------------------------------------
    # Fixed sources

    def _append_fixed(self, crd_gal, Tsrc):
        """Append pre-computed galactic unit vectors and temperature spectra."""
        crd_gal = np.atleast_2d(np.asarray(crd_gal, dtype=self.dtype))
        Tsrc = np.atleast_2d(np.asarray(Tsrc, dtype=self.dtype))
        self._crd_gal = np.concatenate([self._crd_gal, crd_gal], axis=0)
        self._Tsrc = np.concatenate([self._Tsrc, Tsrc], axis=0)

    def add_sources(self, skycoords, F0_Jys, spectral_indices, fq0=178e6):
        """
        Add fixed sources from sky coordinates and power-law flux models.

        Parameters
        ----------
        skycoords : SkyCoord
            Source positions (any frame; converted to galactic internally).
        F0_Jys : array_like, shape (N,)
            Reference flux densities [Jy] at *fq0*.
        spectral_indices : array_like, shape (N,) or (N, 1)
            Spectral indices α where S(f) = F0 * (f/fq0)^α.
        fq0 : float
            Reference frequency [Hz].
        """
        F0_Jys = np.asarray(F0_Jys)
        spectral_indices = np.asarray(spectral_indices).reshape(-1, 1)
        Jy2K = Jy2K_nside(self.nside, self.freqs)
        Fsrc = F0_Jys[:, None] * (self.freqs[None, :] / fq0) ** spectral_indices
        Tsrc = Fsrc * Jy2K[None, :]
        crd_gal = skycoords_to_galvec(skycoords)
        self._append_fixed(crd_gal, Tsrc)

    def add_vizier_3c(self, S_cut=100, spectral_index=-0.7, fq0=178e6):
        """
        Query the 3CR catalog from Vizier and add all sources above *S_cut* Jy.

        Parameters
        ----------
        S_cut : float
            Minimum 178 MHz flux density [Jy].
        spectral_index : float or array_like
            Spectral index or per-source array.
        fq0 : float
            Reference frequency matching *S_cut* [Hz].
        """
        coords, fluxes = query_vizier(S178MHz_cut=S_cut)
        indices = np.full(len(fluxes), spectral_index)
        self.add_sources(coords, fluxes, indices, fq0=fq0)

    def add_random_sources(self, nsrcs, power_law_index=2.0, min_Jy_flux=2.0,
                           spectral_index_range=(-2, 0), fq0=150e6, seed=0):
        """Add *nsrcs* randomly positioned sources drawn from a power-law count."""
        F0_Jys = random_Jy_fluxes(nsrcs, power_law_index=power_law_index,
                                   min_Jy_flux=min_Jy_flux, seed=seed)
        indices = random_spectral_indices(nsrcs,
                       spectral_index_range=spectral_index_range, seed=seed)
        crd = random_points_on_sphere(nsrcs)
        Jy2K = Jy2K_nside(self.nside, self.freqs)
        Fsrc = F0_Jys[:, None] * (self.freqs[None, :] / fq0) ** indices
        Tsrc = Fsrc * Jy2K[None, :]
        self._append_fixed(crd, Tsrc)

    # ------------------------------------------------------------------
    # Time update

    def update_positions(self, t):
        """
        Update galactic-frame positions of all solar system sources.

        Call this before :meth:`convert_to_healpix` whenever the time changes.

        Parameters
        ----------
        t : `~astropy.time.Time` or str
        """
        t = Time(t)
        for src in self._ss_sources:
            src.update(t)

    # ------------------------------------------------------------------
    # Render

    def convert_to_healpix(self, solar_system=True, fixed=True):
        """
        Render sources onto a galactic HEALPix brightness temperature map.

        Solar system sources must have been updated via :meth:`update_positions`
        before calling this method.

        Parameters
        ----------
        solar_system : bool
            Include solar system sources.  Extended bodies (Sun, Moon, Earth)
            are spread over disc pixels via ``disc_overlap_fraction``; planets
            with a sub-pixel disc are assigned to their nearest pixel.
            Default: True.
        fixed : bool
            Include fixed catalog sources (Vizier, random, user-supplied).
            These are always assigned to their nearest pixel, which is
            appropriate for visualisation but can introduce pixelisation
            artefacts in simulations.  Default: True.

        Returns
        -------
        hpx : ndarray, shape (npix, nfreq)
            Brightness temperature [K] in galactic HEALPix RING scheme.
        """
        npix = hp.nside2npix(self.nside)
        nfreq = len(self.freqs)
        hpx = np.zeros((npix, nfreq), dtype=self.dtype)

        # Fixed sources — nearest-pixel assignment
        if fixed and len(self._crd_gal) > 0:
            pix = hp.vec2pix(self.nside,
                             self._crd_gal[:, 0],
                             self._crd_gal[:, 1],
                             self._crd_gal[:, 2])
            np.add.at(hpx, pix, self._Tsrc)

        # Solar system sources
        if solar_system:
            for src in self._ss_sources:
                if src.crd_gal is None:
                    continue
                T = src.temperature(self.freqs).astype(self.dtype)
                r_ang = src.angular_radius()
                if r_ang > 0:
                    pix_disk, frac = disc_overlap_fraction(
                        self.nside, src.crd_gal, r_ang
                    )
                    hpx[pix_disk] += frac[:, None] * T[None, :]
                else:
                    pix = hp.vec2pix(self.nside, *src.crd_gal)
                    hpx[pix] += T

        return hpx
