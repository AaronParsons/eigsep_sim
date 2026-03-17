import numpy as np
import healpy as hp
import astropy.units as u
from astropy.constants import c, k_B
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord, FK5
from astroquery.vizier import Vizier
from .sun import sun_radius, disc_overlap_fraction

PRECISION = 1

if PRECISION == 1:
    real_dtype = np.float32
    complex_dtype = np.complex64
else:
    assert PRECISION == 2
    real_dtype = np.float64
    complex_dtype = np.complex128


def radec_to_eqvec(ra_rad, dec_rad):
    th_rad = np.pi/2 - dec_rad
    return hp.ang2vec(th_rad, ra_rad)


def eqvec_to_pix(nside, crd_eq, r_rad=None):
    """
    Args:
      nside        : base map resolution
      ra_rad, dec_rad : source position in radians
      r_rad    : angular radius [radians]

    Returns:
      pix: pixels with centers within radius
    """
    if r_rad is not None:
        return hp.query_disc(nside, crd_eq, r_rad, inclusive=True)
    else:
        return hp.vec2pix(nside, crd_eq[..., 0], crd_eq[..., 1], crd_eq[..., 2])

def query_vizier(max_results=50, S178MHz_cut=100):
    '''Query bright radio sources from 3C via Vizier.'''
    Vizier.ROW_LIMIT = max_results
    v = Vizier(columns=["*", "+_r"])
    result = v.query_constraints(catalog="VIII/1A/3CR",
                                 S178MHz=f">{S178MHz_cut}")

    tbl = result[0]
    # Preferred: ICRS columns provided by Vizier
    if ("_RA.icrs" in tbl.colnames) and ("_DE.icrs" in tbl.colnames):
        ra_icrs, dec_icrs = tbl["_RA.icrs"], tbl["_DE.icrs"]
        if np.issubdtype(ra_icrs.dtype, np.number):
            coords = SkyCoord(ra_icrs * u.deg, dec_icrs * u.deg, frame="icrs")
        else:
            # sexagesimal string
            coords = SkyCoord(ra_icrs, dec_icrs, unit=(u.hourangle, u.deg),
                              frame="icrs")

    # Fallback: B1950 columns -> convert FK4(B1950) to ICRS
    elif ("RA1950" in tbl.colnames) and ("DE1950" in tbl.colnames):
        ra_b1950, dec_b1950 = tbl["RA1950"], tbl["DE1950"]
        coords_fk4 = SkyCoord(ra_b1950, dec_b1950, unit=(u.hourangle, u.deg),
                              frame="fk4", equinox="B1950")
        coords = coords_fk4.transform_to("icrs")

    else:
        raise ValueError("No usable RA/Dec columns found in table.")

    return coords, np.asarray(tbl['S178MHz'])


def skycoords_to_eqvec(crds, t_astropy):
    crds_now = crds.transform_to(FK5(equinox=t_astropy))
    ra =  crds_now.ra.to(u.rad).value
    dec = crds_now.dec.to(u.rad).value
    vec = radec_to_eqvec(ra, dec)
    return vec
    

def Jy2K_nside(nside, freqs_Hz, F_Jy=1.0):
    F = F_Jy * u.Jy
    nu = freqs_Hz * u.Hz
    omega_pix = hp.nside2pixarea(nside)
    T_K = (c**2 / (2 * k_B * nu**2) * F / omega_pix).to(u.K)
    return T_K.value 


def random_Jy_fluxes(nsrcs, power_law_index=2.0, min_Jy_flux=2.0, seed=0):
    """Return fluxes for the specified number of sources drawn from a power
    law of strengths."""
    np.random.seed(seed)
    F0_Jy = min_Jy_flux * (1 - np.random.uniform(size=nsrcs))**(-1 / (power_law_index - 1))
    return F0_Jy


def random_spectral_indices(nsrcs, spectral_index_range=(-2, 0), seed=0):
    np.random.seed(seed)
    return np.random.uniform(spectral_index_range[0], spectral_index_range[1],
                             size=(nsrcs, 1))


def random_points_on_sphere(nsrcs, radec=False):
    """
    Generate random points uniformly distributed on the unit sphere.
    Returns array of shape (n, 3).
    """
    phi = np.random.uniform(0, 2*np.pi, nsrcs)     # azimuth
    cos_theta = np.random.uniform(-1, 1, nsrcs)    # cos(polar angle)
    theta = np.arccos(cos_theta)
    if radec:
        return phi, np.pi/2 - theta
    else:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack((x, y, z), axis=1)


class SourceCatalog:
    def __init__(self, nside, freqs_Hz, t_astropy):
        self.nside = nside
        self.freqs = freqs_Hz
        self.times = t_astropy
        self.Jy2K = Jy2K_nside(self.nside, self.freqs)
        # in crd_eq/Tsrc arrays, Sun is earmarked for the first slot
        self.crd_eq = np.empty((1, 3), dtype=real_dtype)
        self.Tsrc = np.empty((1, freqs_Hz.size), dtype=real_dtype)
        self.sun = get_sun(self.times)
        self.update_sun_pos(0)
        self.set_sun_flux()

    def update_sun_pos(self, tind):
        ra =  self.sun.ra[tind].to(u.rad).value
        dec = self.sun.dec[tind].to(u.rad).value
        self.crd_eq[0] = radec_to_eqvec(ra, dec)

    def set_sun_flux(self, F0_Sun=1e4, spectral_index=0, fq0=150e6):
        Tsun = F0_Sun * self.Jy2K * (self.freqs / fq0)**spectral_index
        self.Tsrc[0] = Tsun

    def convert_source_fluxes(self, F0_Jys, indices, fq0=150e6):
        Fsrc = F0_Jys[:, None] * (self.freqs[None, :] / fq0)**indices[:, None]
        Tsrc = Fsrc * self.Jy2K[None, :]
        return Tsrc.astype(real_dtype)

    def add_sources(self, skycoords, F0_Jys, indices, fq0=150e6, tind=0):
        Tsrc = self.convert_source_fluxes(F0_Jys, indices, fq0=fq0)
        crd_eq = skycoords_to_eqvec(skycoords, self.times[tind])
        self.Tsrc = np.concatenate([self.Tsrc, Tsrc], axis=0)
        self.crd_eq = np.concatenate([self.crd_eq, crd_eq], axis=0)

    def add_random_sources(self, nsrcs, power_law_index=2.0, min_Jy_flux=2.0,
                           spectral_index_range=(-2, 0), seed=0):
        F0_Jys = random_Jy_fluxes(nsrcs, power_law_index=power_law_index,
                                  min_Jy_flux=min_Jy_flux, seed=seed)
        indices = random_spectral_indices(nsrcs,
                        spectral_index_range=spectral_index_range, seed=seed)
        Tsrc = self.convert_source_fluxes(F0_Jys, indices)
        crd_eq = random_points_on_sphere(nsrcs).astype(real_dtype)
        self.Tsrc = np.concatenate([self.Tsrc, Tsrc], axis=0)
        self.crd_eq = np.concatenate([self.crd_eq, crd_eq], axis=0)

    def convert_to_healpix(self, t_ind=0):
        hpx = np.zeros((hp.nside2npix(self.nside), self.freqs.size), dtype=real_dtype)
        pix = eqvec_to_pix(self.nside, self.crd_eq[1:])
        hpx[pix, :] = self.Tsrc[1:]  # XXX worry about overlapping pixels?
        r_sun = sun_radius(self.times[t_ind])
        pix_sun, frac = disc_overlap_fraction(self.nside, self.crd_eq[0], r_sun)
        hpx[pix_sun, :] = self.Tsrc[:1] * frac[:, None]
        return hpx
