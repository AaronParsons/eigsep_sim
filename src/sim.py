import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d
import hera_sim
from matvis import coordinates
import fftvis
#from pyuvdata.analytic_beam import AnalyticBeam, AiryBeam, UniformBeam
from matvis.cpu.coords import CoordinateRotationERFA
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import healpy
from pygdsm import GlobalSkyModel16 as GSM16
import eigsep_terrain.utils as etu
import aipy
import tqdm
from .src import SourceCatalog, query_vizier
from .hpm import HPM, float_dtype

PRECISION = 1

if PRECISION == 1:
    real_dtype = np.float32
    complex_dtype = np.complex64
else:
    assert PRECISION == 2
    real_dtype = np.float64
    complex_dtype = np.complex128

DEFAULT_LAT = 39.247699
DEFAULT_LON = -113.402660
DEFAULT_HGT = 1800 # m

DEFAULT_RESISTIVITY = 3e2 # Ohm m

BEAM_NPZ = 'eigsep_vivaldi.npz'
TERRAIN_NPZ = 'horizon_models_v000.npz'
BANDPASS_NPZ = 'bandpass.npz'
S11_NPZ = 'S11_eigsep_bowtie_v000.npz'
T21CM_NPZ = 'models_21cm.npz'

class Beam(HPM):
    def __init__(self, freqs, filename=BEAM_NPZ, peak_normalize=True):
        bm_data = load_beam(freqs, filename=filename)
        self.freqs = freqs
        if peak_normalize:
            bm_data /= bm_data.max()
        nside_beam = healpy.npix2nside(bm_data.shape[0])
        self.set_az(0)
        self.set_alt(0)
        HPM.__init__(self, nside_beam, interp=True)
        self.set_map(bm_data)

    def set_az(self, theta, az_vec=np.array([0, 0, 1], dtype=real_dtype)):
        # XXX angle applied coordinates (*not* the platform); might be confusing
        self.az = theta
        self.rot_az = aipy.coord.rot_m(theta, az_vec)

    def set_alt(self, theta, alt_vec=np.array([1, 0, 0], dtype=real_dtype)):
        # XXX angle applied coordinates (*not* the platform); might be confusing
        self.alt = theta
        self.rot_alt = aipy.coord.rot_m(theta, alt_vec)

    def get_rotation_matrices(self, azs, alts):
        n_rots = azs.size
        rot_ms = np.empty((n_rots, 3, 3), dtype=real_dtype)
        for cnt in range(n_rots):
            self.set_az(azs.flat[cnt])
            self.set_alt(alts.flat[cnt])
            rot_ms[cnt] = self.rot_az.dot(self.rot_alt)
        rot_ms = rot_ms.reshape(azs.shape + (3, 3))
        return rot_ms

    def __getitem__(self, crd_top):
        """Access data on a sphere via hpm[crd].
        crd = either 1d array of pixel indices, (th,phi), or (x,y,z), where
        th,phi,x,y,z are numpy arrays of coordinates."""
        rot_m = self.rot_az.dot(self.rot_alt)
        bx, by, bz = rot_m.dot(crd_top)
        return HPM.__getitem__(self, (bx, by, bz))


class Terrain(HPM):
    def __init__(self,
                 freqs,
                 height=114,
                 filename=TERRAIN_NPZ,
                 resistivity_ohm_m=DEFAULT_RESISTIVITY,
                 transmitters=[],
                ):
        with np.load(filename) as npz:
            nside_horizon = npz['nside']
            heights = npz['heights']
            i = np.argmin(np.abs(heights - height))
            r = npz['r'].astype(real_dtype)
            r = r[i]  # horizon model with closest height (XXX could interpolate)
        HPM.__init__(self, nside_horizon, interp=False)
        self.freqs = freqs
        self.set_map(r)
        self.resistivity_ohm_m = resistivity_ohm_m
        self.set_transmitters(transmitters)

    def set_transmitters(self, transmitters):
        n_tx = len(transmitters)
        self.tx_crd_top = np.empty((3, n_tx), dtype=real_dtype)
        self.tx_flux = np.zeros((n_tx, self.freqs.size), dtype=real_dtype)
        for cnt, (vec, fqs, pwrs) in enumerate(transmitters):
            self.tx_crd_top[:, cnt] = vec
            chs = np.searchsorted(self.freqs, fqs)
            self.tx_flux[cnt, chs] = pwrs

    def reflectivity(self, freqs, resistivity_ohm_m, eta0=1):
        omega = 2 * np.pi * freqs # Hz
        conductivity = etu.conductivity_from_resistivity(resistivity_ohm_m)
        eta = etu.permittivity_from_conductivity(conductivity, freqs)
        gamma = etu.reflection_coefficient(eta, eta0=eta0)
        return gamma.astype(real_dtype)

    def cover_sky(self, crd_top, Isky, Tgnd=300):
        #gamma = self.reflectivity(freqs)
        # TODO: reflect vectors to sky
        tx, ty, tz = crd_top
        r = self[tx, ty, tz]
        I = np.where(np.isnan(r[:, None]), Isky, Tgnd)
        return I
        

def load_T21cm_models(freqs, model_index=None, filename=T21CM_NPZ):
    npz = np.load(filename)
    mdl_freqs = npz['freqs'] * 1e9
    mdl_T = npz['models'] * 1e-3
    if model_index is not None:
        mdl_T = mdl_T[model_index]
    mdl_interp = interp1d(mdl_freqs, mdl_T, kind='cubic', fill_value=0, bounds_error=False)
    T_21cm = mdl_interp(freqs)
    return T_21cm

def load_bandpass(freqs, filename=BANDPASS_NPZ):
    npz = np.load(filename)
    mdl_freqs = npz['freqs']
    bp = npz['bandpass']
    mdl_interp = interp1d(mdl_freqs, bp, kind='cubic', fill_value=0, bounds_error=False)
    bandpass = mdl_interp(freqs)
    return bandpass

def load_beam(freqs, filename=BEAM_NPZ):
    npz = np.load(filename)
    mdl_freqs = npz['freqs']
    bm = npz['bm'].T  # put in px/fq order
    mdl_interp = interp1d(mdl_freqs, bm, kind='cubic', fill_value=0, bounds_error=False)
    bm = mdl_interp(freqs)
    return bm.astype(real_dtype)

def load_S11(freqs, filename=S11_NPZ, termination=None):
    npz = np.load(filename)
    mdl_freqs = npz['freqs']
    if termination is None:
        mdl_S11 = npz['S11']
    else:  # redo S11 for a different termination
        Z = npz['Z']
        gamma = np.abs(Z - termination) / np.abs(Z + termination)
        mdl_S11 = gamma**2
    mdl_interp = interp1d(mdl_freqs, mdl_S11, kind='cubic', fill_value=0, bounds_error=False)
    S11 = mdl_interp(freqs)
    return S11

#   ___ _     _          _ ___ _       
#  / __| |___| |__  __ _| / __(_)_ __  
# | (_ | / _ \ '_ \/ _` | \__ \ | '  \ 
#  \___|_\___/_.__/\__,_|_|___/_|_|_|_|

class GlobalSim:
    """Class for holding simulation parameters."""
    def __init__(self,
                 times,
                 freqs,
                 beam,
                 terrain,
                 monopole=None,
                 lon=DEFAULT_LON, lat=DEFAULT_LAT, height=DEFAULT_HGT,
                ):
        self.eq2ga_m = aipy.coord.convert_m('eq', 'ga')
        self.beam = beam
        self.terrain = terrain
        self.freqs = np.asarray(freqs, dtype=real_dtype)
        self.location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=height)
        self.times = times
        self.set_sky_model(monopole=monopole)
        self.update_crds(times)

    @property
    def nfreqs(self):
        return self.freqs.size

    @property
    def ntimes(self):
        return self.times.size

    def gen_galactic_gsm(self, resolution='lo', chromatic=True, fq0=150e6):
        """Return an HPM containing a GSM."""
        gsm = GSM16(freq_unit='Hz', data_unit='TRJ', resolution=resolution, include_cmb=True)
        if chromatic:
            gsm_map = gsm.generate(self.freqs)
        else:
            gsm_map = gsm.generate(np.array(fq0))
        gsm_hpm = HPM(nside=gsm.nside)
        gsm_hpm.set_map(gsm_map.T)
        return gsm_hpm

    def gen_point_sources(self, nsrcs, chromatic=True, power_law_index=2.0, min_src_flux=2.0,
                                spectral_index_range=(-2, 0), fq0=150e6):
        """Return fluxes for the specified number of sources drawn from a power law of strengths."""
        np.random.seed(0)
        Isky0 = min_src_flux * (1 - np.random.uniform(size=nsrcs))**(-1 / (power_law_index - 1))
        if chromatic:
            index0 = np.random.uniform(spectral_index_range[0], spectral_index_range[1], size=(nsrcs, 1))
            flux = Isky0[:,None] * (self.freqs[None,:] / fq0)**index0
        else:
            flux = Isky0[:,None] * (freqs[None,:] / fq0)**0
        return flux

    def set_sky_model(self,
                      monopole=None,
                      gsm=True,
                      sun=True,
                      three_c=True,
                      nrandom=0,
                      chromatic=True,
                      fq0=150e6,
                     ):
        """Set self.sky_model to a weighted combination of GSM and point sources."""
        gsm = self.gen_galactic_gsm(chromatic=chromatic, fq0=fq0)
        self.nside, self.npix = gsm.nside(), gsm.npix()
        th, self.ra = healpy.pix2ang(self.nside, np.arange(self.npix))
        self.dec = np.pi/2 - th
        self.crd_eq = np.asarray(coordinates.point_source_crd_eq(self.ra, self.dec), dtype=real_dtype)
        gx, gy, gz = self.eq2ga_m @ self.crd_eq
        self.sky_model = gsm[gx, gy, gz] 
        self.catalog = SourceCatalog(self.nside, self.freqs, self.times)
        if not sun:
            self.catalog.set_sun_flux(F0_Sun=0.0)
        if three_c:
            srcs, fluxes = query_vizier()
            self.catalog.add_sources(srcs, fluxes, np.full(fluxes.shape, -1), fq0=178e6)
        if nrandom > 0:
            self.catalog.add_random_sources(nrandom)
        # XXX for now, rasterizing catalog
        self.sky_model += self.catalog.convert_to_healpix()
        if monopole is not None:
            assert monopole.size == self.nfreqs
            self.sky_model += monopole[None, :]

    def get_rotation_matrix(self, tind):
        obsf = self.crds._get_obsf(self.times[tind], self.location)
        astrom = self.crds._apco(obsf)
        # cirs to hadec rot
        ce = np.cos(astrom["eral"])
        se = np.sin(astrom["eral"])
        c2h = np.array([[ce, se, 0], [-se, ce, 0], [0, 0, 1]])
        # Polar motion
        sx = np.sin(astrom["xpl"])
        cx = np.cos(astrom["xpl"])
        sy = np.sin(astrom["ypl"])
        cy = np.cos(astrom["ypl"])
        pm = np.array([[     cx,  0,       sx],
                       [sx * sy, cy, -cx * sy],
                       [-sx * cy, sy, cx * cy]])
        # hadec to enu
        enu = np.array([[              0, 1,              0],
                        [-astrom["sphi"], 0, astrom["cphi"]],
                        [ astrom["cphi"], 0, astrom["sphi"]]])
        eq2top_m = enu.dot(pm.dot(c2h))
        return eq2top_m

    def get_topocentric(self, tind, crd_eq=None):
        eq2top_m = self.get_rotation_matrix(tind)
        if crd_eq is None:
            crd_eq = self.crd_eq
        tx, ty, tz = eq2top_m.dot(crd_eq)
        return tx, ty, tz

    def update_crds(self, times):
        skycrd = SkyCoord(ra=self.ra, dec=self.dec, unit='rad')
        self.crds = CoordinateRotationERFA(
                    skycoords=skycrd,
                    times=self.times,
                    telescope_loc=self.location,
                    flux=self.sky_model,
                    update_bcrs_every=1e9,
                    precision=PRECISION,
        )
        self.crds.setup()


    def time2lst(self):
        raise NotImplementedError

    def beam_response(self, crd_top, alt=0.0, az=0.0):
        self.beam.set_az(az)
        self.beam.set_alt(alt)
        tx, ty, tz = crd_top
        resp = self.beam[tx, ty, tz]
        return resp

    def terrain_screen(self, crd_top, flux, Tgnd):
        return self.terrain.cover_sky(crd_top, flux, Tgnd=Tgnd)

    def sim(self, azalts=np.zeros((1,2), dtype=real_dtype),
            Tgnd=300.0, Trx=50.0, bandpass=1.0, S11=0.0):
        """Convert the specified (healpix) fluxes into visibilities using fftvis."""
        S12 = 1 - S11
        assert azalts.shape[-1] == 2
        nazalt_per_t = azalts.shape[-2]
        if azalts.ndim == 3:
            assert azalts.shape[-3] == self.ntimes
        else:
            azalts = np.broadcast_to(azalts, (self.ntimes, *azalts.shape))
        rot_ms = self.beam.get_rotation_matrices(azalts[..., 0], azalts[..., 1])
        rot_ms = jnp.asarray(rot_ms, dtype=float_dtype)
        vis = np.empty((self.ntimes, nazalt_per_t, self.nfreqs), dtype=real_dtype)
        for tind in tqdm.tqdm(range(self.ntimes)):
            crd_top = self.get_topocentric(tind)
            #self.update_sun_pos(tind)
            #sun_pix = healpy.vec2pix(nside, self._sun_crd_eq)
            #self.sky_model[sun_pix, :] += self._Tsun / sun_pix.size
            #sun_crd_top = self.get_topocentric(tind, crd_eq=self._sun_crd_eq)
            Isky = self.terrain_screen(crd_top, self.sky_model, Tgnd)
            #self.sky_model[sun_pix, :] -= self._Tsun / sun_pix.size
            #crd_top = np.concatenate([crd_top, sun_crd_top, self.terrain.tx_crd_top], axis=1)
            crd_top = np.concatenate([crd_top, self.terrain.tx_crd_top], axis=1)
            #Isky = np.concatenate([Isky, self._Tsun, self.terrain.tx_flux], axis=0)
            Isky = np.concatenate([Isky, self.terrain.tx_flux], axis=0)
            Isky = jnp.asarray(Isky, dtype=float_dtype)
            crd_top = jnp.asarray(crd_top, dtype=float_dtype)
            sky_spec = self.beam.rotate_interpolate_and_sum(Isky, crd_top, rot_ms[tind])
            vis[tind] = bandpass[None, :] * (S12[None, :] * sky_spec + Trx)
        return vis
