import numpy as np
import healpy as hp
import astropy.units as u
from astropy.constants import R_sun
from astropy.coordinates import get_body_barycentric


def sun_radius(t_astropy):
    # Earth–Sun distance via barycentric positions
    r_earth = get_body_barycentric('earth', t_astropy)    # vector Earth wrt SSB
    r_sun   = get_body_barycentric('sun', t_astropy)      # vector Sun   wrt SSB
    r_es    = (r_earth - r_sun).norm()            # distance Earth->Sun
    # Angular radius r = asin(R_sun / r_es)
    r = np.arcsin((R_sun.to(u.AU) / r_es).decompose().value)  # radians
    return r


def disc_overlap_fraction(nside, crd_eq, r_rad, k=3):
    """Area-weighted overlap of a uniform circular disc with HEALPix pixels."""
    th_c, phi_c = hp.vec2ang(crd_eq)
    
    # Candidate pixels (only those that could intersect)
    ipix = hp.query_disc(nside, crd_eq, r_rad, inclusive=True, nest=True)
    nside_hi = nside * (2**k)
    ipix_hi_disc = hp.query_disc(nside_hi, crd_eq, r_rad, inclusive=True, nest=True)
    ipix_hi_disc = np.sort(ipix_hi_disc)

    four_k = 4**k
    frac = np.empty(ipix.size, dtype=np.float32)
    for i, p in enumerate(ipix):
        start = p * four_k
        stop  = start + four_k
        # number of hi-res indices in [start, stop)
        # (ipix_hi_disc is sorted; use searchsorted for O(log N))
        left  = np.searchsorted(ipix_hi_disc, start, side='left')
        right = np.searchsorted(ipix_hi_disc, stop,  side='left')
        frac[i] = (right - left) / float(four_k)
    frac /= np.sum(frac)
    ipix_ring = hp.nest2ring(nside, ipix[frac > 0])
    return ipix_ring, frac[frac > 0]
