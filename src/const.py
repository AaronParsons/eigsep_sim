"""
Module for keeping track of physical constants.

All constants are returned as plain floats in MKS (SI) units.
See description() for a dictionary of constants and their descriptions.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u

# Fundamental constants (SI)
pi = np.pi
c = const.c.value                 # speed of light [m / s]
G = const.G.value                 # gravitational constant [m^3 / kg / s^2]
h = const.h.value                 # Planck constant [J s]
e = const.e.value                 # elementary charge [C]
m_e = const.m_e.value             # electron mass [kg]
m_p = const.m_p.value             # proton mass [kg]
k = const.k_B.value               # Boltzmann constant [J / K]
sigma_sb = const.sigma_sb.value   # Stefan-Boltzmann constant [W / m^2 / K^4]

# Astronomical quantities
au = const.au.value               # meters in 1 AU
r_sun = const.R_sun.value         # radius of sun [m]
m_sun = const.M_sun.value         # mass of sun [kg]
pc = const.pc.value               # meters in 1 parsec

# Time quantities
s_per_day = u.day.to(u.s)         # seconds in a solar Earth day
s_per_yr = u.yr.to(u.s)           # seconds in a Julian year
sidereal_day = u.sday.to(u.s)     # seconds in a sidereal day

# Derived quantities
len_ns = c * 1e-9                 # length of a nanosecond in meters
deg = np.pi / 180.0               # degrees in radians
sq_deg = deg ** 2                 # square degree in steradians
arcmin = deg / 60.0               # arcminute in radians
arcsec = arcmin / 60.0            # arcsecond in radians
ft = u.imperial.ft.to(u.m)        # length of a foot in meters

# Radii
R_MOON = 1_737_400.0              # radius of moon [m] (not in astropy)
R_SUN = const.R_sun.value         # radius of sun [m]
R_EARTH = const.R_earth.value     # radius of earth [m]

# Flux units
Jy = u.Jy.to(u.W / u.m**2 / u.Hz)  # Jansky [W / m^2 / Hz]


def description():
    return {
        "pi": "Pi",
        "c": "Speed of light [m/s]",
        "G": "Newton's gravitational constant [m^3 kg^-1 s^-2]",
        "h": "Planck constant [J s]",
        "e": "Elementary charge [C]",
        "m_e": "Electron mass [kg]",
        "m_p": "Proton mass [kg]",
        "k": "Boltzmann constant [J/K]",
        "sigma_sb": "Stefan-Boltzmann constant [W m^-2 K^-4]",
        "au": "meters in 1 AU",
        "r_sun": "radius of sun [m]",
        "m_sun": "mass of sun [kg]",
        "pc": "meters in 1 parsec",
        "s_per_day": "seconds in a solar Earth day",
        "s_per_yr": "seconds in a Julian year",
        "sidereal_day": "seconds in a sidereal day",
        "len_ns": "length of a nanosecond in meters",
        "deg": "degrees in radians",
        "sq_deg": "square degree in steradians",
        "arcmin": "arcminute in radians",
        "arcsec": "arcsecond in radians",
        "ft": "length of a foot in meters",
        "Jy": "Jansky [W / m^2 / Hz]",
        "R_MOON": "radius of moon [m]",
        "R_SUN": "radius of sun [m]",
        "R_EARTH": "radius of earth [m]",
    }
