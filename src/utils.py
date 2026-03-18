import numpy as np
import eigsep_terrain.reflectivity as etr

from .coord import rot_m
from .const import R_MOON


def moon_surface_distance(angle, d, r=R_MOON):
    a, b, c = 1, -2 * d * np.cos(angle), d**2 - r**2
    radical = b**2 - 4 * a * c
    ans = np.where(radical > 0, -b - np.sqrt(radical.clip(0)) / (2 * a), np.nan)
    #ans = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    #ans1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)  # never the right answer
    return np.where(ans > 0, ans, np.nan)


def moon_reflect_vector(vec_to_surface, moon_pos, r=R_MOON):
    incident = vec_to_surface / np.linalg.norm(vec_to_surface, axis=0)
    normal = (vec_to_surface - moon_pos[:, None]) / r
    #normal /= np.linalg.norm(normal, axis=0)  # if not using hard-coded moon radius above
    vec_out = incident - 2 * np.einsum('ij,ij->j', incident, normal) * normal
    return vec_out


def reflectivity(freqs, resistivity_ohm_m, eta0=1):
    omega = 2 * np.pi * freqs  # Hz
    conductivity = etr.conductivity_from_resistivity(resistivity_ohm_m)
    eta = etr.permittivity_from_conductivity(conductivity, freqs)
    R = etr.reflection_coefficient(eta, eta0=eta0)
    return R


def sample_disk(pos, r_ang, nsamples):
    cos_theta = np.random.uniform(np.cos(r_ang), 1, nsamples)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, nsamples)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    samples = np.vstack((x, y, z))
    pos = pos / np.linalg.norm(pos)
    z = np.array([0, 0, 1])
    rot_axis = np.cross(z, pos)
    rot_axis /= np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(z, pos))
    _rot_m = rot_m(rot_angle, rot_axis)
    return _rot_m @ samples
