#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File coordtrans.py

Last update: 24/105/2017

Usage:
"""

import yaml
import numpy as np
from itertools import ifilter, product

import pyfpm.data as dt
from pyfpm.fpmmath import set_iterator, translate
import pyfpm.data as dt

cfg = dt.load_config()
print(cfg.max_power)


def corrected_coordinates(theta=None, shift=None, phi=None, cfg=None):
    phi = np.degrees(np.arctan((1.*shift+1)/cfg.sample_height))
    return theta, phi


def phi_rot(od, phi):
    """ od rotation -phi angle- or along y axis
    """
    rot_mat = np.array([[np.cos(phi),  0, np.sin(phi)],
                        [0,            1,            0],
                        [-np.sin(phi), 0, np.cos(phi)]])
    return rot_mat.dot(od)


def theta_rot(od, theta):
    """ od rotation -theta angle- or along z axis
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta),  0],
                        [np.sin(theta),  np.cos(theta),  0],
                        [0,              0,              1]])
    return rot_mat.dot(od)


def apply_corrections(origin, light_dir, source_center,
                      source_tilt, platform_tilt):
    """ Apply coordinate corrections as a 3d tranformation of the source
    """
    # First correction: origin translation
    origin[0] = origin[0]+source_center[0]
    origin[1] = origin[1]+source_center[1]

    origin_prime = np.array(origin)

    st_phi = np.radians(source_tilt[1])
    st_theta = np.radians(source_tilt[0])
    pt_phi = np.radians(platform_tilt[1])
    pt_theta = np.radians(platform_tilt[0])

    # direction rotation whith source_tilt and platform_tilt
    light_dir = phi_rot(light_dir, st_phi + pt_phi)
    light_dir = theta_rot(light_dir, st_theta + pt_theta)
    # source vector rotation whith platform_tilt
    origin = phi_rot(origin_prime, pt_phi)
    origin = theta_rot(origin, pt_theta)
    return origin, light_dir


def rotate(prime_pos, theta, phi):
    """ Apply coordinate corrections as a 3d tranformation of the source
    """
    light_dir = np.array([0, 0, 1])
    origin_prime = np.array(prime_pos)
    light_dir = phi_rot(light_dir, phi)
    light_dir = theta_rot(light_dir, theta)
    # origin = phi_rot(origin_prime, phi)
    origin = theta_rot(origin_prime, theta)
    return origin, light_dir
