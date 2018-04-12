#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File coordtrans.py

Last update: 24/105/2017

Usage:
"""

import yaml
import numpy as np

import pyfpm.data as dt
from pyfpm.fpmmath import set_iterator, translate
import pyfpm.data as dt

cfg = dt.load_config()


def tidy(number):
    """ Rounding function to work under mechanical precission.
    """
    return np.around(number, 2)

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


def platform_to_cartesian(plat_coordinates, light_dir, source_center,
                          source_tilt, platform_tilt, height):
    """ Apply coordinate corrections as 3d tranformations of the source to
        convert platform coordinates to cartesian.

    Parameters:
    -----------
        source_position:    (array) source position in platform coordinates ([theta, shift]).
        light_dir:          (array) direction of light vector (default is [0, 0, 1]).
        source_center       (array) source center in cartesian coordinates (default is [0, 0])
                                    projected in the moving platform.
        source_tilt         (array) two parameters defining light source rotation ([theta, phi])
        platform_tilt       (array) two parameters defining platform rotation ([theta, phi])
    """
    # Translatig platform coordinates to 2D cartesian (x, y) in platform
    theta_rad = np.radians(plat_coordinates[0])
    x_plat = plat_coordinates[1] * np.cos(theta_rad)
    y_plat = plat_coordinates[1] * np.sin(theta_rad)
    source_position = [x_plat, y_plat, 0]  # All rotations made in z=0 plane
                                           # later I add the height
    # First correction: origin translation
    source_position[0] += source_center[0]
    source_position[1] += source_center[1]
    new_source_position = np.array(source_position)

    st_phi = np.radians(source_tilt[1])
    st_theta = np.radians(source_tilt[0])
    pt_phi = np.radians(platform_tilt[1])
    pt_theta = np.radians(platform_tilt[0])

    # direction rotation whith source_tilt and platform_tilt
    light_dir = phi_rot(light_dir, st_phi + pt_phi)
    light_dir = theta_rot(light_dir, st_theta + pt_theta)
    # source vector rotation whith platform_tilt
    source_position = phi_rot(new_source_position, pt_phi)
    source_position = theta_rot(source_position, pt_theta)
    source_position[2] += height
    return tidy(source_position), tidy(light_dir)


def platform_to_spherical(plat_coordinates, light_dir, source_center,
                          source_tilt, platform_tilt, height):
    """ Apply coordinate corrections as 3d tranformations of the source to
        convert platform coordinates to spherical.

    Parameters:
    -----------
        source_position:    (array) source position in platform coordinates ([theta, shift]).
        light_dir:          (array) direction of light vector (default is [0, 0, 1]).
        source_center       (array) source center in cartesian coordinates (default is [0, 0])
                                    projected in the moving platform.
        source_tilt         (array) two parameters defining light source rotation ([theta, phi])
        platform_tilt       (array) two parameters defining platform rotation ([theta, phi])

    Output:
    -------
        (array) spherical coordinates of the source (theta, phi, rho)
    """
    source_pos, light_dir = platform_to_cartesian(plat_coordinates, light_dir,
                            source_center, source_tilt, platform_tilt, height)
    x, y, z = source_pos
    rho = np.linalg.norm(source_pos)
    theta = np.arctan(y/x)
    phi = np.arccos(z/rho)
    return tidy(np.degrees(theta)), tidy(np.degrees(phi)), tidy(rho)


def input_angles_to_platform(theta, phi, height):
    """ Calculates platform coordinates from input angles (theta, phi). The 'input'
        word means this angles are **not corrected**.

    Parameters:
    -----------
        theta:    (array) source position in platform coordinates ([theta, shift]).
        phi:          (array) direction of light vector (default is [0, 0, 1]).
        height       (array) source center in cartesian coordinates (default is [0, 0])

    Output:
    -------
        (array) platform coordinates of the source (theta, shift, height)
    """
    shift = tidy(height*np.tan(np.radians(phi)))
    return theta, shift


def light_dir_from_angles(theta, phi, source_tilt):
    """ Calculates uncorrected light direction in cartesian coordinates. in other
    words, phi gives the direction to the center of the sample as if the platform
    had no positioning errors .

    Parameters:
    -----------
        phi:     (float) azimuthal angle.
        theta:   (float) azimuthal angle.

    Output:
    -------
        (array) platform coordinates of the source (theta, shift, height)
    """
    theta += source_tilt[0]
    phi += source_tilt[1]
    rho = 1
    phi_rad = np.radians(-phi)
    theta_rad = np.radians(theta)
    x = np.sin(phi_rad)*np.cos(theta_rad)*rho
    y = np.sin(phi_rad)*np.sin(theta_rad)*rho
    z = rho*np.cos(phi_rad)
    return x, y, z

def calculate_spot_center(source_position, light_dir):
    """ Center of the light spot on the sample plane.

    Returns
        (int array): [xcenter, ycenter] of the spot on the sample plane.
    """
    x_spot = source_position[0] + light_dir[0]*source_position[2]/light_dir[2]
    y_spot = source_position[1] + light_dir[1]*source_position[2]/light_dir[2]
    return x_spot, y_spot, 0


def spot_image(source_position, light_dir, radius=20, color='r'):
    """ Image of the spot of light on the sample plane.

    Returns
        (int array): [xcenter, ycenter] of the spot on the sample plane.
    """
    color_dict = {'r': 0, 'g': 1, 'b': 2}
    xsize, ysize = [400, 400]
    spot_center = calculate_spot_center(source_position, light_dir)
    spot_center += np.asarray([xsize/2, ysize/2, 0])
    image = np.zeros((xsize, ysize))
    xx, yy = np.meshgrid(range(xsize), range(ysize))
    c = (xx-spot_center[0])**2+(yy-spot_center[1])**2
    image_gray = [c < radius**2]
    image[:, :] = image_gray[0]
    return image
