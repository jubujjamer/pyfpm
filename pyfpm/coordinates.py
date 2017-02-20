#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:
"""

import yaml
import numpy as np
from itertools import ifilter, product

config_dict = yaml.load(open('config.yaml', 'r'))
servo_init = config_dict['servo_init']
sample_height = config_dict['sample_height']
platform_tilt = config_dict['platform_tilt']
source_center = config_dict['source_center']
source_tilt = config_dict['source_tilt']
video_size = config_dict['video_size']


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


class PlatformCoordinates(object):
    """ A class to manage the variables of the moving platform as
        coordinates.
    """
    def __init__(self, theta=0, phi=0, shift=0, height=sample_height):
        self._theta = np.radians(theta)
        self._phi = np.radians(phi)
        self._shift = shift
        self._height = height
        self._centered_coords = list()
        # platform defects as taken from config file, check the docs
        # for definitions
        self._platform_tilt = platform_tilt
        self._source_center = source_center
        self._source_tilt = source_tilt

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = np.radians(phi)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = np.radians(theta)

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = shift

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def source_center(self):
        return self._source_center

    @source_center.setter
    def source_center(self, source_center):
        self._source_center = source_center

    @property
    def centered_coords(self):
        return self._centered_coords

    @centered_coords.setter
    def phi_centered(self, _centered_coords):
        self._centered_coords = centered_coords

    def calculate_spot_center(self):
        """In 2D cartesian coordinates on the sample plane.
        """
        # Initial position on the prima coordinate system
        prime_pos = [self.shift, 0, -self.height]
        origin, direction = rotate(prime_pos, self.theta, self.phi)
        origin_corr, dir_corr = apply_corrections(origin, direction,
                                                  source_center,
                                                  source_tilt,
                                                  platform_tilt)
        # line - plane intersection
        x_spot = origin_corr[0] - dir_corr[0]*origin_corr[2]/dir_corr[2]
        y_spot = origin_corr[1] - dir_corr[1]*origin_corr[2]/dir_corr[2]
        image_center = np.asarray(video_size)/2
        spot_center = np.array([x_spot, y_spot]) + image_center
        return spot_center.astype(int)

    def phi_to_center(self):
        """ Calculates the required phi angle for the spot to be centered given
        theta and shift.
        """
        image_center = np.asarray(video_size)/2
        print(self.phi)
        original_phi = self.phi
        c = list()
        for phi in range(-70, 70):
            self.phi = phi
            # print(self.phi)
            spot_center = self.calculate_spot_center()
            c.append([np.linalg.norm((spot_center-image_center)), phi])
            # print(spot_center, image_center, phi, np.linalg.norm((spot_center-image_center)))
        c = np.array(c)
        argmin = np.argmin(c, axis=0)[0]
        phi_min = c[argmin][1]
        self.phi = np.degrees(original_phi)
        self.centered_coords.append([self.theta, phi_min, self.shift])
        return phi_min


    def spot_image(self, radius=40, color='r'):
        """Image on the sample plane.

        """
        color_dict = {'r': 0, 'g': 1, 'b': 2}
        spot_center = self.calculate_spot_center()

        image = np.zeros((video_size[1], video_size[0], 3))
        xx, yy = np.meshgrid(range(video_size[0]), range(video_size[1]))
        c = (xx-spot_center[0])**2+(yy-spot_center[1])**2
        image_gray = [c < radius**2]
        image[:, :, color_dict[color]] = image_gray[0]
        return image
