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
servo_init = config_dict['servo_init']
shift_step = config_dict['shift_step']
shift_max = config_dict['shift_max']


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
    def __init__(self, theta=0, phi=0, shift=0, height=sample_height, power=1):
        self._theta = np.radians(theta)
        self._phi = np.radians(phi)
        self._shift = shift
        self._height = height
        self._power = power
        # platform defects as taken from config file, check the docs
        # for definitions
        self._platform_tilt = platform_tilt
        self._source_center = source_center
        self._source_tilt = source_tilt

    def get_cartesian(self):
        prime_pos = [self._shift, 0, -self._height]
        origin, direction = rotate(prime_pos, self._theta, self._phi)
        geo_center, point_direction = apply_corrections(origin, direction,
                                                        source_center,
                                                        source_tilt,
                                                        platform_tilt)
        return geo_center, point_direction


    def update_spot_center(self):
        """In 2D cartesian coordinates on the sample plane.
        """
        # Initial position on the prima coordinate system
        geo_center, point_direction = self.get_cartesian()
        self._height = np.abs(geo_center[2])

        # line - plane intersection
        x_spot = geo_center[0] - point_direction[0]*geo_center[2]/point_direction[2]
        y_spot = geo_center[1] - point_direction[1]*geo_center[2]/point_direction[2]
        image_center = np.asarray(video_size)/2
        spot_center = np.array([x_spot, y_spot]) + image_center
        return spot_center.astype(int)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = np.radians(phi-servo_init)
        self.update_spot_center()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = np.radians(theta)
        self.update_spot_center()

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = shift*shift_step
        self.update_spot_center()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height
        self.update_spot_center()

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        self._power = power

    def phi_to_center(self):
        """ Calculates the required phi angle for the spot to be centered given
        theta and shift.
        """
        image_center = np.asarray(video_size)/2
        geo_center, point_direction = self.get_cartesian()
        print geo_center
        # print("centered_coords", center)
        # print("angles", self.theta, self.phi, self.shift, self.height)
        phi_result = np.arccos(geo_center.dot([0, 0, -1])/np.linalg.norm(geo_center))
        print(phi_result, type(geo_center))
        phi_result = np.degrees(phi_result)
        return phi_result


    def spot_image(self, radius=40, color='r'):
        """Image on the sample plane.
        """
        color_dict = {'r': 0, 'g': 1, 'b': 2}
        spot_center = self.update_spot_center()

        image = np.zeros((video_size[1], video_size[0], 3))
        xx, yy = np.meshgrid(range(video_size[0]), range(video_size[1]))
        c = (xx-spot_center[0])**2+(yy-spot_center[1])**2
        image_gray = [c < radius**2]
        image[:, :, color_dict[color]] = image_gray[0]
        return image

    def parameters_to_platform(self):
        """ Corrected values for the parameters_to_platform
        """
        theta = self.theta
        phi = self.phi_to_center() + servo_init
        shift = int(self.shift/shift_step)
        return theta, phi, shift

    def shift_adjusted(self):
        # data = np.load('./out/calibration.npy')
        phi = np.radians([62.0, 62.0, 64.0, 65.0, 67.0, 68.0, 69.0, 70.0, 72.0,
                73.0, 74.0, 76.0, 77.0, 78.0, 80.0])-servo_init
        shift = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0,
        800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0])*shift_step
        m, b = np.polyfit(phi, shift, 1)
        shift_adjusted = int((m*self.phi+b)/shift_step)
        if shift_adjusted < 0:
            shift_adjusted = 0
        if shift_adjusted > shift_max:
            shift_adjusted = shift_max
        return shift_adjusted
