#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:
"""

import yaml
import numpy as np

config_dict = yaml.load(open('config.yaml', 'r'))
servo_init = config_dict['servo_init']
sample_height = config_dict['sample_height']
platform_tilt = config_dict['platform_tilt']
source_center = config_dict['source_center']
source_tilt = config_dict['source_tilt']


class PlatformCoordinates(object):
    """ A class to manage the variables of the moving platform as
        coordinates.
    """
    def __init__(self, theta=0, phi=0, shift=0, height=sample_height):
        self._theta = np.radians(theta)
        self._phi = np.radians(phi)
        self._shift = shift
        self._height = height
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
        self.phi = phi

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

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

    def expected_spot_center(self):
        """In 2D cartesian coordinates on the sample plane.
        """
        rot_mat = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                            [np.sin(self.theta), np.cos(self.theta)]])
        x_prime =  self.shift + self.height*np.tan(self.phi)+ source_center[0]
        y_prime = self.source_center[1]
        x, y =  rot_mat.dot([x_prime, y_prime])
        return [x, y]
