#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:
"""

import yaml
import numpy as np
from itertools import ifilter, product

import pyfpm.data as dt

CONFIG_FILE = 'config.yaml'
config_dict = yaml.load(open(CONFIG_FILE, 'r'))
cfg = dt.load_config(CONFIG_FILE)

servo_init = cfg.servo_init
sample_height = config_dict['sample_height']
platform_tilt = config_dict['platform_tilt']
source_center = config_dict['source_center']
source_tilt = config_dict['source_tilt']
video_size = config_dict['video_size']
servo_init = config_dict['servo_init']
shift_step = config_dict['shift_step']
shift_max = config_dict['shift_max']
model_file = config_dict['model_file']
cal_file = config_dict['output_cal']
theta_spr = config_dict['theta_spr']


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
    def __init__(self, theta=0, phi=0, tps=None, shift=0, height=sample_height, power=1):
        self._theta = np.radians(theta)
        self._phi = np.radians(phi)
        self._shift = shift
        self._tps = [np.radians(theta), np.radians(phi), shift]
        if tps is not None:
            self._tps = tps
        self._height = height
        self._power = power
        # platform defects as taken from config file, check the docs
        # for definitions
        self._platform_tilt = platform_tilt
        self._source_center = source_center
        self._source_tilt = source_tilt
        self.model = None # no model without data

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
        self._phi = np.radians(phi*360/cfg.phi_spr)
        self.update_spot_center()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = np.radians(theta*360/theta_spr)
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

    def adjust_power(self):
        phi = np.degrees(self.phi)
        if phi < 3:
            self.power = 50
        if phi >= 3 and phi < 10:
            self.power = 100
        if phi > 10:
            self.power = 255
        return

    def set_coordinates(self, theta, phi, units='degrees'):
        """ This allows the user to set angles in the preferred unit system.
        The default setter is in platform coordinates (raw), but it is advisable to
        use this explicit form.
        """
        if units == 'degrees':
            self._theta = np.radians(theta)
            self._phi = np.radians(phi)
            self._tps = [np.radians(theta), np.radians(phi), None]
        if units == 'raw':
            self._theta = np.radians(theta*360/cfg.theta_spr)
            self._phi = np.radians(phi*360/cfg.phi_spr)
            self._tps = [np.radians(theta*360/cfg.theta_spr),
                         np.radians(phi*360/cfg.phi_spr),
                         None]

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
        try:
            model_dict = yaml.load(open(model_file, 'r'))
        except:
            print "No model created, run 'generate_model' first"
            return
        model = model_dict['model_type']

        if model == 'nomodel':
            theta = self.theta
            phi = self.phi_to_center()
            shift_adjusted = self.shift
            power = self.power

        elif model == 'shift_fit':
            slope = model_dict['slope']
            origin = model_dict['origin']
            phi_degrees = np.degrees(self.phi)
            shift_adjusted = int((slope*phi_degrees + origin))
            if shift_adjusted < 0:
                shift_adjusted = 0
            if shift_adjusted > shift_max:
                shift_adjusted = shift_max
            shift = shift_adjusted*cfg.shift_step
            theta = self.theta
            phi = int(np.degrees(self.phi)*cfg.phi_spr/(2*np.pi))
            self.adjust_power()
            power = self.power

        elif model == 'normal':
            shift_adjusted = np.arctan(self.phi)*cfg.sample_height
            if shift_adjusted < 0:
                shift_adjusted = 0
            if shift_adjusted > shift_max:
                shift_adjusted = shift_max
        shift_plat = int(shift_adjusted/cfg.shift_step)
        theta_plat = int(self.theta*cfg.theta_spr/(2*np.pi))
        phi_plat = int(self.phi*cfg.phi_spr/(2*np.pi))
        self.adjust_power()
        power_plat = self.power
        shift_plat = min(cfg.shift_max, shift_plat)

        return theta_plat, phi_plat, shift_plat, power_plat

    def generate_model(self, model='normal'):
        if model == 'shift_fit':
            try:
                data = np.load(cal_file)
            except:
                print("generate calibration data first")
                return
            phi = np.array([d[1] for d in data])-servo_init
            shift = np.array([d[2] for d in data])
            print(data)
            slope, origin = np.polyfit(phi, shift, 1)
            model = {'model_type': 'shift_fit',
                     'slope': float(slope),
                     'origin': float(origin)}
            with open(model_file, 'w') as outfile:
                yaml.dump(model, outfile, default_flow_style=False)
            return
        if model == 'normal':
            model = {'model_type': 'normal'}
            with open(model_file, 'w') as outfile:
                yaml.dump(model, outfile, default_flow_style=False)
            return
