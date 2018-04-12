#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:
"""

import yaml
import numpy as np
# from itertools import ifilter, product

import pyfpm.data as dt
from pyfpm.fpmmath import set_iterator, translate
import pyfpm.data as dt

cfg = dt.load_config()
print(cfg.max_led_power)

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


class PlatformCoordinates(object):
    """ A class to manage the variables of the moving platform as
        coordinates.
    """
    def __init__(self, theta=0, phi=0, shift=0, height=None, power=1, cfg=None):
        self._theta = np.radians(theta)
        self._phi = np.radians(phi)
        self._shift = shift
        self.cfg = cfg
        if height is None:
            self.height = cfg.sample_height
        else:
            self.height = height
        self.power = power
        if cfg is not None:
            # platform defects as taken from config file, check the docs
            # for definitions
            self.platform_tilt = cfg.platform_tilt
            self.source_center = cfg.source_center
            self.source_tilt = cfg.source_tilt
        self.model = None # no model without data

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

    # @property
    # def height(self):
    #     return self.height
    #
    # @height.setter
    # def height(self, height):
    #     self.height = height
        # self.update_spot_center()

    def source_coordinates(self, mode='cartesian'):
        """ Coordinates of the source. If mode is 'cartesian' are the cartesian
        coordinates with origin on the sample and axes parallel to the sample
        plane. If mode is 'angular' are the corrected theta and phi angles.

        Returns:
            (array): [x0, y0, z0] the source.
            (array): Direction [kx, ky, kz] of the illuminator.
        """

        uncorrected_center = [self._shift, 0, -self.height]
        origin, direction = rotate(uncorrected_center, self._theta, self._phi)
        corr_center, corr_direction = apply_corrections(origin, direction,
                                                        self.source_center,
                                                        self.source_tilt,
                                                        self.platform_tilt)
        if mode == 'cartesian':
            return corr_center, corr_direction
        if mode == 'angular':
            rho = np.sqrt(corr_center[0]**2+corr_center[1]**2)
            if rho == 0:
                corr_theta = 0
            else:
                corr_theta = np.arccos(corr_center[0]/rho)
                if corr_center[1] < 0:
                    corr_theta = 2*np.pi - corr_theta
            corr_phi = np.arctan(rho/self.height)
            return np.degrees(corr_theta), np.degrees(corr_phi)

    def update_spot_center(self):
        """ Center of the light spot on the sample plane.

        Returns
            (int array): [xcenter, ycenter] of the spot on the sample plane.
        """
        # Initial position on the prima coordinate system
        geo_center, point_direction = self.source_coordinates()
        self.height = np.abs(geo_center[2])

        # line - plane intersection
        x_spot = geo_center[0] - point_direction[0]*geo_center[2]/point_direction[2]
        y_spot = geo_center[1] - point_direction[1]*geo_center[2]/point_direction[2]
        image_center = np.asarray(video_size)/2
        spot_center = np.array([x_spot, y_spot]) + image_center
        return spot_center.astype(int)

    def adjust_power(self):
        phi = np.degrees(self.phi)
        power = translate(phi, 0, 90, 10, 255)
        self.power = power
        # if phi < 3:
        #     self.power = 50
        # if phi >= 3 and phi < 10:
        #     self.power = 100
        # if phi > 8:
        #     self.power = 255
        return

    def set_coordinates(self, theta, phi, shift=None, units='degrees'):
        """ This allows the user to set angles in the preferred unit system.
        The default setter is in platform coordinates (raw), but it is
        advisable to use this explicit form.
        """
        if units == 'degrees':
            self._theta = np.radians(theta)
            self._phi = np.radians(phi)
            self._shift = np.tan(self._phi)*self.height
        if units == 'raw':
            self._theta = np.radians(theta*360/cfg.theta_spr)
            self._phi = np.radians(phi*360/cfg.phi_spr)
        if units == 'deg_shift':
            self._theta = np.radians(theta)
            self._phi = np.radians(phi)
            self._shift = shift

    def phi_to_center(self):
        """ Calculates the required phi angle for the spot to be centered given
        theta and shift.
        """
        image_center = np.asarray(self.cfg.video_size)/2
        geo_center, point_direction = self.source_coordinates()
        phi_result = np.arccos(geo_center.dot([0, 0, -1])/np.linalg.norm(geo_center))
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
            model_cfg = dt.load_model_file(cfg.model_name)
            model = model_cfg.model_type
        except:
            print("No model created, run 'generate_model' first")
            model = 'nomodel'

        if model == 'nomodel':
            theta = self.theta
            # phi = self.phi_to_center()
            self.phi
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
                data = np.load(cfg.output_cal)
            except:
                print("generate calibration data first")
                return
            phi = np.array([d[1] for d in data])-servo_init
            shift = np.array([d[2] for d in data])
            slope, origin = np.polyfit(phi, shift, 1)
            model = {'model_type': 'shift_fit',
                     'slope': float(slope),
                     'origin': float(origin)}
            dt.save_model(cfg.model_name, model)
            return
        if model == 'normal':
            model = {'model_type': 'normal'}
            dt.save_model(cfg.model_name, model)
            return

        if model == 'nomodel':
            model = {'model_type': 'nomodel'}
            dt.save_model(cfg.model_name, model)
            return
