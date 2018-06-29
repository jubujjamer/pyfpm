#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py
Last update: 28/10/2016

Usage:

"""
import shutil
from scipy import misc
import numpy as np
import time
from io import StringIO
import os
## To work with py 2 or
import pyfpm.fpmmath as fpmm

class BaseClient(object):
    def acquire_to(self, filename, theta, phi, power):
        raw = self.acquire(url, theta, phi, power)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(raw, out_file)

    def get_pupil_size(self):
        raise NotImplemented


class Client(BaseClient):
    def __init__(self, camera, laseraim, **metadata):
        self.camera = camera
        self.laseraim = laseraim
        self.metadata = metadata

    def acquire(self, theta=None, phi=None, power=None):
        if theta is not None:
            self.laseraim.move_theta(int(theta))
        if phi is not None:
            self.laseraim.move_phi(int(phi))
        if power is not None:
            self.laseraim.set_laser_power(int(power))
        return self.camera.capture_png()

    def get_pupil_size(self):
        return self.metadata['pupil_size']


class LedMatrixClient(BaseClient):
    def __init__(self, camera, ledmat, **metadata):
        self.camera = camera
        self.led_matrix = ledmat
        self.metadata = metadata

    def acquire(self, nx=None, ny=None, power=None, color=None,
                shutter_speed=100, iso=100, xoff=0, yoff=0):
        self.led_matrix.set_pixel(nx, ny, power, color)
        out_image = self.camera.capture_image(shutter_speed, iso, xoff, yoff)
        # print('yuv to rgb out', out_image)
        return out_image

    def set_pixel(self, nx=None, ny=None, power=None, color=None):
        self.led_matrix.set_pixel(nx, ny, power, color)
        return

class LedClient(BaseClient):
    def __init__(self, camera, ledaim, **metadata):
        self.camera = camera
        self.ledaim = ledaim
        self.metadata = metadata

    def acquire(self, theta=None, phi=None, power=None, color=None):
        if theta is not None:
            print("Moving motor")
            self.ledaim.move_theta(int(theta))
        if phi is not None and power is not None and color is not None:
            print("parameters", phi, power, color)
            self.ledaim.set_parameters(int(phi), int(power), str(color))
        else:
            self.ledaim.set_parameters(0, 0, "red")
        return self.camera.capture_png()

    def complete_scan(self, color=None):
        theta_max = 180
        phi_max = 60
        theta_step = 20
        iterator = iterleds(theta_max, phi_max, theta_step)
        for index, theta, phi, power in iterator:
            print(index, theta, phi, power)
            self.ledaim.move_theta(int(theta))
            self.ledaim.set_parameters(int(phi), int(power), str(color))
            self.camera.capture_png()
        self.reset()

    def get_pupil_size(self):
        return self.metadata['pupil_size']


class Laser3dCalibrate(BaseClient):
    def __init__(self, camera, laser3d):
        self.camera = camera
        self.laser3d = laser3d
        self.cal_data = list()

    def set_parameters(self, theta, phi, shift, power=1, mode='nocorrected'):
        if mode == 'nocorrected':
            print(theta, phi, shift, mode)
            self.laser3d.theta = theta
            self.laser3d.phi = phi
            self.laser3d.shift = shift
            self.laser3d.power = power
        if mode == 'corrected':
            self.laser3d.pc.shift = shift
            self.laser3d.pc.phi = phi
            self.laser3d.pc.theta = theta
            self.laser3d.pc.power = power
            [theta, phi, shift, power] = self.laser3d.pc.parameters_to_platform()
            print(theta, phi, shift, power)
            self.laser3d.theta = theta
            self.laser3d.phi = phi
            self.laser3d.shift = shift
            self.laser3d.power = power
        return

    def capture_image(self):
        return self.camera.capture_png(100000, 400)

    def acquire(self, theta=None, phi=None, shift=None, power=None, color=None,
                shutter_speed=100, iso=100):
        # self.set_parameters(theta, phi, shift, power)
        print("In acquire", theta, phi, shift, power, color)
        self.laser3d.theta = float(theta)
        self.laser3d.phi = float(phi)
        self.laser3d.shift = float(shift)
        self.laser3d.power = float(power)
        print("capturing")
        return self.camera.capture_png(shutter_speed, iso)


    def just_move(self, theta=None, phi=None, shift=None, power=None, color=None):
        if theta is not None:
            self.laser3d.theta = float(theta)
        if phi is not None:
            self.laser3d.phi = float(phi)
        if shift is not None:
            self.laser3d.shift = float(shift)
        if power is not None:
            self.laser3d.power = float(power)
        return

    def calibrate_servo(self, theta=None, phi=None, power=None, color=None):
        return self.camera.capture_png()

    def move_phi(self, phi, mode='relative'):
        phi_init = self.laser3d.phi
        self.laser3d.phi = phi_init + phi
        return

    def move_theta(self, theta, mode='relative'):
        theta_init = self.laser3d.theta
        self.laser3d.theta = theta_init + theta
        return

    def move_shift(self, shift, mode='relative'):
        shift_init = self.laser3d.shift
        self.laser3d.shift = shift_init + shift
        return

    def set_power(self, power):
        print("power", power)
        self.laser3d.power = power
        return

    def get_parameters(self):
        theta = self.laser3d.theta
        phi = self.laser3d.phi
        shift = self.laser3d.shift
        # print("parameters", theta, phi, shift)
        # print("Centered parameters", self.laser3d.pc.parameters_to_platform())
        # print(self.laser3d.pc.shift_adjusted())
        return theta, phi, shift

    def append_parameter(self):
        theta, phi, shift = self.get_parameters()
        self.cal_data.append([theta, phi, shift])
        return

    def get_cal_parameters(self):
        return self.cal_data

class SimClient(BaseClient):
    def __init__(self, cfg):# Y Datos del microscopio
        self.cfg = cfg
        HOME_FOLDER = os.path.expanduser("~/git/pyfpm")
        try:
            self.image_mag = self.load_image(os.path.join(HOME_FOLDER, cfg.input_mag))
            self.image_phase = self.load_image(os.path.join(HOME_FOLDER, cfg.input_phase))
            # Transform into complete field image
            mag_array = self.image_mag
            ph_array = np.pi*(self.image_phase)/np.amax(self.image_phase)
            self.im_array = mag_array*np.exp(1j*ph_array)
        except:
            print('File not found.')
            self.image_mag = None
            self.image_phase = None
        # Some repetitively used parameters
        self.ps = float(cfg.pixel_size)/float(cfg.x)
        self.wavelength = float(cfg.wavelength)
        npx = float(cfg.patch_size[0])
        na = float(cfg.na)
        # NOTE: I had to put 2.2 in the denominator in place of 2 in order to
        # have a ps a bit smaller than strictly necesary, because of rounding errors
        # self.ps_req = self.wavelength/(4*(na+np.sin(np.radians(cfg.phi[1]))))
        self.ps_req = self.ps/2.5
        self.lhscale = self.ps/self.ps_req
        self.lrsize = int(np.floor(npx/self.lhscale))
         # NOTE: I was using ps here, but in the tutorial uses ps_req
        self.pupil_radius = int(np.ceil(self.ps*na*npx/self.wavelength))
        print('radius', self.pupil_radius, self.ps_req*na*npx/self.wavelength)
        ## To convert coordinates to discretized kself.
        # k_discrete = sin(theta)*k0/dk = sin(t)*2pi/l*1/(2*pi/(ps*npx))
        # NOTE: The same observation about ps and ps_req
        self.kdsc = self.ps_req*npx/self.wavelength
        # self.pupil_rad = cfg.pupil_size
        # self.image_size = cfg.video_size

    def load_image(self, input_image):
        """ Loads phase and magnitude input images and crops to patch size.
        """
        with open(input_image, 'rb') as imageFile:
            image_array = misc.imread(imageFile, 'F')
            npx = int(self.cfg.patch_size[0])
            return image_array[0:npx, 0:npx]

    def acquire(self, theta=None, phi=None, acqpars=None):
        """ Returs a simulated acquisition with given acquisition parameters.
        Args:
            theta (float):
            phi (float):
            acqpars (list):     [iso, shutter_speed, led_power]

        Returns:
            (ndarray):          complex 2d array
        """
        theta = float(theta)
        phi = float(phi)
        # fpm.simulate_acquisition(theta, phi, acqpars)
        filtered = fpmm.filter_by_pupil_simulate(self.im_array, theta, phi,
                            self.lrsize, self.pupil_radius, self.kdsc)
        return np.abs(filtered)

    def acquire_ledmatrix(self, nx=None, ny=None, acqpars=None):
        """ Returs a simulated acquisition with given acquisition parameters.
        Args:
            theta (float):
            phi (float):
            acqpars (list):     [iso, shutter_speed, led_power]

        Returns:
            (ndarray):          complex 2d array
        """
        filtered = fpmm.filter_by_pupil_simulate(im_array=self.im_array, nx=nx, ny=ny,
                            lrsize=self.lrsize, cfg=self.cfg, mode='ledmatrix')
        return np.abs(filtered)

    def show_filtered(self, theta=None, phi=None, power=None):
        theta = float(theta)
        phi = float(phi)
        # Image processed using laser aiming data
        return show_image('filtered', self.image_mag, theta, phi, power, self.pupil_rad)

    def show_pupil(self, theta=None, phi=None, power=None):
        theta = float(theta)
        phi = float(phi)
        return show_image('pupil', self.image, theta, phi, power, self.pupil_rad)

    def get_pupil_rad(self):
        return self.pupil_rad


class DummyClient(BaseClient):
    def __init__(self): # Y Datos del microscopio
        # self.init_image = image
        # self.proc_image = image
        self.dumb = 100

    def acquire(self, *args, **kwargs):
        # Return np.array processed using laser aiming data
        return("print")

    def move_phi(self, phi, mode='relative'):
        print("Ooookay")
        return("I'm moving but I'm lazy")

    def move_theta(self, phi, mode='relative'):
        print("Ooookay")
        return("I'm moving but I'm lazy")

    def set_power(self, power):
        print("power", power)
        return

    def get_parameters(self):
        theta = 0
        phi = 0
        shift = 0
        return [theta, phi, shift]
