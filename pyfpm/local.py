#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py
Last update: 28/10/2016

Usage:

"""
import shutil

from fpmmath import filter_by_pupil, set_iterator


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

class SimClient(BaseClient):
    def __init__(self, image, image_size, pupil_rad): # Y Datos del microscopio
        # self.init_image = image
        # self.proc_image = image
        self.image = image
        self.pupil_rad = pupil_rad
        self.image_size = image_size
        # self.pir = PupilIterator(self.init_image, self.size, self.overlap, self.pup_rad)

    def acquire(self, theta=None, phi=None, power=None):
        theta = float(theta)
        phi = float(phi)
        # Return np.array processed using laser aiming data
        return filter_by_pupil(self.image, theta, phi, power, self.pupil_rad, self.image_size)

    def show_filtered(self, theta=None, phi=None, power=None):
        theta = float(theta)
        phi = float(phi)
        # Image processed using laser aiming data
        return show_image('filtered', self.image, theta, phi, power, self.pupil_rad)

    def show_pupil(self, theta=None, phi=None, power=None):
        theta = float(theta)
        phi = float(phi)
        return show_image('pupil', self.image, theta, phi, power, self.pupil_rad)

    def get_pupil_rad(self):
        return self.pupil_rad
