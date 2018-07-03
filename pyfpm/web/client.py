##########################################################
# File client.py
# Author:
# Date:
#
##########################################################
import requests
from flask import request
import numpy as np

from ..local import BaseClient


class Client(BaseClient):

    def __init__(self, url):
        self.url = url
        # self.metadata = requests.get(self.url + '/metadata').json()

    def acquire_ledmatrix(self, nx=0, ny=0, power=255, color='G', shutter_speed=100, iso=100, xoff=0, yoff=0):
        filetype = 'yuv'
        if filetype == 'png':
            response = requests.get(self.url +
                                '/acquire_ledmatrix/%d/%d/%d/%s/%d/%d' % (nx, ny, power, color,
                                shutter_speed, iso),
                                stream=True)
            if response.status_code == 200:
                return response.raw()
            else:
                print("Failed to load webpage")
        if filetype == 'yuv':
            response = requests.get(self.url +
                                    '/acquire_ledmatrix/%d/%d/%d/%s/%d/%d/%d/%d' % (nx, ny, power, color,
                                    shutter_speed, iso, xoff, yoff))
            if response.status_code == 200:
                json_receive = response.json()
                return json_receive
            else:
                print("Failed to load webpage")

    def acquire_ledmatrix_pattern(self, pattern=None,power=255, color='G', shutter_speed=100, iso=100, xoff=0, yoff=0):
        """ This has been directly implemented in yuv. It passes a binary code.
        """
        response = requests.get(self.url +
                                '/acquire_ledmatrix_pattern/%s/%d/%s/%d/%d/%d/%d' % (pattern, power, color,
                                shutter_speed, iso, xoff, yoff))
        if response.status_code == 200:
            json_receive = response.json()
            return json_receive
        else:
            print("Failed to load webpage")

    def set_pixel(self, nx=0, ny=0, power=255, color='G'):
        response = requests.get(self.url +
                                '/set_pixel/%d/%d/%d/%s/' % (nx, ny, power, color),
                                stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")

    def acquire(self, theta, phi, shift=0, power=100, color='green', shutter_speed=100, iso=100):
        response = requests.get(self.url +
                                '/acquire/%d/%d/%d/%d/%s/%d/%d' % (theta, phi, shift, power, color,
                                shutter_speed, iso),
                                stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")

    def just_move(self, theta, phi, shift, power, color='green'):
        response = requests.get(self.url +
                                '/just_move/%d/%d/%d/%d/%s' % (theta, phi, shift, power, color),
                                stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")

    def complete_scan(self, color):
        print(self.url + '/%s' % (color))
        response = requests.get(self.url + '/%s' % (color), stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")

    def get_pupil_size(self):
        return self.metadata['pupil_size']

    def get_camera_picture(self):
        print(self.url + '/testcam')
        response = requests.get(self.url + '/testcam', stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")
