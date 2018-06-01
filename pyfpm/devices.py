#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:

"""
import os
import serial
import time
import yaml
from io import BytesIO

# import picamera
import numpy as np
from . import camera
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import pyfpm.data as dt

# from pyfpm.cordinates import PlatformCoordinates

cfg = dt.load_config()


class LedMatrixRGB(object):
    """ Gives interface for the 3D.
    """
    def __init__(self, nx=None, ny=None, color=None, power=0):
        self.matrix_opts = RGBMatrixOptions()
        self.set_power(0)
        self.nx = nx
        self.ny = ny
        self.set_options()

    def set_options(self):
        self.matrix_opts.rows = 32
        self.matrix_opts.chain_length = 1
        # self.matrix_opts.parallel = 1
        self.matrix_opts.pwm_bits = 4
        self.matrix_opts.pwm_lsb_nanoseconds = 900
        self.matrix_opts.drop_privileges = True
        self.matrix_opts.hardware_mapping = 'regular'
        # self.matrix_opts.gpio_slowdown=1
        self.matrix = RGBMatrix(options=self.matrix_opts)

    def set_power(self, power):
        """ Adjusts power manually between 0-255.
        """
        print('doit')

    def set_pixel(self, x, y, power, color):
        self.matrix.Clear()
        nx, ny, power = int(x), int(y), int(power)
        if color == 'R':
            self.matrix.SetPixel(nx, ny, power, 0, 0)
        if color == 'G':
            self.matrix.SetPixel(nx, ny, 0, power, 0)
        if color == 'B':
            self.matrix.SetPixel(nx, ny, 0, 0, power)

    def __del__(self):
        print('Goodbye')


class LaserAim(object):
    """ Aimed Led implemented by means of a servo and a stepper.
    Two degrees of freedom (theta - the stepper - and phi - the servo).
    """
    ZERO_ANGLE = 90
    # Steps covering 90 degrees
    CONV_FACTOR = 3000. / 90.

    def __init__(self, port='/dev/ttyACM0', theta=0, phi=0, power=0):
        self.ser_dev = serial.Serial(port, baudrate=9600,
                                     parity=serial.PARITY_ODD, timeout=1)
        self.set_laser_power(power)
        self.move_theta(theta)
        self.move_phi(phi)

    def set_laser_power(self, laser_power):
        """ Adjusts laser power manually between 0-255.
        """
        serial_message = "POW %d" % int(laser_power)
        self._serial_write(serial_message)
        time.sleep(.1)

    def _move_theta_motor(self, pos):
        serial_message = "ROT %d" % int(pos)
        self._serial_write(serial_message)
        time.sleep(.1)

    def move_theta(self, angle):
        """ Adjusts platform angle in degrees
        """
        self._move_theta_motor(angle*self.CONV_FACTOR)

    def move_phi(self, angle):
        serial_message = "TIL %d" % int(angle+self.ZERO_ANGLE)
        self._serial_write(serial_message)
        time.sleep(.1)

        self.move_theta(0)
        self.move_phi(0)
        self.laser_power(0)

    def _serial_write(self, message):
        self.ser_dev.write(message + " \r".encode())
        print(message + " \n")

    def __del__(self):
        self.ser_dev.close()


class LedAim(object):
    """ This was used by the 9 Nopixels put in line in a 'U' shaped platform.
    Two degrees of freedom. It used a for the theta motion and the digital led
    selection.
    """
    ZERO_ANGLE = 90
    # Steps covering 90 degrees
    CONV_FACTOR = 3000. / 90.

    def __init__(self, port='/dev/ttyACM0', theta=0, phi=0, power=0,
                 color="red"):
        self.ser_dev = serial.Serial(port, baudrate=9600,
                                     parity=serial.PARITY_ODD, timeout=1)
        self.move_theta(theta)
        self.set_parameters(phi, power, color)

    def _move_theta_motor(self, pos):
        serial_message = "ROT %d" % int(pos)
        self._serial_write(serial_message)
        time.sleep(.1)

    def move_theta(self, angle):
        """ Adjusts platform angle in degrees
        """
        self._move_theta_motor(angle*self.CONV_FACTOR)

    def set_led(self, led, mode, power):
        """ There are 9 leds, selectable by the led argument.
        Four modes are implemented:
            0: red only, 1: green only, 2 :blue only, 3: white
        power goes from 0 to 255
        """
        serial_message = "LED %d %d %d" % (int(led), int(mode), int(power))
        self._serial_write(serial_message)

    def set_parameters(self, phi, power, color):
        """ Set acquisition as the client requires.
        """
        allowed_intensities = range(256)
        mode = {"red": 0, "green": 1, "blue": 2, "white": 3}
        allowed_modes = [0, 1, 2, 3]  # See the ditionary above
        allowed_phi = {-80: 8, -60: 7, -40: 6, -20: 5,
                       0: 4, 20: 3, 40: 2, 60: 1, 80: 0}
        led = allowed_phi[min(allowed_phi, key=lambda x: abs(x-phi))]
        if not mode[color] in allowed_modes or not power in allowed_intensities:
            serial_message = "LED %d %d %d" % (0, 0, 0)
            print("Please set the parameters correctly. I'll leave it all shut.")
        else: serial_message = "LED %d %d %d" % (int(led), int(mode[color]), int(power))
        self._serial_write(serial_message)

    def reset(self):
        self._move_theta_motor(0)
        serial_message = "LED %d %d %d" % (0, 0, 0)
        self._serial_write(serial_message)

    def _serial_write(self, message):
        self.ser_dev.write(message + " \r".encode())
        print(message + " \n")

    def __del__(self):
        self.ser_dev.close()


class Laser3d(object):
    """ Latest used moving LED (despite it being called Laer3D).
        Three degrees of freedom (theta, shift, phi) by means of thre steppers,
        plus the led parameters.
    """

    def __init__(self, port=None, theta=None, phi=None, shift=None, power=None):
        self._ser_dev = None
        self._theta = None
        self._phi = None
        self._shift = None
        self._tps = None  # [theta, phi, shift] coordinates
        self._power = None
        # self.pc = PlatformCoordinates()
        self.set_parameters(port, theta, phi, shift, power)

    def set_parameters(self, port, theta, phi, shift, power):
        if port is None:
            port = '/dev/ttyACM0'
        if theta is None:
            theta = 0
        if phi is None:
            phi = 0
        if shift is None:
            shift = 0
        if power is None:
            power = 0
            self._ser_dev = serial.Serial(port, baudrate=9600, timeout=1)
        self._theta = theta
        self._phi = phi
        self._shift = shift
        self._tps = [theta, phi, shift]
        self._power = power
        # self.pc.theta = theta
        # self.pc.shift = shift
        # self.pc.phi = phi

    @property
    def phi(self):
        return self._tps[1]

    @phi.setter
    def phi(self, phi):
        self.move_to(self._theta, phi, self._shift)

    @property
    def theta(self):
        return self._tps[0]

    @theta.setter
    def theta(self, theta):
        self.move_to(theta, self._phi, self._shift)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        self.set_power(power)

    @property
    def shift(self):
        return self._tps[3]

    @shift.setter
    def shift(self, shift):
        self.move_to(self._theta, self._phi, shift)

    def move_to(self, theta, phi, shift):
        # ser_dev.flushInput()
        tcom = 'STMOV 1 %i' % shift
        pcom = 'STMOV 2 %i' % theta
        scom = 'STMOV 3 %i' % phi
        for command in [tcom, pcom, scom]:
            self._send_command(command)
        self._theta = theta
        self._phi = phi
        self._shift = shift
        self._tps = [theta, phi, shift]
        # self.pc.theta = theta
        # self.pc.shift = shift
        # self.pc.phi = phi

    def set_power(self, power):
        lcom = 'LPOW %i' % power
        self._send_command(lcom)
        self._power = power

    # def get_centered_parameters(self):
    #     pc = PlatformCoordinates(self._tps)
    #     phi = self.pc.phi_to_center()
    #     return self.theta, phi, self.shift

    def _send_command(self, command):
        # ser_dev.flushOutput()
        self._ser_dev.write(command + '\r'.encode())
        out = self._ser_dev.read()
        i = 0
        while out != '0' and i < 10:
            out = self._ser_dev.read()
            # print(out == '0', i)
            i += 1
        # ser_dev.flush()

    def __del__(self):
        self._ser_dev.close()


class Camera(object):
    """Microscope camera interface with opencv.

    Setting parameters:
    CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    CV_CAP_PROP_FPS Frame rate.
    CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    CV_CAP_PROP_HUE Hue of the image (only for cameras).
    CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    CV_CAP_PROP_ISO_SPEED The ISO speed of the camera (note: only supported by
    DC1394 v 2.x backend currently)
    CV_CAP_PROP_BUFFERSIZE Amount of frames stored in internal buffer memory
    (note: only supported by DC1394 v 2.x backend currently)

    Another opption is using a system call:
    v4l2-ctl -d /dev/video1 -c exposure_auto=1 -c exposure_auto_priority=0
             -c exposure_absolute=10
    """
    def __init__(self, video_id=0, camtype='picamera'):
        self.camtype = camtype
        if(camtype == 'opencv'):
            import cv2
            cap = cv2.VideoCapture(video_id)
            self.cap = cap
            self.cap.open(video_id)
            self.config_cap()
        elif(camtype == 'picamera'):
            self.cap = camera.RaspiStill(tmpfile='tmp.png', bin='raspistill',
                awb='off', format='png', width=cfg.video_size[1],
                height=cfg.video_size[0], timeacq=100, nopreview='-n',
                iso=400, shutter_speed=1000)
        else:
            self.cap = None


    def config_cap(self):
        prop_dict = {0: 'CAP_PROP_POS_MSEC',
                     1: 'CAP_PROP_POS_FRAMES',
                     2: 'CAP_PROP_POS_AVI_RATIO',
                     3: 'CAP_PROP_FRAME_WIDTH',
                     4: 'CAP_PROP_FRAME_HEIGHT',
                     5: 'CAP_PROP_FPS',
                     6: 'CAP_PROP_FOURCC',
                     7: 'CAP_PROP_FRAME_COUNT',
                     8: 'CAP_PROP_FORMAT',
                     9: 'CAP_PROP_MODE',
                     10: 'CAP_PROP_BRIGHTNESS',
                     11: 'CAP_PROP_CONTRAST',
                     12: 'CAP_PROP_SATURATION',
                     13: 'CAP_PROP_HUE',
                     14: 'CAP_PROP_GAIN',
                     15: 'CAP_PROP_EXPOSURE'}
        # props = [a for a in dir(cv2) if "PROP" in a]
        # for p in range(0, 16):
        #     if self.cap.get(p) != -10.0:
        #         print p, self.cap.get(p), prop_dict[p]
        self.cap.set(10, 0.5)
        # self.cap.set(cv2.CAP_PROP_CONTRAST, 0.7)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, 100)
        # print cap.get(cv2.CAP_PROP_EXPOSURE)

    def capture_png(self, shutter_speed, iso):
        camtype = self.camtype
        if(camtype == 'opencv'):
            time.sleep(.5)
            # Don't know why it doesn't updates up to the 5th reading
            # a buffer flush thing
            for i in range(5):
                ret, frame = self.cap.read()
            ret, buf = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return bytearray(buf)
        elif(camtype == 'picamera'):
            if shutter_speed is not None:
                self.cap.shutter_speed = shutter_speed
            if iso is not None:
                self.cap.iso = iso
                print(self.cap.iso)
            return self.cap.acquire()
        else:
            return None


    def __del__(self):
        # When everything done, release the capture
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except:
            print("Not necessary")
