#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:

"""
import serial
import time

import cv2


class LaserAim(object):
    """Hello
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

    ZERO_ANGLE = 90
    # Steps covering 90 degrees
    CONV_FACTOR = 3000. / 90.

    def __init__(self, port='/dev/ttyACM0', theta=0, phi=0, power=0,
                 color="red"):
	"""
	"""
        self.ser_dev = serial.Serial(port, baudrate=9600, parity=serial.PARITY_ODD, timeout=1)
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
        allowed_modes = [0,1,2,3]

        mode = {"red": 0, "green": 1, "blue":2, "white":3}
        allowed_phi = {-80:8,-60:7,-40:6,-20:5,0:4,20:3,40:2,60:1,80:0}
        led = allowed_phi[min(allowed_phi, key=lambda x: abs(x-phi))]

        if not mode[color] in allowed_modes or not  power in allowed_intensities:
            serial_message = "LED %d %d %d" % (0, 0, 0)
            print("Please set the parameters correctly. I'll leave it all shut.")
        else: serial_message = "LED %d %d %d" % (int(led), int(mode[color]), int(power))
        self._serial_write(serial_message)

    def _serial_write(self, message):
        self.ser_dev.write(message + " \r".encode())
        print(message + " \n")

    def __del__(self):
        self.ser_dev.close()

class Camera(object):
    def __init__(self, video_id = 0):
        self.cap = cv2.VideoCapture(video_id)

    def capture_png(self):
        print("Adding a time delay to leave time the led to set.")
        time.sleep(.5)
        for i in range(5):
            ret, frame = self.cap.read()
        ret, buf = cv2.imencode('.png', frame)
        return buf.tobytes()

    def __del__(self):
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()
