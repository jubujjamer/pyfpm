#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File serve_microscope.py

Last update: 28/10/2016
To be used as a remote server for a microscope.
Run this file on a server connected to the microcope. This will initialize the
camera and serial devices.

Usage:

"""
from pyfpm.web import create_server
import pyfpm.local as local
from pyfpm.devices import LedAim, Camera

# Find serial controller of the light supply
ser_dev = '/dev/ttyACM'
try:
    ledaim = LedAim(port=ser_dev+str(0), theta=0, phi=0, power=0, color="red")
except:
    print(ser_dev+str(0) + " not available, testing with another.")
    ledaim = LedAim(port=ser_dev+str(1), theta=0, phi=0, power=0, color="red")
# Run the camera with open cv
cam = Camera(video_id=1)
# client = local.LedClient(cam, ledaim)
# app = create_server(client)
# app.run(host='0.0.0.0')
