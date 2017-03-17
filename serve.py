#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File serve_microscope.py

Last update: 28/10/2016
To be used as a remote server for a microscope.
Run this file on a server connected to the microcope. This will initialize the
camera and serial devices.

Usage:

"""
import yaml

from pyfpm.web import create_server
import pyfpm.local as local
from pyfpm.devices import Laser3d, Camera
import pyfpm.data as dt
# Find serial controller of the light supply
# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)

config_dict = yaml.load(open('config.yaml', 'r'))
serialport = cfg.serialport
servertype = cfg.servertype
camtype = cfg.camtype

output_file = open(config_dict['output_cal'], "w")
laser3d = Laser3d(port=serialport+str(0))

if servertype == 'sampling':
    try:
        laser3d = Laser3d(port=serialport+str(0))
    except:
        print(serialport+str(0) + " not available, testing with another.")
        laser3d = Laser3d(port=serialport+str(1))
    # Run the camera with open cv
    cam = Camera(video_id=config_dict['video_id'], camtype=camtype)
    client = local.Laser3dCalibrate(cam, laser3d)
elif servertype == 'simulation':
    client = local.DummyClient()

app = create_server(client)
print(config_dict['server_host'])
app.run(host=config_dict['server_host'], debug=config_dict['debug'])
