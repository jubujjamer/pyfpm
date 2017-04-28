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
cfg = dt.load_config()

output_file = open(cfg.output_cal, "w")
# laser3d = Laser3d(port=serialport+str(0))

if cfg.servertype == 'sampling':
    laser3d = Laser3d(port=cfg.serialport)
    # Run the camera with open cv
    cam = Camera(video_id=cfg.video_id, camtype=cfg.camtype)
    client = local.Laser3dCalibrate(cam, laser3d)
elif cfg.servertype == 'simulation':
    client = local.DummyClient()

app = create_server(client)
app.run(host=cfg.server_host, debug=cfg.debug)
