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
# Simulation parameters
cfg = dt.load_config()

USE_RPYC = True

# output_file = open(cfg.output_cal, "w")
# laser3d = Laser3d(port=serialport+str(0))

if cfg.servertype == 'sampling':
    # laser3d = Laser3d(port=cfg.serialport)
    if USE_RPYC:
        import rpyc
        conn = rpyc.connect("localhost", 18861, config={"allow_all_attrs": True})
        ledmat = conn.root.get_instance()
    else:
        from pyfpm.devices import LedMatrixRGB
        ledmat = LedMatrixRGB()
    # Run the camera with open cv
    cam = Camera(video_id=cfg.video_id, camtype=cfg.camtype)
    client = local.LedMatrixClient(cam, ledmat)
    # client = local.Laser3dCalibrate(cam, laser3d)
elif cfg.servertype == 'simulation':
    client = local.SimClient(cfg)

app = create_server(client)
app.run(host=cfg.server_host, debug=cfg.debug)
