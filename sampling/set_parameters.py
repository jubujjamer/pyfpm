#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import os

import numpy as np
import itertools as it

from pyfpm import web
from pyfpm.fpmmath import set_iterator
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)
pc = PlatformCoordinates(cfg=cfg)
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)
# Start image acquisition
for index, theta, phi in iterator:
    print(theta, phi)
    pc.set_coordinates(theta, phi, units='degrees')
    [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
    client.just_move(theta_plat, phi_plat, shift_plat, power=0)
