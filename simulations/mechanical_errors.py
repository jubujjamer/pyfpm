#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File sample.py

Last update: 28/10/2016
To be used as a remote client for a microscope.
Before runing this file make sure there is a server running serve_microscope.py
hosted in some url.

Usage:

"""
from numpy.fft import fft2, ifft2, fftshift
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import signal

import pyfpm.fpmmath as fpm
import pyfpm.data as dt
import pyfpm.coordtrans as ct
import pyfpm.local as local

# Simulation parameters
cfg = dt.load_config()
simclient = local.SimClient(cfg=cfg)

iterator = fpm.set_iterator(cfg)

# for index, theta, phi in iterator:
#      apply_corrections(origin, light_dir, source_center,
#                           source_tilt, platform_tilt)
#     print(index)

height = 100
plat_coordinates = [0, 10]
light_dir = [0, 0, 1]
source_center = [0, 0]
source_tilt = [0, 0]
platform_tilt = [0, 0]
print(ct.platform_to_spherical(plat_coordinates, light_dir, source_center, source_tilt, platform_tilt, height))
print(ct.calculate_spot_center(plat_coordinates, light_dir, source_center,
                              source_tilt, platform_tilt, height))
