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
import mayavi.mlab as mlab

import pyfpm.fpmmath as fpm
import pyfpm.data as dt
import pyfpm.coordtrans as ct
import pyfpm.local as local

# Simulation parameters
cfg = dt.load_config()
simclient = local.SimClient(cfg=cfg)
iterator = fpm.set_iterator(cfg)
# Platform parameters definition
wlen = float(cfg.wavelength)
height = 100  # distance from the sample plane to the platform
plat_coordinates = [0, 0]  # [theta (º), shift (mm)]
light_dir = [0, 0, 1]
source_center = [0, 0]
source_tilt = [0, 0]
platform_tilt = [0, 0]
acqpars = [0, 0, 0]
xx, yy, transference = fpm.simulate_sample(cfg)
# Light beam modeling
t, p = [0, .1]
lb = fpm.laser_beam_simulation(xx, yy, t, p, acqpars, cfg)
Sp = lb*transference
mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
mlab.surf(xx, yy, np.angle(Sp)*xx.max()/5)
mlab.show()
# plt.imshow(np.angle(Sp))
# plt.show()
