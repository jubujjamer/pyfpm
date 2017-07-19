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

for index, theta, phi in iterator:
    print(index)
