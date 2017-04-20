#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File sample.py

Last update: 28/10/2016
To be used as a remote client for a microscope.
Before runing this file make sure there is a server running serve_microscope.py
hosted in some url.

Usage:

"""
from StringIO import StringIO
import time
import os
import datetime

from numpy.fft import fft2, ifft2, fftshift
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import yaml
from scipy import misc, signal


import pyfpm.fpmmath as fpm
from pyfpm.data import save_yaml_metadata
# from pyfpm.data import json_savemeta, json_loadmeta
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates
from pyfpm import implot
import pyfpm.local as local
# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)
simclient = local.SimClient(cfg=cfg)
pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)

iterator = fpm.set_iterator(cfg)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
plt.grid(False)
fig.show()

theta = 45
phi = 5
sim_im_array = simclient.acquire(theta, phi, power=100)
ps_required = fpm.pixel_size_required(cfg.phi[1], cfg.wavelength,
                                      cfg.objective_na)
original_shape = np.shape(sim_im_array)
scale_factor = cfg.pixel_size/ps_required
processing_shape = np.array(original_shape)*scale_factor

pupil_radius = fpm.calculate_pupil_radius(cfg.objective_na, processing_shape[0],
                                          cfg.pixel_size, cfg.wavelength)
pupil = fpm.generate_pupil(theta=0, phi=0,
                           image_size=processing_shape.astype(int),
                           wavelength=cfg.wavelength,
                           pixel_size=ps_required, na=0.01)
sim_im_array = fpm.resize_complex_image(sim_im_array, processing_shape)

pupil_shift = fftshift(pupil)
f_image = fft2(sim_im_array)
f_cut = f_image*pupil_shift
im_rec = np.power(np.abs(ifft2(f_cut)), 1)

sim_im_array = fpm.resize_complex_image(sim_im_array, [150, 150])
im_rec = fpm.resize_complex_image(im_rec, [150, 150])
pupil = fpm.resize_complex_image(pupil, [150, 150])
f_image = fpm.resize_complex_image(f_image, [150, 150])


corr = signal.correlate2d(np.abs(fftshift(f_image)), pupil, mode='same', boundary='fill', fillvalue=0)
# Plotting stuff
ax1.cla(), ax2.cla()
ax1.imshow(np.abs(im_rec), cmap=cm.hot)
ax1.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f'% (np.mean(sim_im_array), phi, theta),
              xy=(0,0), xytext=(80, 20), fontsize=12, color='white')
ax2.imshow(np.abs(corr), cmap=cm.hot, alpha=1.)
fig.canvas.draw()

plt.show()
