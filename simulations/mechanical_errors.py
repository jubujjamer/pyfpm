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
# Platform parameters definition
height = 100  # distance from the sample plane to the platform
plat_coordinates = [0, 10]  # [theta (º), shift (mm)]
light_dir = [0, 0, 1]
source_center = [0, 0]
source_tilt = [0, 0]
platform_tilt = [0, 0]

image_dict = dict()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()
for index, theta, phi, acqpars in iterator:
    iso, shutter_speed, led_power = acqpars
    # pupil = generate_pupil(theta, phi, power, cfg.video_size,
    #                        cfg.wavelength, cfg.pixel_size, cfg.objective_na)
    im_array = simclient.acquire(theta, phi, acqpars)
    image_dict[(theta, phi)] = im_array
    ax1.cla(), ax2.cla()
    img = ax1.imshow(im_array, cmap=plt.get_cmap('hot'), vmin=0, vmax=255)
    if index == 0:
        fig.colorbar(img)
    # plt.xlim([0,450])
    # plt.ylim([0,2.5])
    ax1.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f' % (np.mean(im_array), phi, theta),
                 xy=(0,0), xytext=(80,10), fontsize=12, color='white')
    fig.canvas.draw()
