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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
import numpy as np
import time
import yaml

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
print(cfg.video_size)

out_file = os.path.join(cfg.output_sample,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
in_file = os.path.join(cfg.output_sample,
                        './2017-04-05_11:36:01.npy')
iterator = fpm.set_iterator(cfg)

image_dict = np.load(in_file)[()]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
plt.grid(False)
fig.show()
for index, theta, phi in iterator:
    power = 100
    sim_im_array = simclient.acquire(theta, phi, power)
    im_array = image_dict[(theta, 0)]
    im_array = fpm.crop_image(im_array, cfg.video_size, 170, 245)
    ax1.cla(), ax2.cla()
    ax1.imshow(im_array, cmap=cm.hot)
    ax1.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f'% (np.mean(im_array), phi, theta),
                 xy=(0,0), xytext=(80,10), fontsize=12, color='white')
    ax2.imshow(sim_im_array, cmap=cm.hot)
    ax2.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f'% (np.mean(sim_im_array), phi, theta),
                 xy=(0,0), xytext=(80,10), fontsize=12, color='white')

    fig.canvas.draw()

plt.show()
