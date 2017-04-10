#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import sys
sys.path.append('..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from pyfpm.fpmmath import *
import pyfpm.data as dt
from pyfpm.data import save_yaml_metadata
import pyfpm.local as local

# Simulation parameters
CONFIG_FILE = '../config.yaml'
cfg = dt.load_config(CONFIG_FILE)
client = local.SimClient(cfg=cfg)

iterator = set_iterator(cfg)

image_dict = dict()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()
for index, theta, phi, power in iterator:
    pupil = generate_pupil(theta, phi, power, cfg.video_size,
                           cfg.wavelength, cfg.pixel_size, cfg.objective_na)
    im_array = client.acquire(theta, phi, power)
    if im_array is None:
        break
    image_dict[(theta, phi)] = im_array
    ax1.cla(), ax2.cla()
    ax1.imshow(im_array, cmap=plt.get_cmap('hot'))
    ax2.imshow(pupil, cmap=plt.get_cmap('hot'))
    fig.canvas.draw()
# save_yaml_metadata(dt.generate_out_file(cfg.output_sim), cfg)
# np.save(dt.generate_out_file(cfg.output_sim), image_dict)
