#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import sys
import os
import datetime
sys.path.append('..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from pyfpm.fpmmath import *
import pyfpm.data as dt
from pyfpm.data import save_yaml_metadata
import pyfpm.local as local

# Simulation parameters
CONFIG_FILE = '../config.yaml'
cfg = dt.load_config(CONFIG_FILE)
client = local.SimClient(cfg=cfg)

out_file = os.path.join('.'+cfg.output_sim,
                        '{:%Y-%m-%d_%H%M%S}'.format(datetime.datetime.now()))
iterator = set_iterator(cfg)

image_size = cfg.video_size
image_dict = dict()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()
for index, theta, phi, power in iterator:
    pupil_radius = calculate_pupil_radius(cfg)
    # print(pupil_radius)
    pupil = generate_pupil(theta, phi, power, pupil_radius, image_size)
    im_array = client.acquire(theta, phi, power)
    image_dict[(theta, phi)] = im_array
    ax1.cla(), ax2.cla()
    ax1.imshow(im_array, cmap=plt.get_cmap('hot'))
    ax2.imshow(pupil, cmap=plt.get_cmap('hot'))
    fig.canvas.draw()
save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
