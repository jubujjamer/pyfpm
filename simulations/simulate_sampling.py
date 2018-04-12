#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import datetime

import pyfpm.local as local
import fpmmath as fpmm
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()

out_file = os.path.join(cfg.output_sim,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
iterator = fpmm.set_iterator(cfg)
simclient = local.SimClient(cfg=cfg)

fig, ax1 = plt.subplots(1, 1, figsize=(25, 15))
fig.show()
image_dict = dict()
dt.save_yaml_metadata(out_file, cfg)
for index, theta, phi, acqpars in iterator:
    print(theta, phi)
    iso, shutter_speed, led_power = acqpars
    im_array = simclient.acquire(theta, phi, acqpars)
    image_dict[(theta, phi)] = im_array
    ax1.cla()
    img = ax1.imshow(im_array, cmap=plt.get_cmap('hot'), vmin=0, vmax=255)
    fig.canvas.draw()
np.save(out_file, image_dict)
