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

from . import pyfpm.local as local
import pyfpm.coordtrans as ct
import pyfpm.fpmmath as fpmm
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()

out_file = os.path.join(cfg.output_sim,
                        '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now()))
iterator = ct.set_iterator(cfg)
simclient = local.SimClient(cfg=cfg)

fig, ax1 = plt.subplots(1, 1, figsize=(25, 15))
fig.show()
image_dict = dict()
# for index, theta, phi, acqpars in iterator:
for it in iterator:
    print(it['theta'], it['phi'], it['indexes'])
    # iso, shutter_speed, led_power = acqpars
    im_array = simclient.acquire(it['theta'], it['phi'], it['acqpars'])
    image_dict[it['indexes']] = im_array
    ax1.cla()
    img = ax1.imshow(im_array, cmap=plt.get_cmap('hot'), vmin=0, vmax=255)
    fig.canvas.draw()
dt.save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
