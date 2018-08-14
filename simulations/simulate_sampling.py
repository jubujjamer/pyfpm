#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import datetime

import pyfpm.local as local
import pyfpm.coordtrans as ct
import pyfpm.fpmmath as fpmm
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(fname = 'simtest.npy')
iterator = ct.set_iterator(cfg)
simclient = local.SimClient(cfg=cfg)

fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fig.show()
image_dict = dict()
# First take DPC images
im_up = simclient.acquire_pattern(angle=0, acqpars=None, pupil_radius=fpmm.get_pupil_radius(cfg))
image_dict[(0, -1)] = im_up
im_down = simclient.acquire_pattern(angle=180, acqpars=None, pupil_radius=fpmm.get_pupil_radius(cfg))
image_dict[(-1, 0)] = im_down

for it in iterator:
    print('kx: %.1f ky: %.1f' % (it['nx'], it['ny']))
    im_array = simclient.acquire_ledmatrix(nx=it['nx'], ny=it['ny'],
               acqpars=it['acqpars'], pupil_radius = fpmm.get_pupil_radius(cfg))
    image_dict[it['indexes']] = im_array
    ax1.cla()
    img = ax1.imshow(im_array, cmap=plt.get_cmap('hot'))
    fig.canvas.draw()
dt.save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
