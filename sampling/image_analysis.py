#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
# from StringIO import StringIO
import time

import numpy as np
import itertools as it
import pygame
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
from skimage import measure
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt
# from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
# samples, comp_cfg = dt.open_sampled('2017-05-26_145449.npy')
# background, comp_cfg = dt.open_sampled('2017-05-26_145449_blank.npy')
samples, comp_cfg = dt.open_sampled('outest.npy', mode='simulation')
# 150543
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)
iterator = ct.set_iterator(cfg)

# Start analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
plt.grid(False)
fig.show()

image_dict = dict()
iterator = ct.set_iterator(cfg)
corr_ims = list()

for it in iterator:
    acqpars = it['acqpars']
    im_array = samples[it['indexes']]
    time.sleep(.5)
    im_array *= (255.0/im_array.max())
    ax1.cla(), ax2.cla()
    # # im_array = acquire_image(pc, client, 0, 0, 0, 0)
    ax1.imshow(im_array, cmap=cm.hot)
    fig.canvas.draw()
fig.canvas.draw()
plt.show()
