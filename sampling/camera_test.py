#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
from io import StringIO
import time

import numpy as np
import itertools as it
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
import imageio
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)

def take_image(ss, nx, ny, Nmean=1):
    image_mean = np.zeros(cfg.patch_size)
    for i in range(Nmean):
        image_response = client.acquire_ledmatrix(nx=nx, ny=ny, power=255,
                    shutter_speed=ss, iso=100, xoff=1200, yoff=400)
        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/Nmean


ss_normal = 5000
# Start analysis
fig, axes = plt.subplots(1, 3, figsize=(25, 15))
fig.show()
for ss in range(100):
    start = time.time()
    image_array = take_image(ss=5E4, nx=15, ny=15, Nmean=4)
    print('Took', time.time()-start)
    for ax in axes:
        ax.cla()
    axes[0].imshow(image_array, cmap=cm.gray)
    # axes[0].set_title('xoff: %i, yoff= %i' % (i*100, 500))
    axes[1].hist(image_array.ravel(), bins = 30)
    axes[1].set_xlim([0, 128])
    fig.canvas.draw()
plt.show()
