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
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)

# [1944, 2592]
def acquire_image(ss, nx, ny, xoff, yoff, Nmean=1):
    image_mean = np.zeros(cfg.patch_size)
    for i in range(Nmean):
        image_response = client.acquire_ledmatrix(nx=nx, ny=ny, power=255,
                    shutter_speed=ss, iso=100, xoff=xoff, yoff=yoff)
        print(nx, ny)
        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/Nmean

out_file = dt.generate_out_file(fname = 'outest.npy')
image_dict = dict()

# Start analysis
fig, axes = plt.subplots(3, 3, figsize=(25, 25))

fig.show()
for xoff, yoff in [[1100, 750]]:
    for ax in axes.ravel():
        ax.cla()
    fig.canvas.draw()

    axes[0][0].imshow(acquire_image(ss=5000E4, nx=19, ny=19, Nmean=6, xoff=xoff, yoff=yoff), cmap=cm.gray)
    # # axes[0][1].imshow(acquire_image(ss=5E4, nx=15, ny=16, xoff=xoff, yoff=yoff), cmap=cm.gray)
    # axes[0][2].imshow(acquire_image(ss=50E4, nx=13, ny=17, Nmean=4, xoff=xoff, yoff=yoff), cmap=cm.gray)
    #
    # # axes[1][0].imshow(acquire_image(ss=5E4, nx=16, ny=15, xoff=xoff, yoff=yoff), cmap=cm.gray)
    # axes[1][1].imshow(acquire_image(ss=5E4, nx=15, ny=15, Nmean=4, xoff=xoff, yoff=yoff), cmap=cm.gray)
    # # axes[1][2].imshow(acquire_image(ss=5E4, nx=14, ny=15, xoff=xoff, yoff=yoff), cmap=cm.gray)
    #
    #
    # axes[2][0].imshow(acquire_image(ss=500E4, nx=17, ny=13, Nmean=4, xoff=xoff, yoff=yoff), cmap=cm.gray)
    # # axes[2][1].imshow(acquire_image(ss=5E4, nx=15, ny=14, Nmean=4, xoff=xoff, yoff=yoff), cmap=cm.gray)
    # axes[2][2].imshow(acquire_image(ss=500E4, nx=13, ny=13, Nmean=4, xoff=xoff, yoff=yoff), cmap=cm.gray)
    fig.canvas.draw()
    time.sleep(3)

plt.show()
