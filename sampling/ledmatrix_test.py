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
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)

out_file = dt.generate_out_file(out_folder=cfg.output_sample, fname=None)
image_dict = dict()

def acquire_image(ss, nx, ny, Nmean=1):
    image_mean = np.zeros(cfg.patch_size)
    for i in range(Nmean):
        print(i)
        image_response = client.acquire_ledmatrix(nx=nx, ny=ny, power=255,
                    shutter_speed=ss, iso=400, xoff=1296, yoff=950)
        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/(Nmean)

# Start analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()
iterator = ct.set_iterator(cfg)
nx, ny = 15, 15
fig.canvas.draw()
for ss in np.linspace(1000000, 5000000, 8):
    start = time.time()
    image = acquire_image(ss=ss, nx=15, ny=20, Nmean=1)
    print('Shutter speed: %i' % ss)
    print('Elapsed time: %.1f' % (time.time()-start))
    print('Estimated time: %.3f' % (ss*1E-6))
    ax1.cla()
    ax2.cla()
    ax1.imshow(image, cmap=cm.hot)
    ax2.hist(image.ravel(), bins=50)
    fig.canvas.draw()
