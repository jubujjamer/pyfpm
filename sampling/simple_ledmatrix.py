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

def acquire_image(client, nx, ny, power):
    img = client.acquire_ledmatrix(nx, ny, power, shutter_speed=10000, iso=400)
    # return misc.imread(StringIO(img.read()), 'RGB')

# Start analysis
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
fig.show()

iterator = ct.set_iterator(cfg)
for it in iterator:
    print(it['nx'])
    im_array = acquire_image(client, it['nx'], it['ny'], 255)
    ax1.cla()
    # ax1.imshow(im_array, cmap=cm.hot)
    fig.canvas.draw()
plt.show()
