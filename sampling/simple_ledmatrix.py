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

def acquire_image(client, nx, ny, shutter_speed, iso, power):
    img = client.acquire_ledmatrix(nx, ny, power, shutter_speed=shutter_speed, iso=iso)
    return misc.imread(img, 'F')

out_file = dt.generate_out_file(fname = 'outest.npy')
image_dict = dict()

# Start analysis
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
fig.show()
iterator = ct.set_iterator(cfg)
for it in iterator:
    nx, ny = it['nx'], it['ny']
    shutter_speed = 1E6
    iso = 1000
    if nx in [14, 15, 16, 17] or ny in [14, 15, 16, 17]:
        shutter_speed = 100000
        iso = 200
    print(nx, ny, shutter_speed)
    im_array = acquire_image(client, nx, ny,
                             power=255, shutter_speed=shutter_speed, iso=iso)
    image_dict[it['indexes']] = im_array[:256, :256]
    ax1.cla()
    ax1.imshow(im_array[:256, :256], cmap=cm.hot)
    fig.canvas.draw()
dt.save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
# plt.show()
