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

def acquire_image(client, nx, ny, shutter_speed, iso, power):
    img = client.acquire_ledmatrix(nx, ny, power, shutter_speed=shutter_speed, iso=iso)
    return misc.imread(img, 'F')

# Start analysis
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
fig.show()
iterator = ct.set_iterator(cfg)
nx, ny = 15, 15
for ss in np.linspace(100, 1000, 40):
    print('Shutter time: %i' % ss)
    start = time.time()
    image = acquire_image(client, nx, ny, power=255, shutter_speed=ss, iso=800)
    print(np.shape(image))
    # image_dict[it['indexes']] = im_array[:256, :256]
    ax1.cla()
    ax1.imshow(image, cmap=cm.hot)
    fig.canvas.draw()
    print(time.time()-start)
