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
# [1100, 750]
# [1200, 400]
# xoff=1500, yoff=1300
def acquire_image(ss, nx, ny, Nmean=1):
    image_mean = np.zeros(cfg.patch_size)
    for i in range(Nmean+1):
        image_response = client.acquire_ledmatrix(nx=nx, ny=ny, power=255,
                    shutter_speed=ss, iso=400, xoff=1500, yoff=1250, color='B')

        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/(Nmean)

out_file = dt.generate_out_file(out_folder=cfg.output_sample)
image_dict = dict()

# Start analysis
fig, (axes) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()
iterator = ct.set_iterator(cfg)

total_time = 0
for it in iterator:
    nx, ny = it['nx'], it['ny']
    iso, ss, power = it['acqpars']
    total_time += ss/1E6
    print(nx, ny, ss, total_time)
iterator = ct.set_iterator(cfg)

fig.canvas.draw()

for it in iterator:
    nx, ny = it['nx'], it['ny']
    iso, ss, power = it['acqpars']
    print(nx, ny, ss)
    im_array = acquire_image(ss=ss, nx=nx, ny=ny, Nmean=1)
    image_dict[it['indexes']] = im_array
    for ax in axes:
        ax.cla()
    axes[0].imshow(im_array, cmap=cm.gray)
    axes[0].set_title('nx=%i, ny=%i, ss=%i' % (nx, ny, ss))
    axes[1].hist(im_array.ravel(), bins = 50)
    axes[1].set_xlim([0, 255])
    fig.canvas.draw()

dt.save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
# plt.show()
