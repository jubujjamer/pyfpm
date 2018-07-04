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

# xoff=1250, yoff=950
def acquire_image_pattern(ss, pattern, Nmean=1):
    image_mean = np.zeros(cfg.patch_size)
    for i in range(Nmean):
        image_response = client.acquire_ledmatrix_pattern(pattern=pattern, power=255, color='R', shutter_speed=ss, iso=400, xoff=0, yoff=0)
        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/(Nmean)

def mencoded(angle):
    matrix = fpm.create_led_pattern(shape='semicircle', angle=angle, int_radius=2, radius=10,)
    pattern = fpm.hex_encode(matrix)
    return pattern

circ_matrix = fpm.create_led_pattern(shape='circle')
N = 1
angles = [0, 180, 90, 270, 45, 225, 135, 315]
# angles = [0]
pupil_list = []
image_dict = dict()

fig, (ax) = plt.subplots(1, 1, figsize=(25, 15))
fig.show()
fig.canvas.draw()

for angle in angles:
    image_dict[angle] = acquire_image_pattern(ss=1E4, pattern=mencoded(angle=angle), Nmean=N)
    ax.cla()
    ax.imshow(image_dict[angle], cmap=cm.gray)
    fig.canvas.draw()


image_dict[-1] = acquire_image_pattern(ss=5E4, pattern=fpm.hex_encode(circ_matrix), Nmean=N)
# decoded_matrix = fpm.hex_decode(encoded_matrix)

np.save('out_sampling/DPC_background_R.npy', image_dict)
