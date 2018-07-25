#!/usr/bin/python
# -*- coding: utf-8 -*-
""" dpc_rgb.py

Last update: 24/07/2018
Use reconstruct DPC images.

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
import pyfpm.solvertools as st


# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)

# xoff=1250, yoff=950
def mencoded(angle):
    matrix = fpm.create_led_pattern(shape='semicircle', angle=angle)
    pattern = fpm.hex_encode(matrix)
    return pattern

def get_color_file(color='G'):
    return 'out_sampling/DPC_sample_%s.npy' % color

def get_color_background(color='G'):
    return 'out_sampling/DPC_background_%s.npy' % color

def get_dpc_reconstruction(color, angles):
    image_dict = np.load(get_color_file(color))
    image_list = []
    pupil_list = []
    try:
        image_blanks = np.load(get_color_background(color))
    except:
        image_blanks = np.zeros((1900, 1900))
        image_blanks = image_blanks = np.load(get_color_background('R'))

    # Get a list of background corrected images
    for angle in angles:
        image = image_dict[()][angle]
        blank = image_blanks[()][angle]
        image_corr = image-.3*blank
        image_corr -= np.min(image_corr)
        # image_corr = image
        image_list.append(image_corr)
        pupil_list.append(fpm.create_source_pattern(shape='semicircle', angle=angle, ledmat_shape=[1900, 1900], radius=600, int_radius=60))

    # Apply the DPC reconstructon to that list
    dpc_total = np.zeros_like(image)
    for j in range(2):
        dpc_total += st.dpc_difference(image_list[i], image_list[i+1])
    return dpc_total

image_channels = {'R': [], 'G': [], 'B': []}
rgbArray = np.zeros((1900, 1900, 3))
colorArray = np.zeros((1900, 1900, 3))
angles = [0, 180, 90, 270, 45, 225, 135, 315]

fig1, (axes) = plt.subplots(2, 3, figsize=(25, 15))
fig1.show()
for i, color in enumerate(image_channels.keys()):
    colorArray = np.zeros((1900, 1900, 3))
    # rgbArray[..., 0] = np.zeros((1900, 1900))
    ax = axes[0][i]
    hax = axes[1][i]
    image = get_dpc_reconstruction(color, angles)
    # image = rgbArray[..., i]
    image -= image.min()
    image /= image.max()
    # image -= np.mean(image)
    # image += 0.5
    rgbArray[..., i] = image
    colorArray[..., i] = image
    ax.imshow(colorArray)
    hax.hist(image.ravel(), bins=30)


# next(axiter).imshow(image_list[-1], cmap=plt.get_cmap('gray'))
fig2, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(rgbArray)
ax[1].imshow(np.load(get_color_file('B'))[()][0])
plt.show()
