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
import pyfpm.solvertools as st


# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)

def get_color_dict(color='G'):
    image_dict = np.load('out_sampling/DPC_sample_%s.npy' % color)[()]
    return image_dict

def get_background_dict(color='G'):
    background_dict = np.load('out_sampling/DPC_background_%s.npy' % color)[()]
    return background_dict

def substract_background(image, background):
    image_corrected = image-background
    return image_corrected

def get_extrema(image):
    extrema = 'Min %.1f, Max: %.1f' % (image.min(), image.max())
    return extrema

def rescale_image(image, inf_lim=0, sup_lim=1):
    image -= image.min()
    image /= image.max()
    return image

# angles = [0, 180, 90, 270, 45, 225, 135, 315]
rgbArray = np.zeros((1900, 1900, 3))
image_dict = get_color_dict(color='G')
background_dict = get_background_dict(color='G')
angles = image_dict.keys()

image0, background0 = [image_dict[-1], background_dict[-1]]
corrected0 = substract_background(image0, background0)

test_angle = 0
fig1, axes = plt.subplots(1, 3, figsize=(25, 15))
image, background = [image_dict[test_angle], background_dict[test_angle]]
axes[0].imshow(image), axes[0].set_title(label=get_extrema(image))
axes[1].imshow(background), axes[1].set_title(label=get_extrema(background))
corrected1 = substract_background(image, background)
# corrected1 = image
axes[2].imshow(corrected1), axes[2].set_title(label=get_extrema(corrected1))

test_angle = 180
fig2, axes = plt.subplots(1, 3, figsize=(25, 15))
image, background = [image_dict[test_angle], background_dict[test_angle]]
axes[0].imshow(image), axes[0].set_title(label=get_extrema(image))
axes[1].imshow(background), axes[1].set_title(label=get_extrema(background))
corrected2 = substract_background(image, background)
# corrected2 = image
axes[2].imshow(corrected2), axes[2].set_title(label=get_extrema(corrected2))

fig3, axes = plt.subplots(1, 2)
dpc_image = st.dpc_difference(corrected1, corrected2)

image1 = corrected1 - np.mean(corrected1)
image2 = corrected2 - np.mean(corrected2)
den = image1+image2
den -= (den.min()-1)
dpc_diff = image1-image2
dpc_image = dpc_diff/den

dpc_image = rescale_image(dpc_image)
axes[0].imshow(dpc_image), axes[0].set_title(get_extrema(dpc_image))
axes[1].hist(dpc_image.ravel(), bins=80)

fig4, ax = plt.subplots(1, 1)
ax.imshow(dpc_image, vmin=0.86, vmax=0.98)

fig5, ax = plt.subplots(1, 1)
image, background = [image_dict[test_angle], background_dict[test_angle]]
ax.imshow(image)


plt.show()

#image_blanks = image_blanks = np.load(get_color_background('R'))
#
# # Get a list of background corrected images
# for angle in angles:
#     image = image_dict[()][angle]
#     blank = image_blanks[()][angle]
#     image_corr = image-.3*blank
#     image_corr -= np.min(image_corr)
#     # image_corr = image
#     image_list.append(image_corr)
#     pupil_list.append(fpm.create_source_pattern(shape='semicircle', angle=angle, ledmat_shape=[1900, 1900], radius=600, int_radius=60))
#
# # Apply the DPC reconstructon to that list
# dpc_total = np.zeros_like(image)
# for j in range(2):
#     dpc_total += st.dpc_difference(image_list[i], image_list[i+1])
# return dpc_total
#
#
#
#
#
# fig, axes = plt.subplots(1, 3, figsize=(25, 15))
# plt.show()
