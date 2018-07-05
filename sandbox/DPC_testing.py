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
from scipy import misc, signal
from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt


npx = 32
radius = 10
mag_array = misc.imread('sandbox/alambre.png', 'F')[:npx, :npx]
image_phase = misc.imread('sandbox/lines0_0.png', 'F')[:npx,:npx]
ph_array = np.pi*(image_phase)/np.amax(image_phase)
image_array = mag_array*np.exp(1j*ph_array)
image_fft = fftshift(fft2(image_array))

angles = [0, 180, 90, 270, 45, 225, 135, 315]
pupil_list = []
image_list = []
for angle in angles:
    H = (fpm.create_source_pattern(shape='semicircle', angle=angle, ledmat_shape=[npx, npx], radius = radius, int_radius = 20))
    pupil_list.append(H)
    image_list.append(np.abs(ifft2(H*image_fft)))

fig, (axes) = plt.subplots(2, 2, figsize=(25, 15))
fig.show()
# IDPC2 = np.abs(ifft2(H2*image_fft))
axes[0][0].imshow(pupil_list[0]+pupil_list[2])
axes[0][1].imshow(pupil_list[1]+pupil_list[3])

signal.correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0)

plt.show()
