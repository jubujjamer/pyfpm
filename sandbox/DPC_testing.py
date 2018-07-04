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
from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt


npx = 480
radius = 100
mag_array = misc.imread('sandbox/alambre.png', 'F')[:npx, :npx]
image_phase = misc.imread('sandbox/lines0_0.png', 'F')[:npx,:npx]
ph_array = np.pi*(image_phase)/np.amax(image_phase)
image_array = mag_array*np.exp(1j*ph_array)
image_fft = fftshift(fft2(image_array))

fig, (axes) = plt.subplots(3, 2, figsize=(25, 15))
fig.show()
angle1 = 0
angle2 = angle1+180
pupil_list = []
image_list = []
for angle in range(0, 361, 90):
    H = (fpm.create_led_pattern(shape='annulus', angle=angle, ledmat_shape=[npx, npx], radius = radius, int_radius = 20))
    pupil_list.append(H)
    image_list.append(np.abs(ifft2(H*image_fft)))

# IDPC1 = np.abs(ifft2(H1*image_fft))
# IDPC2 = np.abs(ifft2(H2*image_fft))
axes[1][0].imshow(pupil_list[0])
axes[1][1].imshow((image_list[0]-image_list[1])/(image_list[0]+image_list[1]))

def tik_deconvolution(pupil_list, image_list, alpha=0.1):
    fft_list = []
    H_list = []
    sum_idpc = sum(image_list)
    for idpc in image_list:
        fft_list.append(fftshift(fft2(idpc)))
    for pupil in pupil_list:
        H = pupil/sum_idpc
        H_list.append(np.conjugate(H))
    tik_den = np.sum([np.abs(H.ravel())**2 for H in H_list])+alpha
    tik_nom = 0
    for HC, FFT_IDPC in zip(H_list, fft_list):
        tik_nom += HC*FFT_IDPC
    tik = (ifft2(tik_nom/tik_den))
    return tik

phi_tik = tik_deconvolution(pupil_list, image_list, alpha=1)

axes[2][0].imshow(np.abs(phi_tik))
axes[2][1].imshow(np.real(phi_tik))

plt.show()
