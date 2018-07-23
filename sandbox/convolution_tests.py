#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time

import numpy as np
import itertools as it
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy
from scipy import misc, signal, optimize


from numpy.fft import fft2, ifft2, fftshift, ifftshift
import pyfpm.fpmmath as fpm
import pyfpm.solvertools as st


N = 150
n = 20
r = 30
AP = fpm.aberrated_pupil(image_size=[N, N], pupil_radius=r, aberrations=[1E-6, ],
                    pixel_size=2E-6, wavelength=522E-9)
# P = fpm.create_source_pattern(shape='semicircle', ledmat_shape=[n, n], radius=r)
H = np.real(fftshift(fft2(fftshift(AP))))
fig1, (axes) = plt.subplots(1, 2, figsize=(25, 15))
axes[0].imshow(np.abs(H))
axes[1].imshow(np.angle(H))# Loading image
mag_array = scipy.misc.imread('sandbox/alambre.png', 'F')[:N, :N]
image_phase = scipy.misc.imread('sandbox/lines0_0.png', 'F')[:N,:N]
ph_array = np.pi*(image_phase)/np.amax(image_phase)
image_array = mag_array*np.exp(1j*ph_array)
