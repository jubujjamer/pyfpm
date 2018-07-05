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
from scipy import misc, signal, optimize
from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt


npx = 200
radius = 60
mag_array = misc.imread('sandbox/alambre.png', 'F')[:npx, :npx]
image_phase = misc.imread('sandbox/lines0_0.png', 'F')[:npx,:npx]
ph_array = np.pi*(image_phase)/np.amax(image_phase)
image_array = mag_array*np.exp(1j*ph_array)
image_fft = fftshift(fft2(image_array))

def create_hph(angle1, angle2, npx, radius):
    radius=radius//2
    n = npx//2+1
    P = fpm.create_source_pattern(shape='circle', ledmat_shape=[n, n], radius=radius)
    S1 = fpm.create_source_pattern(shape='semicircle', angle=angle1, ledmat_shape=[n, n], radius=radius)
    S2 = fpm.create_source_pattern(shape='semicircle', angle=angle2, ledmat_shape=[n, n], radius=radius)
    Hph = 1j*(signal.correlate2d(P, S1*P)-signal.correlate2d(P, S2*P))
    Hph /= np.pi*np.max(np.imag(Hph))
    return Hph[0:npx,0:npx]

def simple_dpc(image1, image2):
    den = image1+image2
    idpc = (image1-image2) / den
    idpc /= np.pi*np.max(idpc)
    return idpc

def tik_deconvolution(pupil_list, image_list, alpha=0.1):
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    fft_list = []
    H_list = []
    # sum_idpc = np.sum([np.sum(i.ravel()) for i in image_list])
    sum_idpc = np.sum([i for i in image_list])
    for idpc in image_list:
        fft_list.append(fftshift(fft2(idpc)))
    for pupil in pupil_list:
        H = pupil/(sum_idpc)
        H_list.append(np.conjugate(H))
    tik_den = np.sum([np.abs(H.ravel())**2 for H in H_list])+alpha
    print(tik_den)
    tik_nom = 0
    for HC, FFT_IDPC in zip(H_list, fft_list):
        tik_nom += HC*FFT_IDPC
    tik = (ifft2(tik_nom/tik_den))
    return tik


P = fpm.create_source_pattern(shape='circle', ledmat_shape=[npx, npx], radius=radius)
S1 = fpm.create_source_pattern(shape='semicircle', angle=0, ledmat_shape=[npx, npx], radius=radius)
S2 = fpm.create_source_pattern(shape='semicircle', angle=180, ledmat_shape=[npx, npx], radius=radius)
S3 = fpm.create_source_pattern(shape='semicircle', angle=90, ledmat_shape=[npx, npx], radius=radius)
S4 = fpm.create_source_pattern(shape='semicircle', angle=270, ledmat_shape=[npx, npx], radius=radius)

Hph1 = create_hph(angle1=0, angle2=180, npx=200, radius=radius)
Hph2 = create_hph(angle1=90, angle2=270, npx=200, radius=radius)

I1 = np.abs(ifft2(S1*image_fft))
I2 = np.abs(ifft2(S2*image_fft))
I3 = np.abs(ifft2(S3*image_fft))
I4 = np.abs(ifft2(S4*image_fft))

Idpc1 = simple_dpc(I1, I2)
Idpc2 = simple_dpc(I3, I4)

# phi = tik_deconvolution([Hph1], [Idpc1], alpha=0.1)

print(Idpc1.shape, np.zeros(shape=(npx,1)).shape)
A1 = np.concatenate((Hph1, 1*np.matlib.identity(npx)))
b1 = np.concatenate((Idpc1, np.zeros(shape=(1, npx))))
phi = optimize.nnls(A1, b1)


fig, (axes) = plt.subplots(3, 2, figsize=(25, 15))
fig.show()

Hph = 1j*(signal.correlate2d(P, S1*P)-signal.correlate2d(P, S2*P))
axes[0][0].imshow(np.imag(1/(Hph1+.1)), cmap=plt.get_cmap('hot'))
axes[0][1].imshow(image_phase, cmap=plt.get_cmap('hot'))
axes[1][0].imshow(Idpc1)
axes[1][1].imshow(Idpc2)
axes[2][0].imshow(np.abs(phi))
axes[2][1].imshow(np.imag(phi))
plt.show()
