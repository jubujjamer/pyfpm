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
    for i in range(Nmean+1):
        image_response = client.acquire_ledmatrix_pattern(pattern=pattern, power=255, color='R', shutter_speed=ss, iso=400, xoff=0, yoff=0)
        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/(Nmean)

def mencoded(angle):
    matrix = fpm.create_led_pattern(shape='semicircle', angle=angle)
    pattern = fpm.hex_encode(matrix)
    return pattern

def simple_dpc(image1, image2):
    return (image1-image2) / (image1+image2)



def tik_deconvolution(pupil_list, image_list, alpha=0.1):
    from numpy.fft import fft2, ifft2, fftshift, ifftshift

    fft_list = []
    H_list = []
    sum_idpc = np.sum(image_list)
    # print(sum(image_list))
    for idpc in image_list:
        fft_list.append(fftshift(fft2(idpc)))
    for pupil in pupil_list:
        H = pupil/(sum_idpc+1)
        H_list.append(np.conjugate(H))
    tik_den = np.sum([np.abs(H.ravel())**2 for H in H_list])+alpha
    print(tik_den)
    tik_nom = 0
    for HC, FFT_IDPC in zip(H_list, fft_list):
        tik_nom += HC*FFT_IDPC
    tik = (ifft2(tik_nom/tik_den))
    return tik



image_dict = np.load('out_sampling/DPC_sample_B.npy')

angles = [0, 180, 90, 270, 45, 225, 135, 315]
# angles = [0, 180]
image_list = []
pupil_list = []
for angle in angles:
    image_list.append(image_dict[()][angle])
    pupil_list.append(fpm.create_led_pattern(shape='semicircle', angle=angle, ledmat_shape=[1900, 1900], radius=600, int_radius=40))

tik = tik_deconvolution(pupil_list, image_list, alpha=0.05)
# decoded_matrix = fpm.hex_decode(encoded_matrix)
fig, (axes) = plt.subplots(2, 2, figsize=(25, 15))
fig.show()
# image_dpc2 = (image3-image4) / (image3+image4)
# image_dpc3 = (image5-image6) / (image5+image6)
# image_dpc4 = (image7-image8) / (image7+image8)
dpc_rec = simple_dpc(image_list[0], image_list[1])+simple_dpc(image_list[2], image_list[3])
dpc_rec += simple_dpc(image_list[4], image_list[5])+simple_dpc(image_list[6], image_list[7])


axes[0][0].imshow(image_list[-1], cmap=plt.get_cmap('gray'))
axes[0][1].imshow(dpc_rec, cmap=plt.get_cmap('gray'))
axes[1][0].imshow(np.abs(tik), cmap=plt.get_cmap('gray'))
axes[1][1].imshow(np.angle(tik), cmap=plt.get_cmap('gray'))
plt.show()
