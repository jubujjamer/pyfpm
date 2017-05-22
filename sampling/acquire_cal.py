#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
from StringIO import StringIO
import time

import numpy as np
import itertools as it
import pygame
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
from skimage import measure
import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
from pyfpm.fpmmath import set_iterator
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)
pc = PlatformCoordinates(cfg=cfg)
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)

def acquire_image(pc, client, theta, phi, shift, power):
    pc.set_coordinates(theta=theta, phi=phi, shift=shift,
                       units='deg_shift')
    [theta_plat, phi_plat, shift_plat, power_plat] = pc.parameters_to_platform()
    # ss = fpm.adjust_shutter_speed(tpsp[0], tpsp[1])
    img = client.acquire(theta_plat, phi_plat, shift_plat, power,
                         shutter_speed=10000, iso=400)
    return misc.imread(StringIO(img.read()), 'RGB')

# Start analysis
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
plt.grid(False)
fig.show()

with h5py.File('./misc/centroids_1.h5', 'r') as hf:
    centroids_reference = hf['centroids/originals'][:]


centroids = list()
parameters = list()
theta_range = range(cfg.theta[0], cfg.theta[1], cfg.theta[2])
shift_range = range(cfg.shift[0], cfg.shift[1], cfg.shift[2])
iterator = fpm.set_iterator(cfg)

colors = iter(cm.gist_rainbow(np.linspace(0, 1, 10)))
for index, theta, shift in iterator:
    im_array = acquire_image(pc, client, 0, 0, 0, 0)
    # im_array = acquire_image(pc, client, theta, 0, shift, 150)
    im_array[im_array < 20] = 0
    contours = measure.find_contours(im_array, 0.9)
    n_cont = np.argmax([len(contour) for n, contour in enumerate(contours)])
    contour = contours[n_cont]
    ax1.cla()
    ax1.plot(contour[:, 1], contour[:, 0], 'r', linewidth=1)
    cx, cy = np.mean(contour[:, 1]), np.mean(contour[:, 0])
    centroids.append([cx, cy])
    parameters.append([theta, shift, cx, cy])
    for c in centroids:
         ax1.plot(c[0], c[1], 'c*', linewidth=2, markersize=5)
    for t, s, cx, cy in centroids_reference:
         ax1.plot(cx, cy, 'rx', linewidth=.1, markersize=2)
    ax1.imshow(im_array, cmap=cm.hot)
    fig.canvas.draw()


with h5py.File('./misc/centroids_2.h5', 'w') as hf:
    dset = hf.create_dataset('centroids/originals',  data=parameters, dtype='f')
    hf['centroids'].attrs['documentation'] = 'Imaged light source centroid measurement'
    dset.attrs['data_order'] = '[theta, shift, centroid_x, centroid_y]'
    dset.attrs['whatami'] = 'Raw measurements.'
    dset.attrs['centroids/originals'] = 'Raw theta and shift measurements'
plt.show()
