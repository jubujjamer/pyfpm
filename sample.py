#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File sample.py

Last update: 28/10/2016
To be used as a remote client for a microscope.
Before runing this file make sure there is a server running serve_microscope.py
hosted in some url.

Usage:

"""
from StringIO import StringIO
import time

import matplotlib.pyplot as plt
import h5py
from scipy import misc
import numpy as np
import time

from pyfpm import web
from pyfpm.fpmmath import iterleds, recontruct, to_leds_coords, correct_angles
from pyfpm.data import save_metadata
from pyfpm.data import json_savemeta, json_loadmeta


# Connect to a web client running serve_microscope.py
client = web.Client('http://10.99.38.48:5000/acquire')
out_file = './out_sampling/2016_11_17_1.npy'
json_file = './out_sampling/2016_11_17_1.json'
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
color = 'green'
ns = 0.3   # Complement of the overlap between sampling pupils
pupil_radius = 120
phi_max = 40
image_size = (480, 640)
theta_max = 360
theta_step = 10
image_dict = {}

# Opens input image as if it was sampled at pupil_pos = (0,0) with high
# resolution details
iterator = iterleds(theta_max, phi_max, theta_step, 'sample')

task = 'reconstruct'
if task is 'acquire':
    json_savemeta(json_file, image_size, pupil_radius, theta_max,
                  theta_step, phi_max)
    for index, theta, phi, power in iterator:
        print(index, theta, phi, power)
        img = client.acquire(theta, phi, power, color)
        im_array = misc.imread(StringIO(img.read()), 'RGB')
        image_dict[(theta, phi)] = im_array
    print(out_file)
    np.save(out_file, image_dict)
    client.acquire(0, 0, 0, color)

elif task is 'reconstruct':
    start_time = time.time()
    data = json_loadmeta(json_file)
    rec = recontruct(out_file, iterator, debug=True, ax=None, data=data)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.imshow(rec), plt.gray()
    plt.show()

if task is 'complete_scan':
    print(task)
    json_savemeta(json_file, image_size, pupil_radius, theta_max, phi_max)
    client.complete_scan(color)

if task is 'test_and_measure':
    image_dict = np.load(out_file)
    for index, theta, phi, power in iterator:
        if phi == 20:
            im_array = image_dict[()][(theta, phi)]
            circle = np.zeros_like(im_array)
            # intensity = np.mean(im_array)
            # print(intensity, theta)
            ax = plt.gca() or plt
            ax.imshow(im_array)
            ax.get_figure().canvas.draw()
            plt.show(block=False)

if task is 'calibration':
    json_savemeta(json_file, image_size, pupil_radius, theta_max,
                  theta_step, phi_max)
    print('calibration')
    iterator = iterleds(theta_max, phi_max, theta_step, 'sampling')
    for index, theta, phi, power in iterator:
        print(theta, phi)
        theta_leds, phi_leds = to_leds_coords(theta, phi)
        img = client.acquire(theta_leds, phi_leds, power, color)
        im_array = misc.imread(StringIO(img.read()), 'RGB')
        # intensity = np.mean(im_array)
        ax = plt.gca() or plt
        ax.imshow(im_array, cmap=plt.get_cmap('gray'))
        ax.get_figure().canvas.draw()
        plt.show(block=False)
        image_dict[(theta, phi)] = im_array
    print(out_file)
    np.save(out_file, image_dict)
    client.acquire(0, 0, 0, color)

if task is 'testing':
    iterator = iterleds(theta_max, phi_max, theta_step, 'sampling')
    for index, theta, phi, power in iterator:
        # slice_ratio = np.cos(np.radians(phi))
        # phi_tilted = get_tilted_phi(theta=theta, alpha=-3, slice_ratio=slice_ratio)
        (theta_corr, phi_corr) = correct_angles(theta, phi)
        print('theta: %d, phi: %d' % (theta, phi))
        print('theta_corr: %f, phi_corr: %f' % (theta_corr, phi_corr))
