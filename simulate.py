#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time

import matplotlib.pyplot as plt
import numpy as np

import pyfpm.local as local
from pyfpm.fpmmath import iterleds, recontruct
from pyfpm.data import json_savemeta, json_loadmeta

from scipy import misc
from StringIO import StringIO

# Simulation parameters
input_image = './imgs/pinholes_calibration.png'
output_folder = './outputs/pinholes_simulated.png'
out_file = './output/pinholes_calibration.npy'
json_file = './output/pinholes_calibration.json'
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
ns = 0.3  # Complement of the overlap between sampling pupils
phi_max = 40
theta_max = 180
theta_step = 10
pupil_radius = 20
image_dict = {}

# Opens input image as if it was sampled at pupil_pos = (0,0) with high
# resolution details
with open(input_image, "r") as imageFile:
    image = imageFile.read()
    image_size = np.shape(misc.imread(StringIO(image), 'RGB'))
client = local.SimClient(image, image_size, pupil_radius, ns)
iterator = iterleds(theta_max, phi_max, theta_step, 'simulation')

task = 'test_and_measure'
if task is 'acquire':
    json_savemeta(json_file, image_size, pupil_radius, theta_max, phi_max)
    for index, theta, phi, power in iterator:
        print(index, theta, phi, power)
        image_dict[(theta, phi)] = client.acquire(theta, phi, power)
    np.save(out_file, image_dict)

elif task is 'reconstruct':
    start_time = time.time()
    data = json_loadmeta(json_file)
    rec = recontruct(out_file, iterator, debug=True, ax=None, data=data)
    print('--- %s seconds ---' % (time.time() - start_time))
    plt.imshow(rec), plt.gray()
    plt.show()

if task is 'test_and_measure':
    image_dict = np.load(out_file)
    for index, theta, phi, power in iterator:
        if phi == 20:
            im_array = image_dict[()][(theta, phi)]
            intensity = np.mean(im_array)
            print(intensity, theta)
            ax = plt.gca() or plt
            ax.imshow(im_array, cmap=plt.get_cmap('gray'))
            ax.get_figure().canvas.draw()
            plt.show(block=False)
