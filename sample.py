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

from pyfpm import web
from pyfpm.fpmmath import iterleds, recontruct
from pyfpm.data import save_metadata
from pyfpm.data import json_savemeta, json_loadmeta


# Connect to a web client running serve_microscope.py
client = web.Client('http://10.99.38.48:5000/acquire')
output_folder = './outputs/sample_acquired.png'
out_file = './out_sampling/sample_acquired.npy'
json_file = './out_sampling/sample_acquired.json'
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
color = "green"
ns = 0.3   # Complement of the overlap between sampling pupils
pupil_radius = 80
phi_max = 40
image_size = (480, 640)
theta_max = 180
theta_step = 20
image_dict = {}

# Opens input image as if it was sampled at pupil_pos = (0,0) with high
# resolution details
iterator = iterleds(theta_max, phi_max, theta_step, 'sample')

task = 'acquire'
if task is 'acquire':
    json_savemeta(json_file, image_size, pupil_radius, theta_max, phi_max)
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
            intensity = np.mean(im_array)
            print(intensity, theta)
            ax = plt.gca() or plt
            ax.imshow(im_array, cmap=plt.get_cmap('gray'))
            ax.get_figure().canvas.draw()
            plt.show(block=False)

if task is 'calibration':
    for n in range(5):
        iterator = iterleds(theta_max, phi_max, theta_step, 'sample')
        for index, theta, phi, power in iterator:
            if phi == 20 or phi == -20:
                img = client.acquire(theta, phi, power, color)
                im_array = misc.imread(StringIO(img.read()), 'RGB')
                intensity = np.mean(im_array)
                print(intensity, theta)
                ax = plt.gca() or plt
                ax.imshow(im_array, cmap=plt.get_cmap('gray'))
                ax.get_figure().canvas.draw()
                plt.show(block=False)
                time.sleep(5)
# iterator_list = list(iterleds(phi_max=phi_max, mode="leds"))
# task = "inspect"
# if task is "acquire":
#     with h5py.File(out_hf, 'w') as hf:
#         print("I will take %s images" % len(iterator_list))
#         save_metadata(hf, image_size, iterator_list, pupil_radius, ns, phi_max)
#         for index, theta, phi, power in iterator_list:
#             img = client.acquire(theta, phi, power, color)
#             im_array = misc.imread(StringIO(img.read()), 'RGB')
#             hf.create_dataset(str(index), data=im_array)
#
# elif task is "inspect":
#     with h5py.File(out_hf, 'r') as hf:
#         for index, theta, phi, power in iterator_list:
#             print("i = %d, theta = %d, phi = %d" % (index, theta, phi))
#             im_array = hf[str(int(index))][:]
#             ax = plt.gca() or plt
#             ax.imshow(im_array, cmap=plt.get_cmap('gray'))
#             ax.get_figure().canvas.draw()
#             plt.show(block=False)
#             time.sleep(.5)
#
# elif task is "reconstruct":
#     print("I will reconstruct on %s images" % len(iterator_list))
#     rec = recontruct(out_hf, debug=True, ax=None)
#     plt.imshow(rec), plt.gray()
#     plt.show()
#     np.save("firstrec.npy", rec)
#
# elif task is "viewrec":
#     rec = np.load("firstrec.npy")
#     rec *= (1.0/rec.max())
#     # im = Image.fromarray(np.uint8(rec*255), 'L')
#     plt.imshow(rec, cmap=plt.get_cmap('gray'))
#     plt.show()
