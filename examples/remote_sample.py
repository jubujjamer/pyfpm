#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File remote_sample.py

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
import json

from pyfpm import web
from pyfpm.fpmmath import iter_positions, recontruct
from pyfpm.data import save_metadata

# Connect to a web client running serve_microscope.py
client = web.Client('http://10.99.38.48:5000/acquire')

color = "green"
out_hf = 'P1.h5'
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
ns = 0.3   # Complement of the overlap between sampling pupils
pupil_radius = 80
phi_max = 90
image_size = (480, 640)

iterator_list = list(iter_positions(phi_max=phi_max, mode="leds"))
task = "inspect"
if task is "acquire":
    with h5py.File(out_hf, 'w') as hf:
        print("I will take %s images" % len(iterator_list))
        save_metadata(hf, image_size, iterator_list, pupil_radius, ns, phi_max)
        for index, theta, phi, power in iterator_list:
            img = client.acquire(theta, phi, power, color)
            im_array = misc.imread(StringIO(img.read()), 'RGB')
            hf.create_dataset(str(index), data=im_array)

elif task is "inspect":
    with h5py.File(out_hf, 'r') as hf:
        for index, theta, phi, power in iterator_list:
            print("i = %d, theta = %d, phi = %d" % (index, theta, phi))
            im_array = hf[str(int(index))][:]
            ax = plt.gca() or plt
            ax.imshow(im_array, cmap=plt.get_cmap('gray'))
            ax.get_figure().canvas.draw()
            plt.show(block=False)
            time.sleep(.5)

elif task is "reconstruct":
    print("I will reconstruct on %s images" % len(iterator_list))
    rec = recontruct(out_hf, debug=True, ax=None)
    plt.imshow(rec), plt.gray()
    plt.show()
    np.save("firstrec.npy", rec)

elif task is "viewrec":
    rec = np.load("firstrec.npy")
    rec *= (1.0/rec.max())
    # im = Image.fromarray(np.uint8(rec*255), 'L')
    plt.imshow(rec, cmap=plt.get_cmap('gray'))
    plt.show()
