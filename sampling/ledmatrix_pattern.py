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

def acquire_image_pattern(ss, pattern, Nmean=1):
    image_mean = np.zeros(cfg.patch_size)
    for i in range(Nmean+1):
        image_response = client.acquire_ledmatrix_pattern(pattern=pattern, power=255, color='R', shutter_speed=ss, iso=400, xoff=1250, yoff=950)
        image_i = np.array(image_response).reshape(cfg.patch_size)
        image_mean += image_i
    return image_mean/(Nmean)

# xx, yy = np.meshgrid(range(32), range(32))
# c = (xx-15)**2+(yy-15)**2
# matrix = 1*[c < 10**2][0]
# matrix = 1*(np.random.randn(32, 32) > 0)
def create_matrix(side='semi_left', radius=11):
    matrix = np.zeros((32, 32), dtype=int)
    for x, y in it.product(range(5, 25), range(5, 25)):
        if side is 'semi_left':
            if ( ((x-15)**2+(y-15)**2) < radius**2 and y > 15):
                matrix[x][y] = 1
        elif side is 'semi_right':
            if ( ((x-15)**2+(y-15)**2) < radius**2 and y < 15):
                matrix[x][y] = 1
        elif side is 'semi_down':
            if ( ((x-15)**2+(y-15)**2) < radius**2 and x < 15):
                matrix[x][y] = 1
        elif side is 'semi_up':
            if ( ((x-15)**2+(y-15)**2) < radius**2 and x > 15):
                matrix[x][y] = 1
        elif side is 'whole':
            if ( ((x-15)**2+(y-15)**2) < radius**2):
                matrix[x][y] = 1
    return matrix

N = 1
matrix = create_matrix(side='semi_left')
image1 = acquire_image_pattern(ss=5E4, pattern=fpm.hex_encode(matrix), Nmean=N)
matrix = create_matrix(side='semi_right')
image2 = acquire_image_pattern(ss=5E4, pattern=fpm.hex_encode(matrix), Nmean=N)
matrix = create_matrix(side='semi_down')
image3 = acquire_image_pattern(ss=5E4, pattern=fpm.hex_encode(matrix), Nmean=N)
matrix = create_matrix(side='semi_up')
image4 = acquire_image_pattern(ss=5E4, pattern=fpm.hex_encode(matrix), Nmean=N)
# decoded_matrix = fpm.hex_decode(encoded_matrix)
fig, (axes) = plt.subplots(2, 2, figsize=(25, 15))
fig.show()

image_dpc1 = (image1-image2) / (image1+image2)
image_dpc2 = (image3-image4) / (image3+image4)


axes[0][0].imshow(acquire_image_pattern(ss=5E4, pattern=fpm.hex_encode(create_matrix(side='whole')), Nmean=N))
axes[0][1].imshow(image_dpc1, cmap=plt.get_cmap('winter'))
axes[1][0].imshow(image_dpc2, cmap=plt.get_cmap('hot'))
axes[1][1].imshow(image_dpc1+image_dpc2, cmap=plt.get_cmap('gray'))

print(matrix)
plt.show()
