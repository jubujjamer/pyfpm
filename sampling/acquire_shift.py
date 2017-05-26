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
comp_images, comp_cfg = dt.open_sampled('2017-05-22_172249.npy')
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
                         shutter_speed=50000, iso=400)
    return misc.imread(StringIO(img.read()), 'RGB')

# Start analysis
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
plt.grid(False)
fig.show()

image_dict = dict()
iterator = fpm.set_iterator(cfg)

for index, theta, shift in iterator:
    print(theta, shift)
    ax1.cla()
    # im_array = acquire_image(pc, client, 0, 0, 0, 0)
    im_array = acquire_image(pc, client, theta, 0, shift, 255)
    image_dict[(theta, shift)] = im_array
    ax1.imshow(im_array, cmap=cm.hot)
    # im_cmp = comp_images[(theta, shift)]
    # ax2.imshow(im_cmp, cmap=cm.hot)
    fig.canvas.draw()
# im_array = acquire_image(pc, client, 0, 0, 0, 255)
# misc.imsave('./strips.png', im_array)
dt.save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
# plt.show()
