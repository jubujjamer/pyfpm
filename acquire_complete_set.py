#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
from StringIO import StringIO
import os
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy import misc
import numpy as np
import time

from pyfpm import web
from pyfpm.fpmmath import set_iterator, translate
from pyfpm.data import save_yaml_metadata
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)

out_file = os.path.join(cfg.output_sample,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
in_file = os.path.join(cfg.output_sample,
                        './2017-03-23_19:04:01.npy')

# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)
pc = PlatformCoordinates()
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)

# Start image acquisition
image_dict = dict()
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
# im1 = ax.imshow(np.ones((480,640)), cmap=plt.get_cmap('hot'))
fig.show()
for index, theta, phi, power in iterator:
    pc.set_coordinates(theta, phi, units='degrees')
    [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
    iso = 400
    ss = translate(phi, 0, 60, 50000, 900000)
    print(theta, phi, ss)
    client.just_move(theta_plat, phi_plat, shift_plat, power)
    # time.sleep(1)
    img = client.acquire(theta_plat, phi_plat, shift_plat, power, shutter_speed=ss, iso=iso)
    im_array = misc.imread(StringIO(img.read()), 'RGB')
    image_dict[(theta, phi)] = im_array
    ax.cla()
    ax.imshow(im_array, cmap=plt.get_cmap('hot'))
    fig.canvas.draw()
client.just_move(0, 0, 0, 0)

save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
