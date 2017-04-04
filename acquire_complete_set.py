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

import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

from pyfpm import web
from pyfpm.fpmmath import set_iterator
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
client = web.Client(server_ip)
pc = PlatformCoordinates()
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)

# Start image acquisition
image_dict = dict()
for index, theta, phi, power in iterator:
    pc.set_coordinates(theta, phi, units='degrees')
    [theta_plat, phi_plat, shift, power] = pc.parameters_to_platform()
    img = client.acquire(theta_plat, phi_plat, shift, power)
    im_array = misc.imread(StringIO(img.read()), 'RGB')
    image_dict[(theta, phi)] = im_array
    ax = plt.gca() or plt
    ax.imshow(im_array)
    ax.get_figure().canvas.draw()
    plt.show(block=False)
save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
client.acquire(0, cfg.servo_init, 0)
