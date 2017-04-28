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
import itertools as it

from pyfpm import web
from pyfpm.fpmmath import set_iterator, translate, adjust_shutter_speed
from pyfpm.data import save_yaml_metadata
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
cfg = dt.load_config()

out_file = dt.generate_out_file(cfg.output_sample)
in_file = os.path.join(cfg.output_sample, './2017-03-23_19:04:01.npy')

# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)
pc = PlatformCoordinates()
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)

# Start image acquisition
image_dict = dict()
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
fig.show()

# Saving other parameters
phi_min, phi_max, phi_step = cfg.phi
theta_min, theta_max, theta_step = cfg.theta
phi_range = range(phi_min, theta_min, phi_step)
theta_range = range(theta_min, theta_max, theta_step)
ss_dict = dict.fromkeys(list(it.product(theta_range, phi_range)))
power_dict = dict.fromkeys(list(it.product(theta_range, phi_range)))
ss_list = [110000, 500000, 800000]


for index, theta, phi in iterator:
    print(theta, phi)
    pc.set_coordinates(theta, phi, units='degrees')
    [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
    # ss = adjust_shutter_speed(theta, phi)
    power = 255
    ss = ss_list[0]
    client.just_move(theta_plat, phi_plat, shift_plat, power)
    # time.sleep(1)
    img = client.acquire(theta_plat, phi_plat, shift_plat, power,
                        shutter_speed=ss, iso=400)

    im_array = misc.imread(StringIO(img.read()), 'RGB')
    print(np.mean(im_array), np.max(im_array))
    if np.max(im_array)<80:
        ss = ss_list[2]
        img = client.acquire(theta_plat, phi_plat, shift_plat, power,
                            shutter_speed=ss, iso=400)
    ss_dict[theta, phi] = ss
    power_dict[theta, phi] = power
    image_dict[(theta, phi)] = im_array
    ax.cla()
    ax.imshow(im_array, cmap=plt.get_cmap('hot'), vmin = 0, vmax=200)
    fig.canvas.draw()
client.just_move(0, 0, 0, 0)

save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
np.save(out_file+'ss_dict', ss_dict)
np.save(out_file+'power_dict', power_dict)
