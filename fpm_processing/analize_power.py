#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import os
import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

from pyfpm.fpmmath import *
import pyfpm.data as dt
from pyfpm.data import save_yaml_metadata
import pyfpm.local as local
from pyfpm.coordinates import PlatformCoordinates
import pyfpm.fpmmath as fpm

# Simulation parameters
CONFIG_FILE = './config.yaml'
cfg = dt.load_config(CONFIG_FILE)
simclient = local.SimClient(cfg=cfg)

in_file = os.path.join(cfg.output_sample, './2017-04-05_11:36:01.npy')

iterator = set_iterator(cfg)
pc = PlatformCoordinates()
pc.generate_model(cfg.plat_model)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()
image_dict = np.load(in_file)[()]

## Saving the data
phi_min, phi_max, phi_step = cfg.phi
theta_min, theta_max, theta_step = cfg.theta
phi_range = range(5, 9, phi_step)
theta_range = range(theta_min, theta_max, theta_step)
data_mean_dict = dict.fromkeys(list(it.product(theta_range, phi_range)))
sim_mean_dict = dict.fromkeys(list(it.product(theta_range, phi_range)))

for index, theta, phi in iterator:
    pc.set_coordinates(theta, phi, units='degrees')
    [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
    shutter_speed = adjust_shutter_speed(theta, phi)
    im_array = image_dict[(theta, phi)]/(power*shutter_speed)
    im_array = fpm.crop_image(im_array, cfg.video_size, 170, 245)

    sim_im_array = simclient.acquire(theta, phi, power)
    if theta == 0 and phi == 0:
        data_norm_factor = np.mean(im_array)
        sim_norm_factor = np.mean(sim_im_array)

    data_mean_dict[theta, phi] = np.mean(im_array)/data_norm_factor
    sim_mean_dict[(theta, phi)] = np.mean(sim_im_array)/sim_norm_factor


for phi in phi_range:
    data_mean_array = [data_mean_dict[t, phi] for t in theta_range]
    data_mean_array[data_mean_array.index(None)] = 0
    ax1.plot(np.log10(data_mean_array), '.-')
    sim_mean_array = [sim_mean_dict[t, phi] for t in theta_range]
    sim_mean_array[sim_mean_array.index(None)] = 0
    ax2.plot(np.log10(sim_mean_array), '.-')
    # plt.xlim([0,450])
    # plt.ylim([0,2.5])
ax1.annotate('Mean value: %.8f \nPHI: %.1f THETA: %.1f' % (4, phi, theta),
             xy=(0,0), xytext=(80,10), fontsize=12, color='black')
fig.canvas.draw()
plt.show()
# save_yaml_metadata(dt.generate_out_file(cfg.output_sim), cfg)
# np.save(dt.generate_out_file(cfg.output_sim), image_dict)
