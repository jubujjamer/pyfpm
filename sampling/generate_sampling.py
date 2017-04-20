#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from pyfpm.fpmmath import *
import pyfpm.data as dt
from pyfpm.data import save_yaml_metadata
import pyfpm.local as local
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
CONFIG_FILE = '/home/lec/pyfpm/etc/config.yaml'
cfg = dt.load_config(CONFIG_FILE)
client = local.SimClient(cfg=cfg)
iterator = set_iterator(cfg)
#
pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)

image_dict = dict()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
fig.show()

for index, theta, phi in iterator:
    # pupil = generate_pupil(theta, phi, power, cfg.video_size,
    #                        cfg.wavelength, cfg.pixel_size, cfg.objective_na)
    pc.set_coordinates(theta=theta, phi=phi, units='degrees')
    t_corr, p_corr = pc.source_coordinates(mode='angular')
    power = 100
    im_array = client.acquire(t_corr, p_corr, power)
    image_dict[(theta, phi)] = im_array
    ax1.cla(), ax2.cla()
    img = ax1.imshow(im_array, cmap=plt.get_cmap('hot'), vmin=0, vmax=255)
    if index == 0:
        fig.colorbar(img)
    # plt.xlim([0,450])
    # plt.ylim([0,2.5])
    ax1.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f' % (np.mean(im_array), phi, theta),
                 xy=(0,0), xytext=(80, 10), fontsize=12, color='white')
    fig.canvas.draw()
plt.show()
save_yaml_metadata(dt.generate_out_file(cfg.output_sim), cfg)
np.save(dt.generate_out_file(cfg.output_sim), image_dict)
