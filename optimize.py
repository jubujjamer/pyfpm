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
import os
import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc, signal
import numpy as np
import time
import yaml
import itertools as it

import pyfpm.fpmmath as fpm
from pyfpm.data import save_yaml_metadata
# from pyfpm.data import json_savemeta, json_loadmeta
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates
import pyfpm.local as local
# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)
simclient = local.SimClient(cfg=cfg)
pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)

in_file = os.path.join(cfg.output_sample, '2017-04-05_113601.npy')

image_dict = np.load(in_file)[()]

def change_pc_parameters(pc, height, ptilt, sc, toff):
    pc.height = height
    pc.platform_tilt = ptilt
    pc.source_center = sc #[xcoord, ycoord] of the calibrated center
    tcorr, pcorr = pc.source_coordinates(mode='angular')
    tcorr += toff
    return tcorr, pcorr

# Last min: [90, 2.1, 1.9, 33]
height_range = range(88, 92)
ptilt_range = it.product([0], [2.1])
sc_range = [[0, 1.9]]
toff_range = range(20, 25)

cum_diff_min = 10
pars_min = list()

for h, pt, sc, toff in it.product(height_range, ptilt_range, sc_range,toff_range):
    iterator = fpm.set_iterator(cfg)
    cum_diff = 0
    for index, theta, phi in iterator:
        pc.set_coordinates(theta=theta, phi=phi, units='degrees')
        tcorr, pcorr = change_pc_parameters(pc, h, pt, sc, toff)
        #####################################################################
        sim_im_array = simclient.acquire(tcorr, pcorr, power=100)
        im_array = fpm.crop_image(image_dict[(theta, phi)], cfg.video_size, 170, 245)
        sim_im_array /= np.max(sim_im_array)
        im_array /= np.max(im_array)
        diff = np.sum(np.abs(sim_im_array - im_array)/(150*150))
        cum_diff += diff
    if cum_diff < cum_diff_min:
        cum_diff_min = cum_diff
        pars_min = [h, pt, sc, toff]
    print("Cumulative :", cum_diff, "Parameters", h, pt, sc, toff, "Partial minima", cum_diff_min, pars_min)
plt.show()
