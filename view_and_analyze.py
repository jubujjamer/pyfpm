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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
import numpy as np
import time
import yaml

from pyfpm.fpmmath import set_iterator, reconstruct, preprocess, rec_test
from pyfpm.data import save_yaml_metadata
# from pyfpm.data import json_savemeta, json_loadmeta
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates
from pyfpm import implot
# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)

out_file = os.path.join(cfg.output_sample,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
in_file = os.path.join(cfg.output_sample,
                        './2017-04-05_11:36:01.npy')
iterator = set_iterator(cfg)

image_dict = np.load(in_file)[()]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
plt.grid(False)
fig.show()
for index, theta, phi, power in iterator:
    print(index)
    time.sleep(0.1)
    im_array = image_dict[(theta, phi)]
    print(im_array.ravel())
    ax1.cla()
    ax2.cla()
    im_hist = np.histogram(im_array.ravel(), bins=50)
    print(im_hist[0][0])
    ax1.imshow(im_array, cmap=cm.hot)
    ax2.hist(im_array.ravel(), bins=512)
    fig.canvas.draw()


plt.show()
