#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File sample.py

Last update: 23/05/2017

Usage:

"""
from StringIO import StringIO

import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import time
import yaml
import itertools as it

from pyfpm import web
from pyfpm.reconstruct import fpm_reconstruct
from pyfpm.fpmmath import set_iterator
from pyfpm.data import save_yaml_metadata
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
cfg = dt.load_config()
samples, comp_cfg = dt.open_sampled('2017-05-26_152307.npy')
background, comp_cfg = dt.open_sampled('2017-05-26_145449_blank.npy')

# Connect to a web client running serve_microscope.py
pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)

# reconstruction
dx = cfg.patch_size[0]
x_range = range(0, cfg.video_size[1], dx)[:-1]
y_range = range(0, cfg.video_size[0], dx)[:-1]
# for i, point in enumerate(it.product(x_range, y_range)):
#     init_point = [point[0], point[1]]
#     rec, phase = fpm_reconstruct(samples, background, iterator, init_point,
#                           cfg=cfg, debug=False)
#     misc.imsave('./misc/ph'+str(i)+'.png', phase)
#     misc.imsave('./misc/im'+str(i)+'.png', rec)
rec, phase = fpm_reconstruct(samples, background, iterator, [200, 170],
                            cfg=cfg, debug=False)
plt.show()
