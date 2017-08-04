#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time

import matplotlib.pyplot as plt
import numpy as np
import datetime

from pyfpm import web
import pyfpm.local as local
from pyfpm.reconstruct import fpm_reconstruct
from pyfpm.fpmmath import set_iterator, generate_pupil
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()
mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
iterator = set_iterator(cfg)
client = local.SimClient(cfg=cfg)
samples, sim_cfg = dt.open_sampled('2017-08-04_18:33:33.npy', mode='simulation')

# Reconstruction
start_time = time.time()
rec, phase = fpm_reconstruct(samples=samples, backgrounds=None, it=iterator,
                             init_point=[0, 0], cfg=cfg, debug=True)
print('--- %s seconds ---' % (time.time() - start_time))
plt.imshow(rec), plt.gray()
plt.show()
