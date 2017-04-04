#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time
from StringIO import StringIO
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import datetime
from scipy import misc

from pyfpm import web
import pyfpm.local as local
from pyfpm.fpmmath import set_iterator, reconstruct, generate_pupil
# from pyfpm.data import json_savemeta, json_loadmeta
import pyfpm.data as dt
from pyfpm.data import save_yaml_metadata


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
out_file = os.path.join(cfg.output_sim,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
in_file = os.path.join(cfg.output_sim, '2017-03-06_10:48:02.npy')
iterator = set_iterator(cfg)
client = local.SimClient(cfg=cfg)
# pc = PlatformCoordinates()

task = 'test'
if task is 'acquire':
    image_dict = dict()
    save_yaml_metadata(out_file, cfg)
    for index, theta, phi, power in iterator:
        image_dict[(theta, phi)] = client.acquire(theta, phi, power)
    np.save(out_file, image_dict)
