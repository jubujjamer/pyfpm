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

from pyfpm import web
from pyfpm.reconstruct import fpm_reconstruct
from pyfpm.fpmmath import set_iterator
from pyfpm.data import save_yaml_metadata
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
cfg = dt.load_config()
samples, comp_cfg = dt.open_sampled('2017-05-22_172249.npy')
background, comp_cfg = dt.open_sampled('2017-05-22_172249_blank.npy')

# Connect to a web client running serve_microscope.py
pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)

# reconstruction
rec = fpm_reconstruct(samples, background, iterator, cfg=cfg, debug=True)
