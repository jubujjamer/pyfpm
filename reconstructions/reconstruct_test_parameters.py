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

import pyfpm.local as local
from pyfpm.reconstruct import fpm_reconstruct_wrapper
import pyfpm.coordtrans as ct
import pyfpm.data as dt
import pyfpm.inspector as ins

# Simulation parameters
cfg = dt.load_config()
mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
client = local.SimClient(cfg=cfg)
samples, sample_cfg = dt.open_sampled('20180628_175813.npy', mode='sampling')
iterator = ct.set_iterator(cfg)
 # Reconstruction
fpm_reconstruct_wrapper(samples=samples, it=iterator,
                             cfg=sample_cfg, debug=cfg.debug)
plt.show()
