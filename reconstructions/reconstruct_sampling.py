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
from pyfpm.reconstruct import fpm_reconstruct_wrap, fpm_reconstruct
import pyfpm.coordtrans as ct
import pyfpm.data as dt
import pyfpm.inspector as ins

# Simulation parameters
cfg = dt.load_config()
mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
client = local.SimClient(cfg=cfg)
samples, sample_cfg = dt.open_sampled('20180618_142136.npy', mode='sampling')
iterator = ct.set_iterator(sample_cfg)

# First inspection
# iterator = ins.inspect_iterator(iterator, sample_cfg)
iterator = ins.inspect_samples(iterator, samples, sample_cfg)
# ins.inspect_pupil(sample_cfg)
# Reconstruction
mag, phase = fpm_reconstruct(samples=samples, it=iterator,
                             cfg=sample_cfg, debug=cfg.debug)
plt.show()
if not cfg.debug:
    import matplotlib.pylab as plt
    plt.imshow(mag), plt.gray()
    plt.show()
