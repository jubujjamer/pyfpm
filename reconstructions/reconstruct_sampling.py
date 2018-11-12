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
from numpy import abs, angle

import pyfpm.local as local
from pyfpm.reconstruct import fpm_reconstruct_epry, fpm_reconstruct
import pyfpm.coordtrans as ct
import pyfpm.data as dt
import pyfpm.inspector as ins

# Simulation parameters
cfg = dt.load_config()
mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
client = local.SimClient(cfg=cfg)
# samples, sample_cfg = dt.open_sampled('20181107_183532.npy', mode='sampling')
samples, sample_cfg = dt.open_sampled('simtest.npy', mode='simulation')

iterator = ct.set_iterator(cfg)

# total_time = 0
# for it in iterator:
#     nx, ny = it['nx'], it['ny']
#     iso, ss, power = it['acqpars']
#     total_time += ss/1E6
#     print(nx, ny, ss, total_time)

# First inspection
# iterator = ins.inspect_iterator(iterator, sample_cfg)
# iterator = ins.inspect_samples(iterator, samples, sample_cfg)
# ins.inspect_pupil(sample_cfg)
# Reconstruction
im_out, pupil =  fpm_reconstruct_epry(samples=samples, it=iterator,
                             cfg=sample_cfg, debug=cfg.debug)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(abs(im_out)), plt.gray()
ax2.imshow(angle(im_out), vmin=-.5, vmax=.5), plt.gray()
ax3.imshow(samples[(15, 15)][:sample_cfg.patch_size[0],:sample_cfg.patch_size[1]]), plt.hot()
ax4.imshow(abs(pupil)), plt.hot()
plt.show()
