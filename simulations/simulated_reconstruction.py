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
from pyfpm.reconstruct import fpm_reconstruct_wrappable, fpm_reconstruct_classic
import pyfpm.coordtrans as ct
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
iterator = ct.set_iterator(cfg)
client = local.SimClient(cfg=cfg)
samples, sim_cfg = dt.open_sampled('simtest.npy', mode='simulation')

# Reconstruction
start_time = time.time()
hr = int(client.lhscale)*int(cfg.patch_size[0])
im_out = fpm_reconstruct_classic(samples=samples, it=iterator, cfg=sim_cfg,  debug=False)
print('--- %s seconds ---' % (time.time() - start_time))
fig, axes = plt.subplots(1, 2, figsize = [15, 8])
axes[0].imshow(np.abs(im_out)), axes[0].set_title('A')
axes[1].imshow(np.angle(im_out)), axes[1].set_title('P')
plt.show()
