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
im_out = fpm_reconstruct_classic(samples=samples, it=iterator, cfg=sim_cfg)
print('--- %s seconds ---' % (time.time() - start_time))

original = client.im_array
pherr = np.angle(original)-np.angle(im_out)
abserr = np.abs(original)-np.abs(im_out)

fig, (axes1, axes2) = plt.subplots(2, 3, figsize = [15, 8])
axes1[0].imshow(np.angle(original)), axes1[0].set_title('Original Phase')
axes1[1].imshow(np.angle(im_out)), axes1[1].set_title('Reconstructed Phase')
axes1[2].plot(pherr[200,:]), axes1[2].set_title('%.1f' % np.max(pherr))

axes2[0].imshow(np.abs(samples[(15, 15)])), axes2[0].set_title('Original Magnitude')
axes2[1].imshow(np.abs(im_out)), axes2[1].set_title('Reconstructed Magnitude')
axes2[2].imshow(abserr), axes2[2].set_title('%.1f' % np.max(abserr))

plt.show()
