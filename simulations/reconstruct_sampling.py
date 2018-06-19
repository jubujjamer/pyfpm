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

# Simulation parameters
cfg = dt.load_config()

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
iterator = ct.set_iterator(cfg)
client = local.SimClient(cfg=cfg)
#20180615_190312
#20180615_190312
# 20180618_163559
# 20180618_142136
samples, sim_cfg = dt.open_sampled('20180618_163559.npy', mode='sampling')

# Reconstruction
start_time = time.time()

hr = int(client.lhscale)*int(cfg.patch_size[0])
rec, phase = fpm_reconstruct(samples=samples, hrshape=[hr, hr], it=iterator,
                             cfg=cfg, debug=cfg.debug, pupil_radius=client.pupil_radius,
                             kdsc=client.kdsc)
print('--- %s seconds ---' % (time.time() - start_time))
plt.show()
if not cfg.debug:
    import matplotlib.pylab as plt
    plt.imshow(rec), plt.gray()
    plt.show()
