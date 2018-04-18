#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import datetime

import pyfpm.local as local
import pyfpm.coordtrans as ct
import pyfpm.fpmmath as fpmm
import pyfpm.data as dt

# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(fname = 'simtest.npy')
iterator = ct.set_iterator(cfg)
simclient = local.SimClient(cfg=cfg)
imshape = simclient.im_array.shape
xc, yc = fpmm.image_center(imshape)


pupil_radius = simclient.pupil_radius
kdsc = simclient.kdsc
pup_list = list()

fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fig.show()
image_dict = dict()
for it in iterator:
    print(it['indexes'])
    [kx, ky] = ct.angles_to_k(it['theta'], it['phi'], kdsc)
    # print('theta: %.1f phi: %.1f' % (it['theta'], it['phi']))
    pup_array = fpmm.pupil_image(yc+ky, yc+kx, pupil_radius, imshape)
    pup_list.append(pup_array)
    ax1.cla()
    img = ax1.imshow(pup_array, cmap=plt.get_cmap('hot'))
    fig.canvas.draw()
dt.save_yaml_metadata(out_file, cfg)
np.save(out_file, image_dict)
