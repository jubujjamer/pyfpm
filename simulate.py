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
import os

from pyfpm import web
import pyfpm.local as local
from pyfpm.fpmmath import set_iterator, reconstruct, itertest
# from pyfpm.data import json_savemeta, json_loadmeta
import pyfpm.data as dt
from pyfpm.data import save_yaml_metadata

from scipy import misc
from StringIO import StringIO

# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
out_file = os.path.join(cfg.output_sim,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
in_file = os.path.join(cfg.output_sim,
                                  '2017-03-06_10:48:02.npy')
iterator = set_iterator(cfg)
client = local.SimClient(cfg=cfg)
# pc = PlatformCoordinates()
task = 'visualize'
if task is 'acquire':
    image_dict = dict()
    save_yaml_metadata(out_file, cfg)
    for index, theta, phi, power in iterator:
        image_dict[(theta, phi)] = client.acquire(theta, phi, power)
    np.save(out_file, image_dict)

elif task is 'reconstruct':
    start_time = time.time()
    rec = recontruct(in_file, iterator, cfg=cfg, debug=False)
    print('--- %s seconds ---' % (time.time() - start_time))
    plt.imshow(rec), plt.gray()
    plt.show()

if task is 'test_and_measure':
    image_dict = np.load(out_file)
    for index, theta, phi, power in iterator:
        if phi == 20 or phi == 40:
            im_array = image_dict[()][(theta, phi)]
            intensity = np.mean(im_array)
            print('int: %f, theta: %d, phi: %d' % (intensity, theta, phi))
            ax = plt.gca() or plt
            ax.imshow(im_array, cmap=plt.get_cmap('gray'))
            ax.get_figure().canvas.draw()
            plt.show(block=False)

if task is 'visualize':
    from pyfpm.plotsgui import plot_crossections

    img = client.load_image(cfg.input_mag)
    image = misc.imread(StringIO(img), 'gray')
    plot_crossections(image)
