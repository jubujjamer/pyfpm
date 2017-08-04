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

import pyfpm.local as local
from pyfpm.reconstruct import fpm_reconstruct
from pyfpm.fpmmath import set_iterator,  generate_pupil
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
cfg = dt.load_config()

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip
out_file = os.path.join(cfg.output_sim,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
iterator = set_iterator(cfg)
client = local.SimClient(cfg=cfg)
samples, comp_cfg = dt.open_sampled('2017-08-04_14:46:29.npy', mode='simulation')

# pc = PlatformCoordinates()

task = 'reconstruct'
if task is 'acquire':
    image_dict = dict()
    save_yaml_metadata(out_file, cfg)
    for index, theta, shift, acqpars in iterator:
        image_dict[(theta, phi)] = client.acquire(theta, phi, power)
    np.save(out_file, image_dict)


elif task is 'reconstruct':
    start_time = time.time()
    rec, phase = fpm_reconstruct(samples=samples, backgrounds=None, it=iterator,
                                init_point=[0, 0], cfg=comp_cfg, debug=True)
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

if task is 'test':
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    fig.show()
    image = client.acquire(0, 0, 100)
    ax.cla()
    pupil = generate_pupil(0, 0, 100, 50, [640, 480])
    im_ax = ax.imshow(image, cmap=plt.get_cmap('gray'))
    fig.canvas.draw()
    ax.format_coord = Formatter(im_ax)
    plt.show()
