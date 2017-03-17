#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File sample.py

Last update: 28/10/2016
To be used as a remote client for a microscope.
Before runing this file make sure there is a server running serve_microscope.py
hosted in some url.

Usage:

"""
from StringIO import StringIO
import time
import os
import datetime


import matplotlib.pyplot as plt
import h5py
from scipy import misc
import numpy as np
import time
import yaml

from pyfpm import web
from pyfpm.fpmmath import set_iterator, reconstruct, preprocess, rec_test
from pyfpm.data import save_yaml_metadata
# from pyfpm.data import json_savemeta, json_loadmeta
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
CONFIG_FILE = 'config.yaml'
cfg = dt.load_config(CONFIG_FILE)

out_file = os.path.join(cfg.output_sample,
                        '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
in_file = os.path.join(cfg.output_sample,
                        './2017-03-06_18:00:43.npy')
blank_images = os.path.join(cfg.output_sample,
                        './2017-03-06_18:41:51.npy')
json_file = './output_sim/out.json'
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
# ns = 0.3  # Complement of the overlap between sampling pupils
# Simulation parameters
image_size = cfg.video_size
wavelength = cfg.wavelength
pixelsize = cfg.pixelsize  # See jupyter notebook
phi_min, phi_max, phi_step = cfg.phi
theta_min, theta_max, theta_step = cfg.theta
pupil_radius = cfg.pupil_size/2

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip

# Connect to a web client running serve_microscope.py
client = web.Client(server_ip)
pc = PlatformCoordinates()
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
# Opens input image as if it was sampled at pupil_pos = (0,0) with high
# resolution details
iterator = set_iterator(cfg)

task = 'acquire'
if task is 'acquire':
    image_dict = dict()
    save_yaml_metadata(out_file, cfg)
    for index, theta, phi, power in iterator:
        pc.set_coordinates(theta, phi, units='degrees')
        [theta_plat, phi_plat, shift, power] = pc.parameters_to_platform()
        print("parameters to platform", theta_plat, phi_plat, shift, power)
        img = client.acquire(theta_plat, phi_plat, shift, power)
        im_array = misc.imread(StringIO(img.read()), 'RGB')
        image_dict[(theta, phi)] = im_array
        # ax = plt.gca() or plt
        # ax.imshow(im_array)
        # ax.get_figure().canvas.draw()
        # plt.show(block=False)
    print(out_file)
    np.save(out_file, image_dict)
    client.acquire(0, cfg.servo_init, 0)

elif task is 'reconstruct':
    start_time = time.time()
    rec = reconstruct(in_file, blank_images, iterator, cfg=cfg, debug=True)
    print('--- %s seconds ---' % (time.time() - start_time))
    plt.imshow(rec), plt.gray()
    plt.show()

if task is 'complete_scan':
    print(task)
    json_savemeta(json_file, image_size, pupil_radius, theta_max, phi_max)
    client.complete_scan(color)

if task is 'test_and_measure':
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    im1 = ax.imshow(np.ones((480,640)), cmap=plt.get_cmap('hot'))
    fig.show()
    while 1:
        img = client.get_camera_picture()
        im_array = misc.imread(StringIO(img.read()), 'RGB')
        ax.cla()
        ax.imshow(im_array, cmap=plt.get_cmap('hot'))
        fig.canvas.draw()

if task is 'calibration':
    print('calibration')
    for index, theta, phi, power in iterator:
        pc.set_in_degrees(theta, phi)
        [theta_plat, phi_plat, shift, power] = pc.parameters_to_platform()
        print("parameters to platform", theta_plat, phi_plat, shift, power)
        img = client.acquire(theta_plat, phi_plat, shift, power)
        im_array = misc.imread(StringIO(img.read()), 'RGB')
        image_dict[(theta, phi)] = im_array
        im_array = misc.imread(StringIO(img.read()), 'RGB')
        # intensity = np.mean(im_array)
        ax = plt.gca() or plt
        ax.imshow(im_array, cmap=plt.get_cmap('gray'))
        ax.get_figure().canvas.draw()
        plt.show(block=False)
        image_dict[(theta, phi)] = im_array
    print(out_file)
    np.save(out_file, image_dict)
    client.acquire(0, 0, 0)

if task is 'testing':
    for index, theta, phi, power in iterator:
        pc.set_coordinates(theta, phi, units='degrees')
        [theta_plat, phi_plat, shift, power] = pc.parameters_to_platform()
        print("parameters", theta, phi)
        print("parameters to platform: theta %i, phi %i, shift %i" % (theta_plat, phi_plat, shift))
        # img = client.acquire(theta_plat, phi_plat, shift, power)
        # im_array = misc.imread(StringIO(img.read()), 'RGB')
        # image_dict[(theta, phi)] = im_array
        # ax = plt.gca() or plt
        # ax.imshow(im_array)
        # ax.get_figure().canvas.draw()
        # plt.show(block=False)

if task is 'fix':
    fixed_dict = preprocess(in_file, blank_images, iterator, cfg=cfg, debug=True)
    # np.save('/output_sampling/fixed.npy', fixed_dict)
