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
from scipy import misc
import numpy as np
import time
import yaml
import pygame

from pyfpm import web
from pyfpm.fpmmath import set_iterator, reconstruct, rec_test
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
                        '2017-04-05_113601.npy')
print(in_file, cfg.output_sample)
blank_images = os.path.join(cfg.output_sample,
                        './2017-04-05_16:17:27.npy')
json_file = './output_sim/out.json'
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
# ns = 0.3  # Complement of the overlap between sampling pupils
# Simulation parameters
image_size = cfg.video_size
wavelength = cfg.wavelength
pixelsize = cfg.pixel_size  # See jupyter notebook
phi_min, phi_max, phi_step = cfg.phi
theta_min, theta_max, theta_step = cfg.theta
pupil_radius = cfg.pupil_size/2

mode = cfg.task
itertype = cfg.sweep
server_ip = cfg.server_ip

# Connect to a web client running serve_microscope.py
client = web.Client(server_ip)
pc = PlatformCoordinates()
pc.generate_model(cfg.plat_model)
# Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
# Opens input image as if it was sampled at pupil_pos = (0,0) with high
# resolution details
iterator = set_iterator(cfg)

task = 'reconstruct'
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
        ax = plt.gca() or plt
        ax.imshow(im_array)
        ax.get_figure().canvas.draw()
        plt.show(block=False)
    print(out_file)
    np.save(out_file, image_dict)
    client.acquire(0, cfg.servo_init, 0)

elif task is 'reconstruct':
    start_time = time.time()
    rec = fpm_reconstruct(in_file, iterator, cfg=cfg, debug=True)
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
    image_dict = np.load(in_file)[()]
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    fig.show()
    for k in image_dict.keys():
        ax.cla()
        img = image_dict[k]
        ax.imshow(img, cmap=plt.get_cmap('hot'))
        fig.canvas.draw()
        misc.imsave('%s.png' % k[0], img)


if task is 'fix':
    fixed_dict = preprocess(in_file, blank_images, iterator, cfg=cfg, debug=True)
    # np.save('/output_sampling/fixed.npy', fixed_dict)

if task is 'manual_move':
    image_dict = dict()
    if cfg.plat_model == 'nomodel':
        plat_units = 'deg_shift'
    else:
        plat_units = 'degrees'
    tp = np.array([0, 0])  # Variables theta and phi
    [ts, ps] = [-20, 2]
    power_set = 0
    shift_set = 0
    pygame.init()
    # Loop until the user clicks the close button.
    done = False
    # Initialize the joysticks
    pygame.joystick.init()
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    # im1 = ax.imshow(np.ones((480,640)), cmap=plt.get_cmap('hot'))
    fig.show()
    # -------- Main Program Loop -----------
    while done is False:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        save_this = False
        show_this = False
        # ax.cla()
        # EVENT PROCESSING STEP
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            if event.type == pygame.JOYBUTTONDOWN:
                buttons = joystick.get_numbuttons()
                power_set += (joystick.get_button(2)) - (joystick.get_button(0))
                shift_set += (joystick.get_button(3)) - (joystick.get_button(1))
                if joystick.get_button(4):
                    save_this = True
                    print("I'm going to save that")
                if joystick.get_button(5):
                    done = True
                if joystick.get_button(6):
                    show_this = True
            if event.type == pygame.JOYAXISMOTION:
                x = np.array([ts*joystick.get_axis(0), ps*joystick.get_axis(1)])
                tp = tp + x
            pc.set_coordinates(tp[0], tp[1], shift_set, units=plat_units)
            [theta_plat, phi_plat, shift_plat, power_plat] = pc.parameters_to_platform()
            power_set = max(0, power_set)
            power_set = min(cfg.max_power, power_set)
            print("parameters to platform", theta_plat, phi_plat, shift_plat, power_set)
            client.just_move(theta_plat, phi_plat, shift_plat, power_set)
            if show_this:
                print("Showing")
                img = client.acquire(theta_plat, phi_plat, shift_plat, power_set)
                im_array = misc.imread(StringIO(img.read()), 'RGB')
                ax.imshow(im_array, cmap=plt.get_cmap('hot'))
                fig.canvas.draw()
            if save_this:
                image_dict[(tp[0], tp[1])] = im_array
    np.save(out_file, image_dict)
    client.acquire(0, cfg.servo_init, 0)
    # Close the window and quit.
    # If you forget this line, the program will 'hang'
    # on exit if running from IDLE.
    pygame.quit ()

if task is 'test':
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    fig.show()

    pc.set_coordinates(0, 0, units='degrees')
    [theta_plat, phi_plat, shift, power] = pc.parameters_to_platform()
    img = client.acquire(theta_plat, phi_plat, shift, power,
                         shutter_speed=6000, iso=100)
    im_array = misc.imread(StringIO(img.read()), 'RGB')
    ax.imshow(im_array, cmap=plt.get_cmap('hot'))
    fig.canvas.draw()
    plt.show()
