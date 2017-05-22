#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import time

import numpy as np
import itertools as it
import pygame

from pyfpm import web
from pyfpm.fpmmath import set_iterator
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

# Simulation parameters
cfg = dt.load_config()
out_file = dt.generate_out_file(cfg.output_sample)
# Connect to a web client running serve_microscope.py
client = web.Client(cfg.server_ip)
pc = PlatformCoordinates(cfg=cfg)
pc.generate_model(cfg.plat_model)
iterator = set_iterator(cfg)
# Start image acquisition
# for index, theta, phi in iterator:
#     print(theta, phi)
#     pc.set_coordinates(theta, phi, units='degrees')
#     [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
#     client.just_move(theta_plat, phi_plat, shift_plat, power=0)

# Joystick move

def move_client(pc, client, tpsp):
    pc.set_coordinates(theta=tpsp[0], phi=tpsp[1], shift=tpsp[2],
                       units='deg_shift')
    [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
    client.just_move(theta_plat, phi_plat, shift_plat, power=tpsp[3])

def test_limits(tpsp, cfg):
    min_theta = cfg.theta[0]
    max_theta = cfg.theta[1]
    min_phi = cfg.phi[0]
    max_phi = cfg.phi[1]
    max_shift = cfg.shift_max
    max_power = cfg.max_power

    tpsp[0] = max(min_theta, tpsp[0])
    tpsp[0] = min(max_theta, tpsp[0])
    tpsp[1] = max(min_phi, tpsp[1])
    tpsp[1] = min(max_phi, tpsp[1])
    tpsp[2] = max(0, tpsp[2])
    tpsp[2] = min(max_shift, tpsp[2])
    tpsp[3] = max(0, tpsp[3])
    tpsp[3] = min(max_power, tpsp[3])
    return tpsp


def button_action(pc, client, joystick, tpsp, tpsp_steps):
    done = False
    if joystick.get_button(0):
        tpsp[3] -= tpsp_steps[3]
        done = False
    if joystick.get_button(1):
        tpsp[3] += tpsp_steps[3]
        done = False
    if joystick.get_button(2):
        tpsp[1] -= tpsp_steps[1]
        done = False
    if joystick.get_button(3):
        tpsp[1] += tpsp_steps[1]
        done = False
    if joystick.get_button(4):
        print("Button 5, quitting")
        done = True
    return done, tpsp, tpsp_steps


def axis_action(pc, client, joystick, tpsp, tpsp_steps):
    xaxis = round(joystick.get_axis(0))
    yaxis = round(joystick.get_axis(1))
    tpsp[0] += tpsp_steps[0]*xaxis
    tpsp[2] += tpsp_steps[2]*yaxis


# Init values
tpsp = np.array([0, 0, 0, 0])  # theta, phi, shift, power
tpsp_steps_init = np.array([5, 1, 5, 10])
tpsp_steps_now = tpsp_steps_init

axis_pressed = False
button_pressed = False
count_pressed = 0
pc = PlatformCoordinates(cfg=cfg)  # init_pygame
pygame.init()
done = False  # Quit trigger
pygame.joystick.init()
while done is False:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    for event in pygame.event.get():
        if event.type is pygame.JOYBUTTONDOWN:
            button_pressed = True
        if event.type is pygame.JOYBUTTONUP:
            count_pressed = 0
            button_pressed = False
        if event.type is pygame.JOYAXISMOTION:
            axis_pressed = not axis_pressed
            count_pressed = 0
            tpsp_steps_now = tpsp_steps_init

    if axis_pressed or button_pressed:
        count_pressed += 1
        if count_pressed == 5:
            tpsp_steps_now = tpsp_steps_init*4
        if axis_pressed:
            axis_action(pc, client, joystick, tpsp, tpsp_steps_now)
        if button_pressed:
            time.sleep(.2)
            button_action(pc, client, joystick, tpsp, tpsp_steps_now)
        tpsp = test_limits(tpsp, cfg)
        move_client(pc, client, tpsp)
pygame.quit()
