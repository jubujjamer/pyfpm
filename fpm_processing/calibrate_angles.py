#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File calibrate_angles.py

Last update: 22/05/2017

Usage:

"""
from StringIO import StringIO
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc, signal
import numpy as np
import h5py

import pyfpm.fpmmath as fpm
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates

cfg = dt.load_config()
# Start analysis
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
plt.grid(False)
fig.show()

with h5py.File('./misc/centroids_1.h5', 'r') as hf:
    centroid_data = hf['centroids/originals'][...]

circle = dict()
for theta, shift, cx, cy in centroid_data:
    # print(shift, cx, cy)
    try:
        circle[shift].append([cx, cy])
    except:
        circle[shift] = [[cx, cy]]

# Fitro outliers
for key in circle.keys():
    for i, c in reversed(list(enumerate(circle[key]))):
        rad = sum((c-np.mean(circle[key], axis=0))**2)
        print(key, rad)
        if rad > 10000:
            del circle[key][i]


colors = iter(cm.gist_rainbow(np.linspace(0,1,10)))
riter = iter([8, 14, 22, 30, 38, 46, 55, 63, 71, 80])
centers = list()
for key in circle.keys():
    color = colors.next()
    for c in circle[key]:
        ax1.plot(c[0], c[1],'o', color = color, linewidth=2, markersize=5)
    center = np.mean(circle[key], axis=0)
    centers.append(center)
    ax1.plot(center[0], center[1], '*', color=color, linewidth=2, markersize=5)
    r = riter.next()
    print((r/8.)/cfg.sample_height, np.degrees(np.arctan((r/8.)/cfg.sample_height)) ) 
    theta = np.arange(0, 360, 1)
    circle_rec = [r*np.cos(theta)+centers[0][0], r*np.sin(theta)+centers[0][1]]
    ax1.plot(circle_rec[0], circle_rec[1], '*', color=color, linewidth=1, markersize=2)

ax1.set_xlim([0, 320])
ax1.set_ylim([0, 240])
ax1.set_aspect('equal')
plt.show()
