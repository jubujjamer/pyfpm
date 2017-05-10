#!/usr/bin/python
# -*- coding: utf-8 -*-
""" image_analysis.py

Last update: 08/05/2017
Image analysis script..

"""
import os
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import itertools as it

import pyfpm.fpmmath as fpm
import pyfpm.data as dt
from pyfpm.coordinates import PlatformCoordinates
# Simulation parameters
cfg = dt.load_config()

pc = PlatformCoordinates(cfg=cfg)
pc.generate_model(cfg.plat_model)

image_dict, image_cfg = dt.open_sampled('2017-04-12_200840.npy')

iterator = fpm.set_iterator(image_cfg)
fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
plt.grid(False)
fig.show()



def parametrize_circle(contour):
    r = 453
    p0 = [160, 570]
    meas_min = 1e18
    p_min = p0
    r_min = r
    for px, py, r in it.product(range(110, 140), range(550, 600), range(430, 470)):
        measure = sum(((contour[:, 1]-px)**2+(contour[:, 0]-py)**2-r**2)**2)
        if measure < meas_min:
            meas_min = measure
            p_min = [px, py]
            r_min = r
            # print(meas_min, p_min, r)
        print(r_min, p_min)
    return p_min, r_min


for index, theta, phi in iterator:
    # if not(theta == 0 and phi == 8):
    #     continue
    im_array = image_dict[theta, phi]
    # im_array[im_array < 90] = 0
    # im_array[im_array > 95] = 50
    contours = measure.find_contours(im_array, 0.8)

    ax1.cla()
    ax1.imshow(im_array, cmap=cm.hot, vmin=0, vmax=255)
    ax1.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f'
                 % (np.mean(im_array), phi, theta),
                 xy=(0, 0), xytext=(80, 10), fontsize=12, color='white')
    # n_cont = np.argmax([len(contour) for n, contour in enumerate(contours)])
    # contour = contours[n_cont]
    # ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ## Encuentro la circunferencia que mejor se adapta a los datos
    # p0, r_min = parametrize_circle(contour)
    # circ_theta = np.arange(0, 360)
    #
    # cx = r_min*np.cos(np.radians(circ_theta))+p0[0]
    # cy = r_min*np.sin(np.radians(circ_theta))+p0[1]
    # ax1.plot(cx, cy, linewidth=2, color='green')
    # ax1.plot(p0[0], p0[1], 'gx', linewidth=2)
    fig.canvas.draw()
    time.sleep(1)
plt.show()
