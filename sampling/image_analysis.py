#!/usr/bin/python
# -*- coding: utf-8 -*-
""" image_analysis.py

Last update: 08/05/2017
Image analysis script..

"""
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure



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

for index, theta, phi in iterator:
    im_array = image_dict[theta, phi]
    im_array [im_array < 90] = 0
    im_array [im_array > 95] = 255
    contours = measure.find_contours(im_array, 0.8)

    ax1.cla()
    ax1.imshow(im_array, cmap=cm.hot, vmin=0, vmax=255)
    ax1.annotate('Mean value: %.4f \nPHI: %.1f THETA: %.1f'
                 % (np.mean(im_array), phi, theta),
                 xy=(0, 0), xytext=(80, 10), fontsize=12, color='white')
    print([(n, len(contour)) for n, contour in enumerate(contours)])
    ax1.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    fig.canvas.draw()
plt.show()
