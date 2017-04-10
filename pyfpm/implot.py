#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File visualization.py

Last update: 28/10/2016

Usage:

"""
__version__= "1.1.1"
__author__='Juan M. Bujjamer'
__all__=['plot_images', 'image_center', 'generate_pupil']

from io import BytesIO
from StringIO import StringIO

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy import misc

#
# class imagePlotter(object):
#     def __init__(self):
#         self.calls = 0
#         # if images_array is None:
#         #     return
#         self.images_array = list()
#         # self.axes = []
#         self.fig = []
#         self.columns = 2
#         self.rows = []
#
#     def update_plot(self, images_array=None):
#         total_plots = len(images_array)
#         self.images_array = images_array
#         self.rows = max(1, int(np.ceil(1.*total_plots/self.columns)))
#         def plot_image(ax, image):
#             ax.cla()
#             return ax.imshow(image, cmap=plt.get_cmap('gray'))
#
#         if self.calls == 0:
#             fig, axes = plt.subplots(self.rows, self.columns, figsize=(25, 15))
#
#             self.fig = fig
#             self.axes = axes
#             plt.grid(False)
#             axes_iter = iter(self.axes.ravel())
#             print("size", len(self.axes.ravel()))
#             for image in self.images_array:
#                 ax = axes_iter.next()
#                 print(ax)
#
#                 fig.colorbar(cmap=cm.hot, mappable=image, ax=ax)
#             fig.show()

        # elif len(images_array) ==  sum(np.shape(self.axes)):
        #     axes_iter = iter(self.axes.ravel())
        #     for image in self.images_array:
        #         ax = axes_iter.next()
        #         im = plot_image(ax, image)
        #         self.fig.colorbar(cmap=cm.hot, mappable=im, cax=ax)
        #         fig.canvas.draw()
        # self.calls += 1
#
def init_plot(total_plots=None):
    if total_plots is None:
        total_plots = 1
    columns = 2
    rows = max(1, int(np.ceil(1.*total_plots/columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(25, 15))
    plt.grid(False)
    fig.show()
    return fig, axes


def update_plot(images_array, fig, axes):
    def plot_image(ax, image):
        ax.cla()
        return ax.imshow(image, cmap=cm.hot)

    if len(images_array) ==  sum(np.shape(axes)):
        axes_iter = iter(axes.ravel())
        for image in images_array:
            ax = axes_iter.next()
            im = plot_image(ax, image)
            fig.colorbar(cmap=cm.hot, mappable=im, cax=ax)
        fig.canvas.draw()
