#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File coordtrans.py
Last update: 12/04/2018

Description:

Usage:
"""
__version__ = "1.1.1"
__author__ = 'Juan M. Bujjamer'
__all__ = ['get_acquisition_pars', 'set_iterator', 'tidy', 'phi_rot']

import yaml
import numpy as np
from itertools import product, cycle
import matplotlib.pylab as plt

import pyfpm.data as dt
import pyfpm.coordtrans as ct
import pyfpm.fpmmath as fpmm

cfg = dt.load_config()


def inspect_iterator(iterator, cfg):
    N = 1024
    xx, yy = np.meshgrid(range(N), range(N))
    image = np.ones((N, N))
    def circle_mask(nx, ny, rad=10):
        # mask = np.ones((N, N))
        c = (xx-nx)**2+(yy-ny)**2
        circle_mask = [c < rad**2][0]
        return circle_mask

    conv_fact = N/15/2
    for it in iterator:
        nx, ny = it['indexes']
        nx = int(nx*conv_fact)
        ny = int(ny*conv_fact)
        image[circle_mask(nx, ny)] = 0

    plt.imshow(image)
    plt.show()
    return ct.set_iterator(cfg)


def inspect_samples(iterator, samples, cfg):
    # nplots = int(cfg.array_size)
    center = 15
    nplots = 7
    f, axarr = plt.subplots(nplots, nplots)
    for it in iterator:
        nx, ny = it['indexes']
        ny = 30-ny
        print(nx ,ny)
        sample = samples[it['indexes']]
        tocenter = center - (nplots-1)/2
        if np.abs(nx-center)>(nplots-1)/2 or np.abs(ny-center)>(nplots-1)/2:
            break
        cx = int(nx-tocenter)
        cy = int(ny-tocenter)
        axarr[cx, cy].imshow(sample)
        axarr[cx, cy].axis('off')
        axarr[cx, cy].text(30, 200, '%i %i' % (nx, ny))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    return ct.set_iterator(cfg)

def inspect_pupil(cfg):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    center = 15
    image_size = fpmm.get_image_size(cfg)
    pupil_radius = fpmm.get_pupil_radius(cfg)
    aberrations = [1E-6, 0]
    ps = fpmm.get_pixel_size(cfg)
    wlen = fpmm.get_wavelength(cfg)
    pupil = fpmm.aberrated_pupil(image_size=cfg.patch_size, pupil_radius=pupil_radius,
                    aberrations=aberrations, pixel_size=ps, wavelength=wlen)
    CTF = fpmm.generate_CTF(fx=0, fy=0, image_size=cfg.patch_size,
                            pupil_radius=pupil_radius)

    def put_colorbar(ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[0].imshow(np.abs(pupil*CTF), cmap='Blues')
    put_colorbar(axes[0], im)
    im = axes[1].imshow(np.imag(pupil*CTF), cmap='RdBu')
    put_colorbar(axes[1], im)
    plt.show()
    return ct.set_iterator(cfg)
