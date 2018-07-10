#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File fpmmath.py

Last update: 28/10/2016

Usage:

"""
__version__ = "1.1.1"
__author__ = 'Juan M. Bujjamer'
__all__ = ['image_center', 'generate_pupil', 'fpm_reconstruct', 'calculate_pupil_radius', 'adjust_shutter_speed',
           'pixel_size_required', 'crop_image']

from io import BytesIO
from io import StringIO
import time
import yaml

import scipy
import itertools as it
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import fsolve, nnls, lsq_linear
from PIL import Image
from scipy import ndimage


def convmat(pattern, N):
    """
        Creates a convolution matrix from kernel
    """
def make_convolution_matrix(img, kernel=None, debug=True):
    kernel_size=kernel.shape
    PSF = kernel
    dims = img.shape
    d = len(PSF) ## assmuming square PSF (but not necessarily square image)
    N = dims[0]*dims[1]
    ## pre-fill a 2D matrix for the diagonals
    diags = np.zeros((d*d, N))*1j
    offsets = np.zeros((d*d))
    heads = np.zeros((d*d))*1j ## for this a list is OK
    i = 0
    for y in range(len(PSF)):
        for x in range(len(PSF[y])):
            diags[i,:] += PSF[y,x]
            heads[i] = PSF[y,x]
            xdist = d/2 - x
            ydist = d/2 - y ## y direction pointing down
            offsets[i] = (ydist*dims[1]+xdist)
            i+=1
    H = scipy.sparse.dia_matrix((diags, offsets),shape=(N,N))
    return H


def nl_solve(A, b, lm):
    n_variables = A.shape[1]
    print(n_variables)
    A2 = scipy.sparse.vstack((A, np.sqrt(lm)*np.eye(n_variables)))
    # A2 = np.concatenate([A, np.sqrt(lm)*np.eye(n_variables)])
    b2 = np.concatenate([b, np.zeros(n_variables)])
    result = lsq_linear(A=A2, b=b2)
    return result.x

def sparse_solve(A, b, lm):
    """
    Tykhonov regularization of an observed signal, given a linear degradation matrix
    and a Gamma regularization matrix.
    Formula is

    x* = (H'H + G'G)^{-1} H'y

    With y the observed signal, H the degradation matrix, G the regularization matrix.

    This function is better than dense_tykhonov in the sense that it does
    not attempt to invert the matrix H'H + G'G.
    """
    t1=time.time()
    N = len(b)
    G = scipy.sparse.identity(N)
    # G=makespdiag([1],N)  ## identity
    Gamma = lm*G
    H1 = A.T.dot(A) # may not be sparse any longer in the general case
    H2 = H1 + Gamma # same
    b2 = A.T.dot(b)
    result = scipy.sparse.linalg.cg(H2, b2)
    return result[0]
