#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File acquire_complete_set.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
from io import StringIO
import time

import numpy as np
import itertools as it
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import scipy
from scipy import misc, signal, optimize

from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import h5py

import pyfpm.fpmmath as fpm
from pyfpm import web
import pyfpm.coordtrans as ct
import pyfpm.data as dt

def plotkernel(mykernel, ksize):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, ksize, 1)
    Y = np.arange(0, ksize, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, mykernel, rstride=1,
                           cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, np.max(mykernel))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def mydebug(*objs):
    import sys
    print("**debug: ", *objs, file=sys.stderr)

def make_kernel_2D(PSF, dims, debug=True):
    """
        PSF is the 2D kernel
        dims are is the side size of the image in order (r,c)
    """
    d = len(PSF) ## assmuming square PSF (but not necessarily square image)
    print("kernel dimensions=", dims)
    N = dims[0]*dims[1]
    if debug:
        mydebug("Making kernel with %d diagonals all %d long\n" % (d*d, N))
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
    ## for debugging
    if debug:
        mydebug("Offsets: ", offsets)
        mydebug("Diagonal heads", heads)
        mydebug("Diagonals", diags) # only useful for small test matrices
    ## create linear operator

    H = scipy.sparse.dia_matrix((diags, offsets),shape=(N,N), dtype=np.complex128,)
    return(H)

def make_blur_matrix(img, kernel_size=12, debug=True):
    n = kernel_size
    # k2 = np.zeros(shape=(n,n))
    # k2[n//2,n//2] = 1
    # sigma = kernel_size/5.0 ## 2.5 sigma
    # testk=scipy.ndimage.gaussian_filter(k2,sigma)  ## already normalized
    r = n//3
    P = fpm.create_source_pattern(shape='circle', ledmat_shape=[n, n], radius=r)
    # S1 = fpm.create_source_pattern(shape='semicircle', angle=0, ledmat_shape=[n, n], radius=r)
    # S2 = fpm.create_source_pattern(shape='semicircle', angle=180, ledmat_shape=[n, n], radius=r)
    # Hph = 1j*(signal.correlate2d(P, S1*P)-signal.correlate2d(P, S2*P))
    # Hph /= np.pi*np.max(np.imag(Hph))
    Hph = fftshift(fft2(P))
    Hph = np.arange(1, n**2+1).reshape(n,-1)*(1+2*1j)
    mydebug("kernel: ", Hph)

    # Hph = Hph
    if (debug):
        plt.imshow(np.abs(Hph))
        # plt.show()
        # plotkernel(np.abs(Hph), n)

    blurmat = make_kernel_2D(Hph, img.shape)
    mydebug("Convolution matrix: ", blurmat.toarray())

    return(blurmat)

def blur_noise_image(given_image, blur_matrix, noise_scale=0.002):
    '''
        This code applies a linear operator in the form of a matrix, similar to
        refblur = scipy.ndimage.convolve(img, testk, mode='constant', cval=0.0)
    '''
    ## apply blur + noise
    # reshaped to vector
    imgshape=given_image.shape
    N = np.prod(imgshape)
    blurredvect = blur_matrix.dot(given_image.reshape(-1)) + np.random.normal(scale=noise_scale,size=N)
    blurredimg = blurredvect.reshape(imgshape)
    return blurredimg

def makespdiag(pattern, N):
    """
    This makes a diagonal sparse matrix,similar to a convolution kernel operator

    e.g:
        makediag([1,2,1],5).todense()

    matrix([[ 2.,  1.,  0.,  0.,  0.],
            [ 1.,  2.,  1.,  0.,  0.],
            [ 0.,  1.,  2.,  1.,  0.],
            [ 0.,  0.,  1.,  2.,  1.],
            [ 0.,  0.,  0.,  1.,  2.]])
    """
    # strangely, all diags may have the same length
    diags=[] # empty list
    for i in pattern :
        diags.append(np.zeros(N) + i)
    n = len(pattern)
    positions = np.arange(-(n/2),(n/2)+1)
    ## print positions
    mat = scipy.sparse.dia_matrix((diags, positions), shape=(N, N))
    return mat

A = make_blur_matrix(np.zeros((3, 3)), kernel_size=2, debug=True)


b = A.dot(np.ones((9, 1)))
print(b)
