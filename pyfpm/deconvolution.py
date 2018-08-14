#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File fpmmath.py

Last update: 28/10/2016

Usage:

"""
__version__ = "1.1.1"
__author__ = 'Juan M. Bujjamer'
__all__ = ['dcts', 'dct2', 'dctshift', 'calculate_pupil_radius', 'adjust_shutter_speed',
           'pixel_size_required', 'crop_image']
import scipy
import numpy as np
from scipy.fftpack import dct


def dcts(x):
    """ Scipy implementation of the cosine transform.

    Parameters
    ----------
    x : array
        column vector or matrix. If x is a matrix then dcts computes the DCT of each column.

    Returns
    -------
    array
        the discrete cosine transform of x.

    """
    from scipy.fftpack import dct
    xdct = dct(x, type=2, axis=0, norm='ortho')
    return xdct

def dct2(x):
    """ Implementation of the 2-D cosine transform from its 1-D version. The two dimensional DCT is obtained by computing a one-dimensional DCT of the columns followed by a one-dimensional DCT of the rows.

    Parameters
    ----------
    x : array
        Input signal.

    Returns
    -------
    array
        The two dimensional discrete cosine transform of x.

    """
    xdct = dcts(dcts(x.T).T)
    return xdct

def dctshift(PSF, center):
    """ Create an array containing the first column of a bluring matrix when implementing reflexive boundary conditions.

    Parameters
    ----------
    PSF : ndarray
        Array containing the point spread function.
    center : type
        List containig the center coordinate of the PSF.

    Returns
    -------
    array
        Array containing the first column of the blurring matrix.
    """
    m, n = np.shape(PSF);
    i = center[0]
    j = center[1]
    k = np.min([i, m-i, j-1, n-j])+1
    PP = PSF[i-k:i+k, j-k: j+k]
    Z1 = np.diag(np.ones(k+1), k-1)
    Z2 = np.diag(np.ones(k), k)
    PP = Z1@PP@Z1.T + Z1@PP@Z2.T + Z2@PP@Z1.T + Z2@PP@Z2.T
    Ps = np.zeros((m, n))*1j
    Ps[:2*k+1, :2*k+1] = PP
    return Ps


def idcts(x):
    xidct = dct(x, type=3, axis=0, norm='ortho')
    return xidct

def idct2(x):
    yidct2 = idcts(idcts(x.T).T)
    return yidct2


def gcv_tik(s, bhat):
    """ GCV_TIK Choose GCV parameter for Tikhonov image deblurring.
     This function uses generalized cross validation (GCV) to choose
     a regularization parameter for Tikhonov filtering.

     Input:
     s Vector containing singular or spectral values
     of the blurring matrix.
     bhat Vector containing the spectral coefficients of the blurred
     image.

     Output:
     alpha Regularization parameter.
     Reference: See Chapter 6,
     "Deblurring Images - Matrices, Spectra, and Filtering"
     by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
     SIAM, Philadelphia, 2006.  """

    def GCV(alpha, s, bhat):
        phi_d = 1/(np.abs(s)**2 + alpha**2)
        G = sum(abs(bhat*phi_d)**2)/(np.sum(phi_d)**2)
        return G

    from scipy.optimize import fminbound

    alpha = fminbound(GCV, np.min(abs(s)), np.max(abs(s)), args=[s, bhat])
    return alpha

def blured_image(PSF, X):
    yc, xc = np.array(np.shape(PSF))/2
    center = [int(yc), int(xc)]
    e1 = np.zeros_like((PSF))
    e1[0, 0] = 1
    S = dct2( dctshift(PSF, center) ) / dct2(e1)
    B = idct2(S*dct2(X))
    return B

def naive_deconv(PSF, B):
    yc, xc = np.array(np.shape(PSF))/2
    center = [int(yc), int(xc)]
    e1 = np.zeros_like((PSF))
    e1[0, 0] = 1
    S = dct2( dctshift(PSF, center) ) / dct2(e1)
    B = idct2(dct2(B)/S)
    return B


def tik_dct(PSF, B, alpha=0, cg=False, **kwargs):
    """ TIK_DCT Tikhonov image deblurring using the DCT algorithm.
        B, PSF, center, alpha must be supplied
    """
    yc, xc = np.array(np.shape(PSF))/2
    center = [int(yc), int(xc)]
    e1 = np.zeros_like((PSF))
    e1[0, 0] = 1
    bhat = dct2(B)
    S = dct2( dctshift(PSF, center) ) / dct2(e1)
    s = S
    D = np.conj(s)*s + np.abs(alpha)**2
    bhat = np.conj(s)*bhat
    xhat = bhat/D
    xhat = xhat.reshape(np.shape(B))
    X = idct2(xhat)

    G=None
    if cg:
        phi_d = 1/(np.abs(s)**2 + alpha**2)
        G = sum(abs(bhat*phi_d)**2)/(np.sum(phi_d)**2)
        G =  np.sum(G)
    return X, G
