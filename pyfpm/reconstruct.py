#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File reconstruct.py

Last update: 03/03/2017
FPM reconstruction methods. Originally based as an implementation of the
alternating projections (Gerchberg-Saxton) method.

Usage:

"""
import time
import yaml

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from scipy import ndimage
from numpy import abs, exp, angle, pi, imag, real, arctan, amax, conj
from numpy.random import rand, randn

# from pyfpm.coordinates import PlatformCoordinates
import pyfpm.fpmmath as fpmm
from . import coordtrans as ct

# from . import implot
# import fpmmath.optics_tools as ot

def get_mask(samples, backgrounds=None, xoff=None, yoff=None, cfg=None):
    """ Returns a thresholded mask where an imaged object is suspected to be.
    It could be useful in high contrast images where object borders are
    sufficiently well determined.

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        xoff: offset in the 'x' direction of this sample's center.
        yoff: offset in the 'y' direction of this sample's center.
        cfg: configuration (named tuple)
    Returns:
    --------
        (ndarray) mask containing the object to be reconstructed.
    """
    corr_ims = list()
    iterator = fpmm.set_iterator(cfg)
    for index, theta, shift in iterator:
        image = samples[(theta, shift)]
        image = fpmm.crop_image(image, cfg.patch_size, xoff, yoff)
        # image, image_size = image_rescaling(image, cfg)
        background = backgrounds[(theta, shift)]
        background = fpmm.crop_image(background, cfg.patch_size, xoff, yoff)
        # background, image_size = image_rescaling(background, cfg)
        corr_ims.append(image_correction(image, background, mode='background'))
    mask = np.mean(corr_ims, axis=0)
    #
    thres = 140 # hardcoded
    mask[mask < thres] = 1
    mask[mask > thres] = 0
    # print(Et[np.abs(Et) > .1])
    return mask


def image_correction(sample, background, mask=None, mode='threshold'):
    """ A set of image correction methods. They use backround sampled images
    or masks to perform the correction.

    Args:
    -----
        sample: the (xpy, npx) image to be corrected.
        background: the (xpy, npx) background corresonding to the sample.
        mask: object mast (see the get_mask() method).
        mode: the method to be applied for correction:
            * threshold: aplies a threshold to the sampled image.
            * background: substracst or corrects background intensity.
            * mask: only corrects inside a mask.

    Returns:
    --------
        (ndarray) upsampled image and final high resolution shape
    """
    if mode == 'threshold':
        image_corrected = sample
    elif mode == 'background':
        # image *= (255.0/image.max())
        # background *= (255.0/background.max())
        image_corrected = sample / background
        # image_corrected *= (255.0/image_corrected.max())
        # image_corrected -= np.min(image_corrected[:])-1
    elif mode == 'mask':
        image_corrected = sample - 1.*background
        image_corrected -= np.min(image_corrected[:])-5
        image_corrected = sample*mask+10
    # im_array = (im_array-.1*blank_array)
    # im_array -= np.min(im_array[:])
    # im_array[im_array < np.min(im_array)+2] = 0
    # image_corrected /= np.max(image_corrected)
    return image_corrected


def image_rescaling(lr_image, cfg):
    """ Image rescaling acording to sampling requirements. The numerical
    aperture, wavelength, pixel size and maximum sampling are used to calculate
    the upsampled image size to contain all the new information.

    Args:
    -----
        image: the (xpy, npx) image to be rescaled
        cfg: configuration (named tuple)

    Returns:
    --------
        (ndarray) upsampled image and final high resolution shape
    """
    phi_max = float(cfg.phi[1])
    wavelength = float(cfg.wavelength)
    na = float(cfg.objective_na)
    ps_required = fpmm.ps_required(phi_max, wavelength, na)
    scale_factor = cfg.pixel_size/ps_required
    Ih = ndimage.zoom(lr_image, scale_factor, order=0)  # HR image
    hr_shape = np.shape(Ih)
    return Ih, hr_shape


def preprocess_images(samples, backgrounds, xoff, yoff, cfg, corr_mode='background'):
    """ Applies a correction method to all the sampled images. Some of the
    methods need backroung images to substract illumination inhomogeneities.

    Args:phase
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        xoff: offset in the 'x' direction of this sample's center.
        yoff: offset in the 'y' direction of this sample's center.
        cfg: configuration (named tuple)
        corr_mode: the correction method
            * background: substracts  backround from samples.
            * bypass: does nothing (what a wonderful method!)

    Returns:
    --------
        (dictionary) dictionary with teh corrected samples.
    """
    iterator = ct.set_iterator(cfg)
    if corr_mode == 'background':
        for index, theta, shift, aqcpars in iterator:
            sample = fpmm.crop_image(samples[(theta, shift)],
                                     cfg.patch_size, xoff, yoff)
            background = fpmm.crop_image(backgrounds[(theta, shift)],
                                         cfg.patch_size, xoff, yoff)
            im_array = image_correction(sample, background, mode=corr_mode)
            im_array, resc_size = image_rescaling(im_array, cfg)
            samples[(theta, shift)] = im_array
    if corr_mode == 'bypass':
        do_nothing = 1
    return samples


def initialize(hrsize=None, backgrounds=None, xoff=None, yoff=None, cfg=None,
               mode='zero'):
    """ Initializes the algorithm using one of various modalities.

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        xoff: offset in the 'x' direction of this sample's center.
        yoff: offset in the 'y' direction of this sample's center.
        cfg: configuration (named tuple)
        mode: the modality according to which the algorithm is going to be
        initilized. Can be one of the following options:
            * zero: constant phase and zero amplitude.
            * transmission: the transmitted image at (0, 0) angles.
            * mean: takes all the samples and substracts the (measured)
                    backround. Then takes the mean of all of them.

    Returns:
    --------
        (ndarray) a complex image with the same size of the samples.
    """
    # image = fpmm.crop_image(samples[(0, 0)], cfg.patch_size, xoff, yoff)
    # image, image_size = image_rescaling(image, cfg)
    if mode == 'zero':
        Ih_sq = 0.5 * np.ones(hrsize)  # Homogeneous amplitude
        Ph = np.zeros(hrsize)          # and null phase
        Et = Ih_sq * np.exp(1j*Ph)
    elif mode == 'transmission':
        background = fpmm.crop_image(backgrounds[(0, 0)], cfg.patch_size, xoff, yoff)
        background, image_size = image_rescaling(background, cfg)
        image = image_correction(image, background, mode='background')
        Ih_sq = np.sqrt(image)
        Ph = np.zeros_like(Ih_sq)  # and null phase
        Et = Ih_sq * np.exp(1j*Ph)
    elif mode == 'mean':
        corr_ims = list()
        iterator = fpmm.set_iterator(cfg)
        for index, theta, shift in iterator:
            # Final step: squared inverse fft for visualization
            image = fpmm.crop_image(samples[(theta, shift)],
                                       cfg.patch_size, xoff, yoff)
            background = fpmm.crop_image(backgrounds[(theta, shift)],
                                         cfg.patch_size, xoff, yoff)
            image, image_size = image_rescaling(image, cfg)
            background, image_size = image_rescaling(background, cfg)
            corr_ims.append(image_correction(image, background, mode='background'))
        Ih = np.mean(corr_ims, axis=0)
        # Ph = 0.5+np.pi*np.abs(Et)/np.max(Et)
        Et = np.sqrt(Ih) * np.exp(1j*0)
    return Et


def generate_il(im_array, f_ih, theta, phi, cfg):
    """ Returns the low resolution sampled image with added reconstructed phase
    in the according spectral position. More detailed explanation:
    Takes the high resolution spectrum, calculates a low resolution sample in
    the spectral area occupied by the sampled image (im_array) by cuting it
    with the pupil at the (theta, phi), takes just the phase information in
    this area and replaces its modulus with the acquired im_array.
    Why this is outside the main function? Because it is also used in the
    quality metric (to be reimpemented here).
    """
    ps = fpmm.ps_required(cfg.phi[1], cfg.wavelength, cfg.na)
    image_size = np.shape(im_array)
    pupil = fpmm.generate_pupil(theta=theta, phi=phi, image_size=image_size,
                                wavelength=cfg.wavelength, pixel_size=ps,
                                na=cfg.na)
    pupil_shift = fftshift(pupil)
    # Step 2: lr of the estimated image using the known pupil
    f_il = ifft2(f_ih*pupil_shift)  # space pupil * fourier image
    Phl = np.angle(f_il)
    # Step 3: spectral pupil area replacement
    Il = np.sqrt(im_array) * np.exp(1j*Phl)  # Spacial update
    Iupdate = Il
    # Iupdate /= np.max(Iupdate)
    # Iupdate *= 150
    return Iupdate

def fpm_reconstruct(samples=None, it=None, cfg=None,  debug=False):
    """ FPM reconstructon using the alternating projections algorithm. Here
    the complete samples and (optional) background images are loaded and Then
    cropped according to the patch size set in the configuration tuple (cfg).

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        it: iterator with additional sampling information for each sample.
        init_point: [xoff, yoff] center of the patch to be reconstructed.
        cfg: configuration (named tuple)
        debug: set it to 'True' if you want to see the reconstruction proccess
               (it slows down the reconstruction).

    Returns:
    --------
        (ndarray) The reconstructed modulus and phase of the sampled image.
    """
    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    # objectRecover = initialize(hrshape, cfg, 'zero')
    lrsize = fpmm.get_image_size(cfg)
    pupil_radius = fpmm.get_pupil_radius(cfg)
    ps = fpmm.get_pixel_size(cfg)
    wlen = fpmm.get_wavelength(cfg)
    hrshape = fpmm.get_reconstructed_shape(cfg)
    kdsc = fpmm.get_k_discrete(cfg)

    # Initialization
    xc, yc = fpmm.image_center(hrshape)
    CTF = fpmm.generate_CTF(0, 0, [lrsize, lrsize], pupil_radius)
    # gk_prime = np.ones(hrshape)*exp(1j*rand(hrshape)*pi)
    # gfk = np.ones(hrshape)*exp(1j*rand(hrshape[0], hrshape[1])*pi)
    objectRecover = np.ones(hrshape)
    gfk = fftshift(fft2(objectRecover))  # shifted transform


    factor = (lrsize/hrshape[0])**2
    # For the convergence index
    N = len(samples)
    beta = 0.
    gm=0
    # lrarray = np.zeros((lrsize, lrsize, N))
    e_gk = 0
    gk_prev = np.zeros_like(gfk)
    gk = np.ones_like(gfk)
    gpk_prev = np.zeros_like(gfk)
    gpk = np.ones_like(gfk)
    for iteration in range(5):
        iterator = ct.set_iterator(cfg)
        # Patching for testing
        e_gk = np.sum((abs(gk) - abs(gk_prev))**2)/(np.sum(abs(gk)**2))
        e_gpk = np.sum((angle(gk)-angle(gk_prev))**2)/(np.sum(angle(gk)**2))
        gk_prev = gk
        print('Iteration %d | gk error %.4f' % (iteration, e_gk))
        print(e_gpk)
        if (e_gk < 5E-5):
            break
        for it in iterator:
            gm = gm*e_gk
            CTF = fpmm.generate_CTF(0, 0, [lrsize, lrsize], pupil_radius)
            acqpars = it['acqpars']
            # indexes, theta, phi = it['indexes'], it['theta'], it['phi']
            indexes, kx_rel, ky_rel = ct.n_to_krels(it, cfg=cfg)
            lr_sample = np.copy(samples[it['indexes']][:lrsize, :lrsize])
            # Calculating coordinates
            [ky, kx] = kdsc*ky_rel, kdsc*kx_rel
            kyl = int(np.round(yc+ky-(lrsize+1)/2))
            kyh = kyl + lrsize
            kxl = int(np.round(xc+kx-(lrsize+1)/2))
            kxh = kxl + lrsize
            # Fourier-space update
            delta_gfk = factor*gfk[kyl:kyh, kxl:kxh]*CTF
            # Step 2: lr of the estimated image using the known pupil
            delta_gk = ifft2(ifftshift(delta_gfk))+gm*rand(lrsize,lrsize)
            # phase = arctan(real(delta_gk)/(imag(delta_gk)+0.))
            phase = angle(delta_gk)
            gk_prime = lr_sample*exp(1j*phase)/factor
            delta_gfk = fftshift(fft2(gk_prime))*CTF
            gfk[kyl:kyh, kxl:kxh] = (1-CTF)*gfk[kyl:kyh, kxl:kxh]+delta_gfk
        gk = ifft2(fftshift(gfk))
    return gk, gk

def fpm_reconstruct_epry(samples=None, it=None, cfg=None,  debug=False):
    """ FPM reconstructon using the alternating projections algorithm. Here
    the complete samples and (optional) background images are loaded and Then
    cropped according to the patch size set in the configuration tuple (cfg).

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        it: iterator with additional sampling information for each sample.
        init_point: [xoff, yoff] center of the patch to be reconstructed.
        cfg: configuration (named tuple)
        debug: set it to 'True' if you want to see the reconstruction proccess
               (it slows down the reconstruction).

    Returns:
    --------
        (ndarray) The reconstructed modulus and phase of the sampled image.
    """
    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    lrsize = fpmm.get_image_size(cfg)
    pupil_radius = fpmm.get_pupil_radius(cfg)
    ps = fpmm.get_pixel_size(cfg)
    wlen = fpmm.get_wavelength(cfg)
    hrshape = fpmm.get_reconstructed_shape(cfg)
    kdsc = fpmm.get_k_discrete(cfg)

    objectRecover = np.ones(hrshape)
    xc, yc = fpmm.image_center(hrshape)
    CTF = fpmm.generate_CTF(0, 0, [lrsize, lrsize], 1.*pupil_radius)
    # print(lrsize,hrshape)
    # plt.imshow(CTF)
    # plt.show()
    # pupil = np.ones_like(CTF, dtype='complex')
    pupil = 1
    # pupil = fpmm.aberrated_pupil(image_size=[lrsize, lrsize], pupil_radius=pupil_radius, aberrations=[-.1E-6,], pixel_size=ps, wavelength=wlen)*CTF
    # pupil = (rand(lrsize, lrsize)*(1+0*1j)+1)*0.5
    # gfk = exp(1j*rand(hrshape[0], hrshape[1])*pi)*fpmm.generate_CTF(0, 0, [hrshape[0], hrshape[1]], pupil_radius*.1)
    # gfk = np.zeros(hrshape, dtype='complex')
    gfk = fftshift(fft2(objectRecover))
    # gfk += rand(hrshape[0], hrshape[1])+1
    # gfk[yc-lrsize//2:yc+lrsize//2, xc-lrsize//2:xc+lrsize//2] = 1+rand(lrsize, lrsize)
    # gfk[yc, xc] = 1
    # Steps 2-5
    factor = (float(lrsize)/hrshape[0])**2
    # For convergence index
    e_gk = 0
    gk_prev = np.zeros_like(gfk)
    gk = np.ones_like(gfk)*(1+1j)
    gpk_prev = np.zeros_like(gfk)
    gpk = np.ones_like(gfk)
    ek = 1
    gm = 0
    a = 1
    b = 1
    sum_lr = 0
    for it in ct.set_iterator(cfg):
        indexes, kx_rel, ky_rel = ct.n_to_krels(it=it, cfg=cfg)
        sum_lr += np.sum(samples[it['indexes']])
    for iteration in range(15):

        iterator = ct.set_iterator(cfg)
        print('Iteration n. %d' % iteration)
        # Patching for testing
        e_gk = np.sum((abs(gk) - abs(gk_prev))**2)/(np.sum(abs(gk)**2))
        e_gpk = np.sum((angle(gk)-angle(gk_prev))**2)/(np.sum(angle(gk)**2))
        gk_prev = gk
        print('Iteration %d | gk error %.2e | gpk error %.4f | ek %.2e |' % (iteration, e_gk, e_gpk, ek))
        # if (ek < 0.00001):
        #     break
        ek = 0
        gm = gm*e_gk
        for it in iterator:
            acqpars = it['acqpars']
            indexes, kx_rel, ky_rel = ct.n_to_krels(it=it, cfg=cfg, xoff=0., yoff=0.)
            lr_sample = np.copy(samples[it['indexes']][:lrsize, :lrsize])
            # From generate_il
            # Calculating coordinates
            [kx, ky] = kdsc*kx_rel, kdsc*ky_rel # k_rel is sin(phi)
            # print(kdsc)
            kyl = int(np.round(yc+ky-(lrsize+1)/2))
            kyh = kyl + lrsize
            kxl = int(np.round(xc+kx-(lrsize+1)/2))
            kxh = kxl + lrsize
            delta_gfk1 = gfk[kyl:kyh, kxl:kxh]*pupil*CTF
            # print(kyl, kyh, kxl, kxh, yc+kx, yc+ky)
            # Step 2: lr of the estimated image using the known pupil
            delta_gk = ifft2(ifftshift(delta_gfk1))
            gk_prime = 1/factor*lr_sample*delta_gk/abs(delta_gk)
            delta_gfk2 = fftshift(fft2(gk_prime))
            # update of the pupil and the fourier transform of the sample.
            delta_phi = delta_gfk2-delta_gfk1
            sampled_gfk = gfk[kyl:kyh, kxl:kxh]
            sampled_gfk += a*delta_phi*conj(CTF*pupil)/amax(abs(CTF*pupil))**2
            if iteration > -1:
                pupil += b*delta_phi*conj(sampled_gfk)/amax(abs(sampled_gfk))**2
            gfk[kyl:kyh, kxl:kxh] = sampled_gfk
            # test_img = delta_phi
            # plt.plot(abs(delta_phi[25,:]))
            # plt.show()
            ek += np.sum(abs(delta_gfk2-delta_gfk1)**2)/sum_lr**2
        gk = ifft2(fftshift(gfk))
        # plt.imshow(np.log(abs(gfk)))
        # plt.show()
    return gk, pupil

# def fpm_reconstruct_epry(samples=None, it=None, cfg=None,  debug=False):
#     """ FPM reconstructon using the alternating projections algorithm. Here
#     the complete samples and (optional) background images are loaded and Then
#     cropped according to the patch size set in the configuration tuple (cfg).
#
#     Args:
#     -----
#         samples: the acquired samples as a dictionary with angles as keys.
#         backgrounds: the acquired background as a dictionary with angles as
#                      keys. They must be acquired right after or before taking
#                      the samples.
#         it: iterator with additional sampling information for each sample.
#         init_point: [xoff, yoff] center of the patch to be reconstructed.
#         cfg: configuration (named tuple)
#         debug: set it to 'True' if you want to see the reconstruction proccess
#                (it slows down the reconstruction).
#
#     Returns:
#     --------
#         (ndarray) The reconstructed modulus and phase of the sampled image.
#     """
#     # Getting the maximum angle by the given configuration
#     # Step 1: initial estimation
#     # objectRecover = initialize(hrshape, cfg, 'zero')
#     lrsize = fpmm.get_image_size(cfg)
#     pupil_radius = fpmm.get_pupil_radius(cfg)
#     ps = fpmm.get_pixel_size(cfg)
#     wlen = fpmm.get_wavelength(cfg)
#     hrshape = fpmm.get_reconstructed_shape(cfg)
#     kdsc = fpmm.get_k_discrete(cfg)
#
#     objectRecover = np.ones(hrshape)
#     xc, yc = fpmm.image_center(hrshape)
#     CTF = fpmm.generate_CTF(0, 0, [lrsize, lrsize], pupil_radius)
#     pupil = fpmm.aberrated_pupil(image_size=[lrsize, lrsize], pupil_radius=pupil_radius, aberrations=[0,], pixel_size=ps, wavelength=wlen)
#     objectRecoverFT = fftshift(fft2(objectRecover))  # shifted transform
#     # Steps 2-5
#     factor = (lrsize/hrshape[0])**2
#     # For the convergence index
#     N = len(samples)
#     conv_index = 0
#     # lrarray = np.zeros((lrsize, lrsize, N))
#     for iteration in range(10):
#         iterator = ct.set_iterator(cfg)
#         conv_index = 0
#         print('Iteration n. %d' % iteration)
#         # Patching for testing
#         for it in iterator:
#             acqpars = it['acqpars']
#             # indexes, theta, phi = it['indexes'], it['theta'], it['phi']
#             indexes, kx_rel, ky_rel = ct.n_to_krels(it=it, cfg=cfg, xoff=0, yoff=0)
#             # r = 10*np.sqrt(indexes[0]**2+indexes[1]**2)
#             lr_sample = np.copy(samples[it['indexes']])
#             # lr_sample[lr_sample > 252] = 0
#             # lr_sample[lr_sample < 3] = 0
#             # lr_sample*=(5E4/(5.E5+acqpars[1]))
#             # print((1+5E4/acqpars[1]/4))
#             # From generate_il
#             # Calculating coordinates
#             [kx, ky] = kdsc*kx_rel, kdsc*ky_rel
#             kyl = int(np.round(yc+ky-(lrsize+1)/2))
#             kyh = kyl + lrsize
#             kxl = int(np.round(xc+kx-(lrsize+1)/2))
#             kxh = kxl + lrsize
#             # Il = generate_il(im_array, f_ih, theta, phi, cfg)
#             lowResFT = factor * objectRecoverFT[kyl:kyh, kxl:kxh]*pupil*CTF
#             # Step 2: lr of the estimated image using the known pupil
#             im_lowRes = ifft2(ifftshift(lowResFT))  # space pupil * fourier image
#             im_lowRes = 1/factor * lr_sample * np.exp(1j*np.angle(im_lowRes))
#             lowResFT2 = fftshift(fft2(im_lowRes))*CTF/pupil
#             ORFT = objectRecoverFT[kyl:kyh, kxl:kxh].ravel()
#             objectRecoverFT[kyl:kyh, kxl:kxh] +=1E-2* (lowResFT2-lowResFT)*np.conjugate(pupil)/np.max(np.abs(pupil.ravel())**2)
#             pupil +=1E-2*(lowResFT2-lowResFT)*np.conjugate(objectRecoverFT[kyl:kyh,kxl:kxh])/np.max(np.abs(ORFT)**2)
#     return ifft2(ifftshift(objectRecoverFT))


def fpm_reconstruct_wrappable(samples=None, it=None, cfg=None,  debug=False):
    """ FPM reconstructon using the alternating projections algorithm. Here
    the complete samples and (optional) background images are loaded and Then
    cropped according to the patch size set in the configuration tuple (cfg).

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        it: iterator with additional sampling information for each sample.
        init_point: [xoff, yoff] center of the patch to be reconstructed.
        cfg: configuration (named tuple)
        debug: set it to 'True' if you want to see the reconstruction proccess
               (it slows down the reconstruction).

    Returns:
    --------
        (ndarray) The reconstructed modulus and phase of the sampled image.
    """
    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    lrsize = fpmm.get_image_size(cfg)
    pupil_radius = fpmm.get_pupil_radius(cfg)
    ps = fpmm.get_pixel_size(cfg)
    wlen = fpmm.get_wavelength(cfg)
    hrshape = fpmm.get_reconstructed_shape(cfg)
    kdsc = fpmm.get_k_discrete(cfg)

    objectRecover = np.ones(hrshape)
    xc, yc = fpmm.image_center(hrshape)
    CTF = fpmm.generate_CTF(0, 0, [lrsize, lrsize], pupil_radius)
    objectRecoverFT = fftshift(fft2(objectRecover))  # shifted transform
    # Steps 2-5
    factor = (lrsize/hrshape[0])**2
    # For the convergence index
    N = len(samples)

    for iteration in range(1):
        iterator = ct.set_iterator(cfg)
        # Patching for testing
        for it in iterator:
            # Coordinates manipulation
            acqpars = it['acqpars']
            indexes, kx_rel, ky_rel = ct.n_to_krels(it=it, cfg=cfg, xoff=0, yoff=0)
            lr_sample = np.copy(samples[it['indexes']])
            [kx, ky] = kdsc*kx_rel, kdsc*ky_rel
            # Iteration step
            kyl = int(np.round(yc+ky-(lrsize+1)/2))
            kyh = kyl + lrsize
            kxl = int(np.round(xc+kx-(lrsize+1)/2))
            kxh = kxl + lrsize
            lowResFT = factor * objectRecoverFT[kyl:kyh, kxl:kxh]*CTF
            # Step 2: lr of the estimated image using the known pupil
            im_lowRes = ifft2(ifftshift(lowResFT))  # space pupil * fourier image
            im_lowRes = 1/factor * lr_sample * np.exp(1j*np.angle(im_lowRes))
            ## pupil correction update
            lowResFT2 = fftshift(fft2(im_lowRes))*CTF*1./pupil
            ORFT = objectRecoverFT[kyl:kyh, kxl:kxh].ravel()
            objectRecoverFT[kyl:kyh, kxl:kxh] += (lowResFT2-lowResFT)*np.conjugate(pupil)/np.max(np.abs(pupil.ravel())**2)
            pupil +=(lowResFT2-lowResFT)*np.conjugate(objectRecoverFT[kyl:kyh,kxl:kxh])/np.max(np.abs(ORFT)**2)
            ####################################################################
    im_out = ifft2(ifftshift(objectRecoverFT))
    return im_out


def fpm_reconstruct_wrapper(samples=None, it=None, cfg=None,  debug=False):
    zdefocus=0
    alpha=.1E-2
    beta = 1E-2
    xoff=0
    yoff=0
    zdefocus_array = np.linspace(3E-6, 3.1E-6, 9)
    fig, (axes) = plt.subplots(3, 3, figsize=(25, 15))
    fig.show()
    axes_iter = iter(axes)
    for zdefocus in zdefocus_array:
        ax = next(axes_iter)
        im_out = fpm_reconstruct_wrappable(samples=None, it=None, cfg=None,  debug=False, zdefocus=0,
                            alpha=1, beta = 1, xoff=0, yoff=0)


def fpm_reconstruct_wrap(samples=None, hrshape=None, it=None, pupil_radius=None,
                    kdsc=None, cfg=None,  debug=False):
    """ FPM reconstructon using the alternating projections algorithm. Here
    the complete samples and (optional) background images are loaded and Then
    cropped according to the patch size set in the configuration tuple (cfg).

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        it: iterator with additional sampling information for each sample.
        init_point: [xoff, yoff] center of the patch to be reconstructed.
        cfg: configuration (named tuple)
        debug: set it to 'True' if you want to see the reconstruction proccess
               (it slows down the reconstruction).

    Returns:
    --------
        (ndarray) The reconstructed modulus and phase of the sampled image.
    """
    from skimage.measure import compare_ssim as ssim

    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    # objectRecover = initialize(hrshape, cfg, 'zero')
    im_out = None
    objectRecover = np.ones(hrshape)
    lrsize = samples[(15, 15)].shape[0]
    xc, yc = fpmm.image_center(hrshape)

    def pupil_wrap(zfocus, radius):
        CTF = fpmm.generate_pupil(0, 0, [lrsize, lrsize], pupil_radius)
        # focus test
        # dky = 2*np.pi/(float(cfg.ps_req)*hrshape[0])
        kmax = np.pi/float(cfg.pixel_size)
        step = kmax/((lrsize-1)/2)
        kxm, kym = np.meshgrid(np.arange(-kmax,kmax+1,step), np.arange(-kmax,kmax+1, step));
        k0 = 2*np.pi/float(cfg.wavelength)
        kzm = np.sqrt(k0**2-kxm**2-kym**2);
        pupil = np.exp(1j*zfocus*np.real(kzm))*np.exp(-np.abs(zfocus)*np.abs(np.imag(kzm)));
        return CTF*pupil;

    objectRecoverFT = fftshift(fft2(objectRecover))  # shifted transform
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))
        fig.show()
    # Steps 2-5
    factor = (lrsize/hrshape[0])**2

    im_cmp = Image.fromarray(samples[(15, 15)])
    im_cmp = im_cmp.resize(hrshape)
    im_cmp = np.array(im_cmp)
    im_cmp = im_cmp.astype('float64')
    for iteration in range(5):
        objectRecoverFT = fftshift(fft2(objectRecover))  # shifted transform

        zfocus = 0
        xoff = 0
        yoff = 0.
        pupil = pupil_wrap(zfocus, pupil_radius)
        if im_out is not None:
            im_out /= np.amax(im_out)
            im_cmp /= np.amax(im_cmp)
            print('ssim %.2f' % ssim(im_cmp, im_out))
        for iteration in range(5):
            iterator = ct.set_iterator(cfg)
            print('Iteration n. %d' % iteration)
            # Patching for testing
            for it in iterator:
                acqpars = it['acqpars']
                indexes, kx_rel, ky_rel = ct.n_to_krels(it, cfg, xoff, yoff)
                # if indexes[0] > 19 or indexes[0] < 11 or indexes[1] > 19 or indexes[1] < 11:
                #      continue
                lr_sample = samples[it['indexes']]
                # From generate_il
                # Calculating coordinates
                [kx, ky] = kdsc*kx_rel, kdsc*ky_rel
                kyl = int(np.round(yc+ky-(lrsize+1)/2))
                kyh = kyl + lrsize
                kxl = int(np.round(xc+kx-(lrsize+1)/2))
                kxh = kxl + lrsize

                # Il = generate_il(im_array, f_ih, theta, phi, cfg)
                lowResFT = factor * objectRecoverFT[kyl:kyh, kxl:kxh]*pupil
                # Step 2: lr of the estimated image using the known pupil
                im_lowRes = ifft2(ifftshift(lowResFT))  # space pupil * fourier image
                im_lowRes = 1/factor * lr_sample * np.exp(1j*np.angle(im_lowRes))
                lowResFT = fftshift(fft2(im_lowRes))*pupil
                objectRecoverFT[kyl:kyh, kxl:kxh] = (1-pupil)*objectRecoverFT[kyl:kyh, kxl:kxh] + lowResFT
                # Step 3: spectral pupil area replacement
                ####################################################################
        im_out = np.abs(ifft2(ifftshift(objectRecoverFT)))
        if debug:
            fft_rec = np.log10(np.abs(objectRecoverFT))
            # fft_rec *= (255.0/fft_rec.max())
            # Il = Image.fromarray(np.uint8(Il), 'L')
            # im_rec *= (255.0/im_rec.max())
            def plot_image(ax, image, title):
                ax.cla()
                ax.imshow(image, cmap=plt.get_cmap('gray'))
                ax.set_title(title)
            axiter = iter([(ax1, 'Reconstructed FFT'), (ax2, 'Reconstructed magnitude'),
                        (ax3, 'Acquired image'), (ax4, 'Reconstructed phase')])
            for image in [np.abs(fft_rec), im_out, im_cmp, np.angle(ifft2(ifftshift(objectRecoverFT)))]:
                ax, title = next(axiter)
                plot_image(ax, image, title)
            fig.canvas.draw()
            # print("Testing quality metric", fpmm.quality_metric(samples, Il, cfg))
    return np.abs(im_out), np.angle(im_out)

def fpm_reconstruct_classic(samples=None, it=None, cfg=None,  debug=False):
    """ FPM reconstructon using the alternating projections algorithm. Here
    the complete samples and (optional) background images are loaded and Then
    cropped according to the patch size set in the configuration tuple (cfg).

    Args:
    -----
        samples: the acquired samples as a dictionary with angles as keys.
        backgrounds: the acquired background as a dictionary with angles as
                     keys. They must be acquired right after or before taking
                     the samples.
        it: iterator with additional sampling information for each sample.
        init_point: [xoff, yoff] center of the patch to be reconstructed.
        cfg: configuration (named tuple)
        debug: set it to 'True' if you want to see the reconstruction proccess
               (it slows down the reconstruction).
    Returns:
    --------
        (ndarray) The reconstructed modulus and phase of the sampled image.
    """
    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation

    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    lrsize = fpmm.get_image_size(cfg)
    pupil_radius = fpmm.get_pupil_radius(cfg)
    ps = fpmm.get_pixel_size(cfg)
    wlen = fpmm.get_wavelength(cfg)
    hrshape = fpmm.get_reconstructed_shape(cfg)
    # k relative to absolute k factor
    npx = fpmm.get_image_size(cfg)
    ps_req = fpmm.get_pixel_size_required(cfg)
    wlen = fpmm.get_wavelength(cfg)
    led_gap = float(cfg.led_gap)
    height = float(cfg.sample_height)

    objectRecover_mag = (samples[(0, -1)])/np.amax(samples[(0, -1)])
    objectRecover_phase = (samples[(0, -1)]-samples[(-1, 0)])/(samples[(0 ,-1)]+samples[(-1, 0)])
    objectRecover_phase = np.pi*(objectRecover_phase)/np.amax(objectRecover_phase)
    objectRecover = objectRecover_mag*np.exp(1j*objectRecover_phase)
    objectRecover = np.ones(hrshape)
    # plt.imshow(objectRecover_phase)
    # plt.show()
    center = fpmm.image_center(hrshape)
    CTF = fpmm.generate_CTF(image_size=[lrsize, lrsize], pupil_radius=pupil_radius)*.1
    # pupil = fpmm.aberrated_pupil(image_size=[lrsize, lrsize], pupil_radius=pupil_radius,
    #                             aberrations=[0,], pixel_size=ps, wavelength=wlen)
    objectRecoverFT = fftshift(fft2(objectRecover))  # shifted transform
    # Steps 2-5
    factor = (lrsize/hrshape[0])**2
    # For the convergence index
    # Steps 2-5
    for iteration in range(10):
        iterator = ct.set_iterator(cfg)
        # print('Iteration n. %d' % iteration)
        for it in iterator:
            # indexes, kx_rel, ky_rel = ct.n_to_krels(it=it, cfg=cfg, xoff=0, yoff=0)
            indexes = it['indexes']
            lr_sample = np.copy(samples[it['indexes']])
            led_number = np.array([it['ny'], it['nx']])
            # Convert indexes to pupil center coordinates kx, ky
            mc = np.array(cfg.mat_center) #  Matrix center
            k_rel = np.sin(np.arctan((led_number-mc)*led_gap/height))
            k_abs = k_rel*ps_req*npx/wlen
            pup_center = k_abs+center
            # Pupil borders
            lp = np.round(pup_center-(lrsize+1)/2).astype(int) # Extreme left low point
            hp = lp + lrsize # Extreme right high point
            pupil_square = [slice(lp[0], hp[0]), slice(lp[1], hp[1])]

            lowResFT = factor * objectRecoverFT[pupil_square]*CTF
            im_lowRes = ifft2(ifftshift(lowResFT))
            im_lowRes = (1/factor) * lr_sample *np.exp(1j*np.angle(im_lowRes))
            lowResFT = fftshift(fft2(im_lowRes))*CTF

            objectRecoverFT[pupil_square] = (1-CTF)*objectRecoverFT[pupil_square]+lowResFT
    im_out = ifft2(fftshift(objectRecoverFT))
    return im_out

# def fpm_reconstruct(samples=None, backgrounds=None, it=None, init_point=None,
#                     cfg=None,  debug=False):
#     """ FPM reconstructon using the alternating projections algorithm. Here
#     the complete samples and (optional) background images are loaded and Then
#     cropped according to the patch size set in the configuration tuple (cfg).
#
#     Args:
#     -----
#         samples: the acquired samples as a dictionary with angles as keys.
#         backgrounds: the acquired background as a dictionary with angles as
#                      keys. They must be acquired right after or before taking
#                      the samples.
#         it: iterator with additional sampling information for each sample.
#         init_point: [xoff, yoff] center of the patch to be reconstructed.
#         cfg: configuration (named tuple)
#         debug: set it to 'True' if you want to see the reconstruction proccess
#                (it slows down the reconstruction).
#
#     Returns:
#     --------
#         (ndarray) The reconstructed modulus and phase of the sampled image.
#     """
#     xoff, yoff = init_point  # Selection of the image patch
#     ps_required = fpmm.ps_required(cfg.phi[1], cfg.wavelength, cfg.na)
#     # Getting the maximum angle by the given configuration
#     # Step 1: initial estimation
#     Et = initialize(samples, backgrounds, xoff, yoff, cfg, 'zero')
#     f_ih = fft2(Et)  # unshifted transform, shift is later applied to the pupil
#     if debug:
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))
#         fig.show()
#     # Steps 2-5
#     samples = preprocess_images(samples, backgrounds, xoff, yoff, cfg,
#                                 corr_mode='bypass')
#     for iteration in range(cfg.n_iter):
#         iterator = ct.set_iterator(cfg)
#         print('Iteration n. %d' % iteration)
#         # Patching for testing
#         for it in iterator:
#             index, theta, phi = it['index'], it['theta'], it['phi']
#             print(it['theta'], it['phi'], it['indexes'])
#             # theta, phi = ct.corrected_coordinates(theta=theta, shift=shift,
#             #                                       cfg=cfg)
#             # Final step: squared inverse fft for visualization
#             im_array = fpmm.crop_image(samples[it['indexes']],
#                                        cfg.patch_size, xoff, yoff)
#             print(im_array.dtype)
#             # background = fpmm.crop_image(backgrounds[(theta, shift)],
#             #                              cfg.patch_size, xoff, yoff)
#             # im_array = image_correction(im_array, background, mode='background')
#             im_array, resc_size = image_rescaling(im_array, cfg)
#             Il = generate_il(im_array, f_ih, theta, phi, cfg)
#             pupil = fpmm.generate_pupil(theta=theta, phi=phi,
#                                         image_size=np.shape(im_array),
#                                         wavelength=cfg.wavelength,
#                                         pixel_size=ps_required,
#                                         na=cfg.objective_na)
#             pupil_shift = fftshift(pupil)  # Shifts pupil to match unshifted fft
#             f_il = fft2(Il)
#             f_ih = f_il*pupil_shift + f_ih*(1 - pupil_shift)
#             # If debug mode is on
#             if debug and index % 1 == 0:
#                 fft_rec = np.log10(np.abs(f_ih)+1)
#                 # fft_rec *= (255.0/fft_rec.max())
#                 fft_rec = fftshift(fft_rec)
#                 # Il = Image.fromarray(np.uint8(Il), 'L')
#                 im_rec = ifft2(f_ih)
#                 # im_rec *= (255.0/im_rec.max())
#                 def plot_image(ax, image, title):
#                     ax.cla()
#                     ax.imshow(image, cmap=plt.get_cmap('hot'))
#                     ax.set_title(title)
#                 axiter = iter([(ax1, 'Reconstructed FFT'), (ax2, 'Reconstructed magnitude'),
#                             (ax3, 'Acquired image'), (ax4, 'Reconstructed phase')])
#                 for image in [np.abs(fft_rec), np.abs(im_rec), im_array, np.angle(ifft2(f_ih))]:
#                     ax, title = next(axiter)
#                     plot_image(ax, image, title)
#                 fig.canvas.draw()
#             # print("Testing quality metric", fpmm.quality_metric(samples, Il, cfg))
#     return np.abs(np.power(ifft2(f_ih), 2)), np.angle(ifft2(f_ih+1))


def dpc_init(samples=None, backgrounds=None, it=None, init_point=None,
             cfg=None,  debug=False):
    """ Absolutely experimental reconstruction function. Virtually analog to
    fpm_reconstruct() to be used as a test sandbox.
    """
    # pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)
    xoff, yoff = init_point  # Selection of the image patch
    ps_required = fpmm.ps_required(cfg.phi[1], cfg.wavelength, cfg.na)
    # mask = get_mask(samples, backgrounds, xoff, yoff, cfg)
    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    Et = initialize(samples, backgrounds, xoff, yoff, cfg, 'zero')
    f_ih = fft2(Et)  # unshifted transform, shift is later applied to the pupil
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))
        fig.show()
        # fig, axes = implot.init_plot(4)
    # Steps 2-5
    for iteration in range(cfg.n_iter):
        iterator = ct.set_iterator(cfg)
        print('Iteration n. %d' % iteration)
        # Patching for testing
        for index, theta, shift in iterator:
            theta, phi = ct.corrected_coordinates(theta=theta, shift=shift,
                                                  cfg=cfg)
            print(theta, phi)
            # Final step: squared inverse fft for visualization
            im_array = fpmm.crop_image(samples[(theta, shift)],
                                       cfg.patch_size, xoff, yoff)
            background = fpmm.crop_image(backgrounds[(theta, shift)],
                                         cfg.patch_size, xoff, yoff)

            im_array = image_correction(im_array, background, mode='background')
            im_array, resc_size = image_rescaling(im_array, cfg)
            Il = generate_il(im_array, f_ih, theta, phi, cfg)
            #     print("Testing quality metric", quality_metric(image_dict, Il, cfg), phi)
            pupil = fpmm.generate_pupil(theta=theta, phi=phi,
                                        image_size=resc_size, wavelength=cfg.wavelength,
                                        pixel_size=ps_required, na=cfg.objective_na)
            pupil_shift = fftshift(pupil)
            f_il = fft2(Il)
            f_ih = f_il*pupil_shift + f_ih*(1 - pupil_shift)
            if debug and index % 1 == 0:
                fft_rec = np.log10(np.abs(f_ih)+1)
                fft_rec *= (255.0/fft_rec.max())
                fft_rec = fftshift(fft_rec)
                # fft_rec = Image.fromarray(np.uint8(fft_rec*255), 'L')
                im_rec = ifft2(f_ih)
                # im_rec *= (255.0/im_rec.max())
                im_rec_abs = np.abs(im_rec*np.conj(im_rec))
                def plot_image(ax, image):
                    ax.cla()
                    ax.imshow(image, cmap=plt.get_cmap('hot'))
                ax = iter([ax1, ax2, ax3, ax4])
                for image in [np.abs(fft_rec), im_rec, im_array, np.angle(im_rec)]:
                    plot_image(ax.next(), image)
                fig.canvas.draw()
        # print("Testing quality metric", quality_metric(image_dict, Il, cfg, max_phi))
    return np.abs(np.power(ifft2(f_ih), 2)), np.angle(ifft2(f_ih+1))
