import time
import yaml

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from PIL import Image
from scipy import ndimage

from pyfpm.coordinates import PlatformCoordinates
import pyfpm.fpmmath as fpmm
import coordtrans as ct

# from . import implot
# import fpmmath.optics_tools as ot

def image_correction(image, background, mode='threshold'):
    if mode == 'threshold':
        image_corrected = image
    elif mode == 'background':
        image_corrected = image - 1.*background
        image_corrected -= np.min(image_corrected[:])-5
    # im_array = (im_array-.1*blank_array)
    # im_array -= np.min(im_array[:])
    # im_array[im_array < np.min(im_array)+2] = 0
    image_corrected /= np.max(image_corrected)
    return image_corrected*255

def image_rescaling(image, cfg):
    """ Image rescaling acording to sampling requirements.
    """
    phi_max = cfg.phi[1]
    wavelength = cfg.wavelength
    na = cfg.objective_na
    ps_required = fpmm.ps_required(cfg.phi[1], cfg.wavelength, cfg.na)
    scale_factor = cfg.pixel_size/ps_required
    Ih = ndimage.zoom(image, scale_factor, order=0) # HR image
    hr_shape = np.shape(Ih)
    return Ih, hr_shape


def initialize(samples, backgrounds=None, xoff=None, yoff=None, cfg=None,
               mode='zero'):
    """ Initialization of the algotithm
        zero: constant phase and zero amplitude.
        first: zero order image
    """
    image = fpmm.crop_image(samples[(0, 0)], cfg.patch_size, xoff, yoff)
    image, image_size = image_rescaling(image, cfg)
    if mode == 'zero':
        Ih_sq = 0.5 * np.ones_like(image)
        Ph = np.zeros_like(Ih_sq)  # and null phase
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
        Et = np.mean(corr_ims, axis=0)
        # Ph = 0.5+np.pi*np.abs(Et)/np.max(Et)
        Et = np.sqrt(Et) * np.exp(1j*0)
    #
    # plt.imshow(np.abs(Et))
    # plt.show()
    return Et


def generate_il(im_array, f_ih, theta, phi, cfg):
    ps = fpmm.ps_required(cfg.phi[1], cfg.wavelength, cfg.na)
    image_size = np.shape(im_array)
    pupil = fpmm.generate_pupil(theta=theta, phi=phi,
                                image_size=image_size, wavelength=cfg.wavelength,
                                pixel_size=ps, na=cfg.na)
    pupil_shift = fftshift(pupil)
    # Step 2: lr of the estimated image using the known pupil
    f_il = ifft2(f_ih*pupil_shift)  # space pupil * fourier image
    Phl = np.angle(f_il)
    # Step 3: spectral pupil area replacement
    Il = np.sqrt(im_array) * np.exp(1j*Phl)  # Spacial update
    Iupdate = ifft2(f_ih)*0.05 + Il
    Iupdate /= np.max(Iupdate)
    Iupdate *= 150
    return Iupdate

def fpm_reconstruct(samples=None, backgrounds=None, it=None, init_point = None, cfg=None,  debug=False):
    """ FPM reconstructon from pupil images

        Parameters:
            input_file:     the sample images dictionary
            blank_images:   images taken on the same positions as the sample images
    """
    pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)
    xoff, yoff = init_point  # Selection of the image patch
    ps_required = fpmm.ps_required(cfg.phi[1], cfg.wavelength, cfg.na)
    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    Et = initialize(samples, backgrounds, xoff, yoff, cfg, 'mean')
    f_ih = fft2(Et)  # unshifted transform, shift is later applied to the pupil
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))
        fig.show()
        # fig, axes = implot.init_plot(4)
    # Steps 2-5
    for iteration in range(cfg.n_iter):
        iterator = fpmm.set_iterator(cfg)
        print('Iteration n. %d' % iteration)
        # Patching for testing
        for index, theta, shift in iterator:
            theta, phi = ct.corrected_coordinates(theta=theta, shift=shift,
                                                  cfg=cfg)
            # print(theta, phi)
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
                fft_rec *= (1.0/fft_rec.max())
                fft_rec = fftshift(fft_rec)
                fft_rec = Image.fromarray(np.uint8(fft_rec*255), 'L')
                im_rec = np.power(np.abs(ifft2(f_ih)), 2)
                im_rec *= (1.0/im_rec.max())
                def plot_image(ax, image):
                    ax.cla()
                    ax.imshow(image, cmap=plt.get_cmap('hot'))
                ax = iter([ax1, ax2, ax3, ax4])
                for image in [pupil, im_rec, im_array, 10*np.angle(ifft2(f_ih+1))]:
                    plot_image(ax.next(), image)
                fig.canvas.draw()
                # image_plotter.update_plot([pupil, im_rec, Im, np.angle(ifft2(f_ih))])
                # implot.update_plot([pupil, im_rec, Im, np.angle(ifft2(f_ih))], fig, axes)
        # print("Testing quality metric", quality_metric(image_dict, Il, cfg, max_phi))
    return np.abs(np.power(ifft2(f_ih), 2)), np.angle(ifft2(f_ih+1))
