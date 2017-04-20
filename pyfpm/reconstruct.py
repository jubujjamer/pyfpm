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
from pyfpm.fpmmath import *

# from . import implot
# import fpmmath.optics_tools as ot

def generate_il(im_array, f_ih, theta, phi, power, image_size,
                wavelength, pixel_size, na):
    pupil = generate_pupil(theta, phi, power, image_size, wavelength,
                           pixel_size, na)
    pupil_shift = fftshift(pupil)
    # Step 2: lr of the estimated image using the known pupil
    f_il = ifft2(f_ih*pupil_shift)  # space pupil * fourier image
    Phl = np.angle(f_il)
    Expl = np.exp(1j*Phl)
    # Step 3: spectral pupil area replacement
    # OBS: OPTIMIZE
    Im = np.resize(im_array, image_size)
    Im_sq = np.sqrt(Im)
    # Il_sq update in the pupil area using Im_sq
    # Step 3 (cont.): Fourier space hr image update
    Il = Im_sq * Expl  # Spacial update
    return Il, Im

def fpm_reconstruct(input_file,  iterator, cfg=None, debug=False):
    """ FPM reconstructon from pupil images

        Parameters:
            input_file:     the sample images dictionary
            blank_images:   images taken on the same positions as the sample images
    """
    image_dict = np.load(input_file)[()]
    image_size = cfg.video_size
    n_iter = cfg.n_iter
    pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)

    # Getting the maximum angle by the given configuration
    # Step 1: initial estimation
    phi_max = cfg.phi[1]+cfg.phi_max_err
    wavelength = cfg.wavelength
    na = cfg.objective_na
    ps_required = pixel_size_required(phi_max, wavelength, na)
    scale_factor = cfg.pixel_size/ps_required
    Ih = ndimage.zoom(np.ones(image_size), scale_factor, order=0) # HR image
    hr_shape = np.shape(Ih)
    image_size = hr_shape

    # Ih_sq = 0.5 * np.ones(image_size)  # Constant amplitude
    Ih_sq = 0.5 * Ih
    # Ih_sq = np.sqrt(image_dict[(0, 0)])
    Ph = np.ones_like(Ih_sq)  # and null phase
    Ih = Ih_sq * np.exp(1j*Ph)
    f_ih = fft2(Ih)  # unshifted transform, shift is applied to the pupil
    print(n_iter)
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))
        fig.show()
        # fig, axes = implot.init_plot(4)
    # Steps 2-5
    for iteration in range(n_iter):
        iterator = set_iterator(cfg)
        print('Iteration n. %d' % iteration)
        # Patching for testing
        for index, theta, phi in iterator:
            # Final step: squared inverse fft for visualization
            pc.platform_tilt = [0, 2]
            pc.source_center = [0, 2]  # [xcoord, ycoord] of the calibrated center
            im_array = image_dict[(theta, phi)]
            pc.set_coordinates(theta, phi, units='degrees')
            [theta_plat, phi_plat, shift_plat, power] = pc.parameters_to_platform()
            ss = adjust_shutter_speed(theta, phi)
            t_corr, p_corr = pc.source_coordinates(mode='angular')
            theta = t_corr + 45
            phi = p_corr
            # print("Mean intensity", np.mean(im_array), phi)
            # im_array = (im_array-.1*blank_array)
            # im_array -= np.min(im_array[:])
            # im_array[im_array < np.min(im_array)+2] = 0
            ss = adjust_shutter_speed(theta, phi)
            im_array = crop_image(im_array, cfg.video_size, 170, 245)*255./(power*ss**2)
            im_array = ndimage.zoom(im_array, scale_factor, order=0)

            #
            Il, Im = generate_il(im_array, f_ih, theta, phi, power, image_size,
                                cfg.wavelength, ps_required, cfg.objective_na)
            # for phi in np.arange(phi-5,phi+5,1):
            #     Il, Im = generate_il(im_array, f_ih, theta, phi, power, pupil_radius,
            #                        image_size)
            #     print("Testing quality metric", quality_metric(image_dict, Il, cfg), phi)
            pupil = generate_pupil(theta, phi, power, image_size, cfg.wavelength,
                                   ps_required, cfg.objective_na)
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
                    ax.imshow(image, cmap=plt.get_cmap('gray'))
                ax = iter([ax1, ax2, ax3, ax4])
                for image in [pupil, im_rec, Im, np.angle(ifft2(f_ih+1))]:
                    plot_image(ax.next(), image)
                fig.canvas.draw()
                # image_plotter.update_plot([pupil, im_rec, Im, np.angle(ifft2(f_ih))])
                # implot.update_plot([pupil, im_rec, Im, np.angle(ifft2(f_ih))], fig, axes)
        # print("Testing quality metric", quality_metric(image_dict, Il, cfg, max_phi))
    return np.abs(np.power(ifft2(f_ih), 2))
