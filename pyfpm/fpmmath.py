#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File devices.py

Last update: 28/10/2016

Usage:

"""
from io import BytesIO
from itertools import ifilter, product
from StringIO import StringIO
import time
import yaml
# import matplotlib
# matplotlib.use('gtkagg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.optimize import fsolve
from PIL import Image
from scipy import misc
import random

#
config_dict = yaml.load(open('config.yaml', 'r'))


def laser_power(theta, phi, mode='simulation'):
    """ Returns power 0-255 given the theta, phi coordinates
    """
    power = 255
    if mode is 'simulation':
        pass
    elif mode is 'sampling':
        if phi == 0:
            power = 50
        else:
            power = 255
    elif mode is 'calibration':
            power = 1
    return power

def set_iterator(cfg=None):
    wavelength = cfg.wavelength
    pixelsize = cfg.pixelsize  # See jupyter notebook
    image_size = cfg.video_size
    phi_min, phi_max, phi_step = cfg.phi
    theta_min, theta_max, theta_step = cfg.theta
    pupil_radius = cfg.pupil_size/2
    image_dict = {}
    mode = cfg.task
    itertype = cfg.sweep

    if itertype == 'neopixels':
            """ Constructs an iterator of pupil center positions in the correct order
                for de LEDs sweep.

                Keywords:
                theta_max   radius of the pupil in the Fourier plane, given by NA
                phi_max     spherical angles in sexagesimal degrees
                theta_step  1-radial_overlap (radius segment overlap)
            """
            # yield 0, -80, 0, 0
            theta_range = range(-180, 180, theta_step)
            index = 0
            phi_step = 20  # Already defined by the geometry
            phi_range = [-20, 0, 20, 40]
            for theta in theta_range:
                for phi in phi_range:
                    index += 1
                    power = laser_power(theta, phi, mode)
                    yield index, theta, phi, power

    elif itertype == 'radial':
            """ Constructs an iterator of pupil center positions.

                Keywords:
                pupil_radius    radius of the pupil in the Fourier plane, given by NA
                phi,theta       spherical angles in sexagesimal degrees
                ns              1-radial_overlap (radius segment overlap)
                phi_max         maximum phi for the acquisition
                image_size      size of the created pupil image
            """
            # yield 0, 0, 0, 0
            index = 0
            for phi in range(phi_min, phi_max, phi_step):
                for theta in range(theta_min, theta_max, theta_step):
                    if phi == 0 and index > 0:
                        continue
                    power = laser_power(theta, phi, mode)
                    yield index, theta, phi, power
                    index += 1


    # elif itertype == 'radial':
    #         """ Constructs an iterator of pupil center positions.
    #
    #             Keywords:
    #             pupil_radius    radius of the pupil in the Fourier plane, given by NA
    #             phi,theta       spherical angles in sexagesimal degrees
    #             ns              1-radial_overlap (radius segment overlap)
    #             phi_max         maximum phi for the acquisition
    #             image_size      size of the created pupil image
    #         """
    #         yield 0, 0, 0, 0
    #         ns = 0.5
    #         index = 0
    #         cycle = 1
    #         r = 0
    #         theta = 0
    #         rmax_abs = image_size[1]  # Absolute maximum for pupil center
    #         rmax_iter = np.abs(rmax_abs*np.sin(phi_max*np.pi/180))
    #         while r < rmax_iter:
    #             # Iterator update
    #             r = 2*pupil_radius*ns*cycle
    #             # delta_theta = np.arctan(2 * pupil_radius * ns / r)
    #             delta_theta = 2*np.arctan(1/(2.*cycle))
    #             theta = theta + delta_theta
    #             if theta > 2*np.pi:  # cycle ended
    #                 cycle = cycle + 1
    #                 r = 2*pupil_radius*ns*cycle
    #                 theta = 0
    #             phi = np.arcsin(r/rmax_abs)  # See how to get phi
    #             index = index + 1
    #             power = laser_power(theta, phi, mode)
    #             yield index, np.degrees(theta), np.degrees(phi), power


def correct_angles(theta=0, phi=0):
    """ Returns the phi angle considering a tilted plane of rotation.

    Parameters
    alpha   the tilt angle
    ypos    the y coordinate of the plane
    """
    # c is the R/a ratio related to the initial plane of the rotating leds
    theta = theta - 110

    if phi == -20:
        phi = 25 - abs(0.05*(abs(theta) - 180))
    if phi == 0:
        phi = 1
    if phi == 20:
        # theta = theta + 180
        phi = 25 - abs(0.048*(abs(theta) - 180))
    return theta, phi


def itertest(theta_max=180, phi_max=80, theta_step=10, mode='simulation'):
    """ Itetrator with a particular behavior used for testing purposes.

    e.g. Tilted plane acquisition for calibration correction
    """
    theta_range = range(0, theta_max, theta_step)
    phi_0 = np.radians([0, 20, 40])
    slice_ratios = [np.cos(phi_0)][0]
    index = 0
    tilt = 5
    # yield 0, -80, 0, 0

    for theta in theta_range:
        for c in slice_ratios:
            phi = get_tilted_phi(theta, tilt, c)
            power = laser_power(theta, phi, mode)
            index += 1
            yield index, theta, phi, power


def to_leds_coords(theta, phi):
    """ Normalize coordinates for compatibility of different types of sweeps.
    """
    # In this mode the leds are considered well distributed and the movement
    # spherically symmetrical (theta and theta + 180 are equivalent)
    theta_leds = theta
    phi_leds = phi
    if theta >= 180:
        theta_leds = theta - 180
        phi_leds = -phi
    return theta_leds, phi_leds


def generate_pupil(theta, phi, power, pup_rad, image_size):
    rmax = image_size[1]/2  # Half image size  each side
    phi_rad = np.radians(phi)
    # if phi_rad < 0:
    #     theta = theta + 180  # To make it compatible with the "leds" sweep
    theta_rad = np.radians(theta)
    pup_matrix = np.zeros(image_size, dtype=np.uint8)
    nx, ny = image_size
    image_center = (nx/2, ny/2)
    # CHECK
    # conversion from phi to r
    r = np.abs(rmax * np.sin(phi_rad))
    # r = np.abs(rmax * phi_rad/(np.pi/2))
    # Pupil positioning
    kx = np.floor(np.cos(theta_rad)*r)
    ky = np.floor(np.sin(theta_rad)*r)
    # Another way
    pup_pos = [image_center[0]+ky, image_center[1]+kx]
    xx, yy = np.meshgrid(range(ny), range(nx))
    c = (xx-pup_pos[1])**2+(yy-pup_pos[0])**2
    # image_gray1 = [c < (pup_rad*0.75)**2]
    # image_gray2 = [c < pup_rad**2]
    # image_gray = image_gray1[0]*0.75 + image_gray2[0]*0.25
    image_gray = [c < pup_rad**2][0]

    # This could be slow, see in coords
    # def dist(a, m):
    #     return np.linalg.norm(np.asarray(a)-m)
    # c = list(ifilter(lambda a: dist(a, pup_pos) < pup_rad, coords))
    # pup_matrix[[np.asarray(c)[:, 0], np.asarray(c)[:, 1]]] = 1
    return image_gray


def filter_by_pupil(im_array, theta, phi, power, cfg):
    image_size = cfg.video_size
    pupil_radius = cfg.pupil_size/2
    # phi = phi*random.uniform(0.95, 1.05)
    # im_array = np.abs(im_array)
    pupil = generate_pupil(theta, phi, power, pupil_radius, image_size)
    f_ih = fft2(im_array)

    # Step 2: lr of the estimated image using the known pupil
    shifted_pupil = fftshift(pupil)
    proc_array = shifted_pupil * f_ih  # space pupil * fourier im
    proc_array = ifft2(proc_array)
    proc_array = np.power(np.abs(proc_array), 2)
    return proc_array


def show_filtered_image(self, image, theta, phi, power, pup_rad):
    img = self.filter_by_pupil(image, theta, phi, power, pup_rad)
    img = Image.fromarray(np.uint8((proc_array)*255))
    with BytesIO() as output:
        img.save(output, 'png')
        proc_image = output.getvalue()
    return bytearray(proc_image)


def show_pupil(theta, phi, power, pup_rad):
    pup_matrix = generate_pupil(theta, phi, power, pup_rad)
    # Converts the image to 8 bit png and stores it into ram
    img = Image.fromarray(pup_matrix*255, 'L')
    with BytesIO() as output:
        img.save(output, 'png')
        pupil = output.getvalue()
    return pupil


def array_as_image(image_array):
    image_array *= (1.0/image_array.max())
    # Converts the image to 8 bit png and stores it into ram
    img = Image.fromarray(image_array*255, 'L')
    with BytesIO() as output:
        img.save(output, 'png')
        output = output.getvalue()
    return output


def show_image(imtype='original', image=None, theta=0, phi=0,
               power=0, pup_rad=0):
    arg_dict = {'pupil': show_pupil(theta, phi, power, pup_rad),
                'filtered': show_filtered_image(image, theta, phi, power, pup_rad)}
    return bytearray(arg_dict[imtype])


def resample_image(image_array, new_size):
    return np.resize(image_array, new_size)

def crop_image(im_array, image_size, osx, osy):
    return im_array[osx:(osx+image_size[0]), osy:(osy+image_size[1])]
    # return im_array[offset:100, 0:]


def recontruct(input_file, iterator, cfg=None, debug=False):
    """ FPM reconstructon from pupil images
    input_file: the image dictionary
    """
    image_dict = np.load(input_file)[()]
    image_size = cfg.video_size
    phi_min, phi_max, phi_step = cfg.phi
    theta_min, theta_max, theta_step = cfg.theta
    pupil_radius = cfg.pupil_size/2
    n_iter = cfg.n_iter
    print(cfg.objective_na, image_size[0], cfg.pixelsize, cfg.wavelength)
    pupil_radius = cfg.objective_na*image_size[0]*float(cfg.pixelsize)/float(cfg.wavelength)

    # image_size, iterator_list, pupil_radius, ns, phi_max = get_metadata(hf)
    # Step 1: initial estimation
    Ih_sq = 0.5 * np.ones(image_size)  # Constant amplitude
    # Ih_sq = np.sqrt(image_dict[(0, 0)])
    Ph = np.ones_like(Ih_sq)  # and null phase
    Ih = Ih_sq * np.exp(1j*Ph)
    f_ih = fft2(Ih)  # unshifted transform, shift is applied to the pupil
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))
        im1 = im2 = im3 = im4 = ax1.imshow(Ih_sq, cmap=plt.get_cmap('gray'))
        fig.show()
    # Steps 2-5
    for iteration in range(n_iter):
        iterator = set_iterator(cfg)
        print('Iteration n. %d' % iteration)
        # Patching for testing
        for index, theta, phi, power in iterator:
            # Final step: squared inverse fft for visualization
            im_array = image_dict[(theta, phi)]
            im_array = crop_image(image_dict[(theta, phi)], image_size, 250, 0)
            # phi = phi-cfg.servo_init
            # if phi == 0 and index > 1:
            #     continue
            if phi == 1:
                continue
            if phi > 18:
                 continue
            phi = phi*1
            print('theta: %d, phi: %d, power: %d' % (theta, phi, power))
            pupil = generate_pupil(theta, phi, power, pupil_radius,
                                   image_size)
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
            f_il = fft2(Il)
            # Fourier update
            f_ih = f_il*pupil_shift + f_ih*(1 - pupil_shift)

            if debug:
                def plot_image(ax, image):
                    ax.cla()
                    ax.imshow(image, cmap=plt.get_cmap('gray'))
                print("Debugging")
                fft_rec = np.log10(np.abs(f_ih)+1)
                fft_rec *= (1.0/fft_rec.max())
                fft_rec = fftshift(fft_rec)
                fft_rec = Image.fromarray(np.uint8(fft_rec*255), 'L')
                im_rec = np.power(np.abs(ifft2(f_ih)), 2)
                plot_image(ax1, pupil)
                plot_image(ax2, im_rec)
                plot_image(ax3, Im)
                plot_image(ax4, np.angle(ifft2(f_ih)))
                fig.canvas.draw()
    return np.abs(np.power(ifft2(f_ih), 2))

def preprocess(input_file, iterator, cfg=None, debug=False):
    """ FPM reconstructon from pupil images
    input_file: the image dictionary
    """
    fixed_dict = dict()
    image_dict = np.load(input_file)[()]
    image_size = cfg.video_size
    phi_min, phi_max, phi_step = cfg.phi
    theta_min, theta_max, theta_step = cfg.theta
    pupil_radius = cfg.objective_na*image_size[0]*float(cfg.pixelsize)/float(cfg.wavelength)

    # image_size, iterator_list, pupil_radius, ns, phi_max = get_metadata(hf)
    if debug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
        im1 = im2 = ax1.imshow(image_dict[0, 0], cmap=plt.get_cmap('gray'))
        fig.show()
    phi_available = range(17)
    mean_intensity = dict()
    for phi in phi_available:
        mean_intensity[phi] = 0
    for index, theta, phi, power in iterator:
        # Final step: squared inverse fft for visualization
        im_array = image_dict[(theta, phi)]
        im_array = crop_image(image_dict[(theta, phi)], image_size, 200)
        mean_intensity[phi] += np.mean(im_array)
        print('theta: %d, phi: %d, power: %d' % (theta, phi, power))

    iterator = set_iterator(cfg)
    for phi in phi_available:
        mean_intensity[phi] /= 36
    for index, theta, phi, power in iterator:
        im_array = image_dict[(theta, phi)]
        print(mean_intensity[phi], np.mean(im_array))
        # im_array = crop_image(image_dict[(theta, phi)], image_size, 200)
        fixed_dict[(theta, phi)] = im_array*mean_intensity[phi]/np.mean(im_array)
        if debug:
            def plot_image(ax, image):
                ax.cla()
                ax.imshow(image, cmap=plt.get_cmap('gray'))
            plot_image(ax1, im_array)
            plot_image(ax2, fixed_dict[(theta, phi)])
            fig.canvas.draw()

    return fixed_dict
