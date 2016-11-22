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

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.optimize import fsolve
from PIL import Image
from scipy import misc


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


def iterlaser(pupil_radius=50, ns=0.5, phi_max=90,
              image_size=(480, 640), mode='simulation'):

    """ Constructs an iterator of pupil center positions.

        Keywords:
        pupil_radius    radius of the pupil in the Fourier plane, given by NA
        phi,theta       spherical angles in sexagesimal degrees
        ns              1-radial_overlap (radius segment overlap)
        phi_max         maximum phi for the acquisition
        image_size      size of the created pupil image
    """
    yield 0, 0, 0, 0
    index = 0
    cycle = 1
    r = 0
    theta = 0
    rmax_abs = image_size[1]  # Absolute maximum for pupil center
    rmax_iter = np.abs(rmax_abs*np.sin(phi_max*np.pi/180))
    while r < rmax_iter:
        # Iterator update
        r = 2*pupil_radius*ns*cycle
        # delta_theta = np.arctan(2 * pupil_radius * ns / r)
        delta_theta = 2*np.arctan(1/(2.*cycle))
        theta = theta + delta_theta
        if theta > 2*np.pi:  # cycle ended
            cycle = cycle + 1
            r = 2*pupil_radius*ns*cycle
            theta = 0
        phi = np.arcsin(r/rmax_abs)  # See how to get phi
        index = index + 1
        power = laser_power(theta, phi, mode)
        yield index, np.degrees(theta), np.degrees(phi), power


# def iterleds(theta_max=360, phi_max=80, theta_step=10, mode='simulation'):
#     """ Constructs an iterator of pupil center positions in the correct order
#         for de LEDs sweep.
#
#         Keywords:
#         theta_max   radius of the pupil in the Fourier plane, given by NA
#         phi_max     spherical angles in sexagesimal degrees
#         theta_step  1-radial_overlap (radius segment overlap)
#     """
#     # yield 0, -80, 0, 0
#     theta_range = range(0, int(theta_max/2), theta_step)
#     index = 0
#     phi_step = 20  # Already defined by the geometry
#     phi_range = range(0, phi_max+1, phi_step)
#     for theta in theta_range:
#         for phi in phi_range:
#             index += 1
#             power = laser_power(theta, phi, mode)
#             yield index, theta, phi, power
#             if theta + 180 <= theta_max:
#                 index += 1
#                 power = laser_power(theta + 180, phi, mode)
#                 yield index, theta + 180, phi, power


def iterleds(theta_max=360, phi_max=80, theta_step=10, mode='simulation'):
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


def get_tilted_phi(theta=0, alpha=0, slice_ratio=1):
    """ Returns the phi angle considering a tilted plane of rotation.

    Parameters
    alpha   the tilt angle
    ypos    the y coordinate of the plane
    """
    # c is the R/a ratio related to the initial plane of the rotating leds
    c = slice_ratio  # aprox cos(phi_0)
    theta_rad = np.radians(theta)
    alpha = np.radians(alpha)

    def opt_func(phi):
        return np.tan(phi)-1/(np.cos(theta_rad)*np.tan(alpha)+c/(np.sin(phi)))
    phi_initial_guess = 1.5
    # Use the numerical solver to find the roots
    phi_solved = fsolve(opt_func, phi_initial_guess)[0]
    phi_solved = round(np.degrees(phi_solved), 2)
    return np.abs(phi_solved)


# def correct_angles(theta=0, phi=0):
#     """ Returns the phi angle considering a tilted plane of rotation.
#
#     Parameters
#     alpha   the tilt angle
#     ypos    the y coordinate of the plane
#     """
#     # c is the R/a ratio related to the initial plane of the rotating leds
#     if phi == -20:
#         phi = 17-abs(0.05*(abs(theta)-180))
#     if phi == 0:
#         phi = 7.37-abs(0.04*(abs(theta)-180))
#     if phi == 20:
#         theta = theta + 180
#         phi = 22.38-abs(0.048*(abs(theta)-180))
#     return theta, phi

def correct_angles(theta=0, phi=0):
    """ Returns the phi angle considering a tilted plane of rotation.

    Parameters
    alpha   the tilt angle
    ypos    the y coordinate of the plane
    """
    # c is the R/a ratio related to the initial plane of the rotating leds
    if phi == -20:
        phi = 17 - abs(0.05*(abs(theta) - 180))
    if phi == 0:
        phi = 7.37 - abs(0.04*(abs(theta) - 180))
    if phi == 20:
        theta = theta + 180
        phi = 22.38 - abs(0.048*(abs(theta) - 180))
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
    # Pupil positioning
    kx = np.floor(np.cos(theta_rad)*r)
    ky = np.floor(np.sin(theta_rad)*r)
    pup_pos = [image_center[0]+ky, image_center[1]+kx]
    # Put ones in a circle of radius defined in class
    coords = product(range(0, nx), range(0, ny))

    def dist(a, m):
        return np.linalg.norm(np.asarray(a)-m)
    c = list(ifilter(lambda a: dist(a, pup_pos) < pup_rad, coords))
    pup_matrix[[np.asarray(c)[:, 0], np.asarray(c)[:, 1]]] = 1
    return pup_matrix


def filter_by_pupil(image, theta, phi, power, pup_rad, image_size):
    pupil = generate_pupil(theta, phi, power, pup_rad, image_size)
    im_array = misc.imread(StringIO(image), 'RGB')
    print np.shape(im_array)
    f_ih = fft2(im_array)
    # Step 2: lr of the estimated image using the known pupil
    shifted_pupil = fftshift(pupil)
    proc_array = np.multiply(shifted_pupil, f_ih)  # space pupil * fourier im
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


def recontruct(input_file, iterator, debug=False, ax=None, data=None):
    """ FPM reconstructon from pupil images
    """
    image_dict = np.load(input_file)
    image_size = data['image_size']
    pupil_radius = data['pupil_radius']
    theta_max = data['theta_max']
    phi_max = data['phi_max']
    theta_step = data['theta_step']
    pupil_radius = 95
    ip = 440
    image_size = (480, 640)

    # image_size, iterator_list, pupil_radius, ns, phi_max = get_metadata(hf)
    # Step 1: initial estimation
    Ih_sq = 0.5 * np.ones(image_size)  # Constant amplitude
    Ih_sq = np.sqrt(image_dict[()][(10, 0)])
    Ph = np.ones_like(Ih_sq)  # and null phase
    Ih = Ih_sq * np.exp(1j*Ph)
    f_ih = fft2(Ih)  # unshifted transform, shift is applied to the pupil
    # Steps 2-5
    iterations_number = 2  # Total iterations on the reconstruction
    if debug:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # f, ax2 = plt.subplots(1, 1)
    for l in range(iterations_number):
        iterator = iterleds(theta_max, phi_max, theta_step, 'sample')
        print('iteration %d' % l)
        # Patching for testing
        for index, theta, phi, power in iterator:
            # Final step: squared inverse fft for visualization
            # im_array = hf[str(int(index))]
            im_array = image_dict[()][(theta, phi)]
            # im_array = im_array[220-ip/2:220+ip/2, 220-ip/2:220+ip/2]
            (theta_corr, phi_corr) = correct_angles(theta, phi)
            scaling_factor = 1.

            if phi == 20 or phi == -20:
                scaling_factor = 1.
            if index == 1 or phi == 20 or phi == 20 or phi == 40:
                print('passing')
                continue

            print('theta: %d, phi: %d, theta_corr=%f, phi_corr: %f'
                  % (theta, phi, theta_corr, phi_corr))
            print(power)
            im_array = im_array*scaling_factor
            pupil = generate_pupil(theta_corr, phi_corr, power, pupil_radius,
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
            # Step 3 ()cont.: Fourier space hr image update
            Il = Im_sq * Expl  # Spacial update
            f_il = fft2(Il)
            # Fourier update
            f_ih = f_il*pupil_shift + f_ih*(1 - pupil_shift)
            if debug:
                # fig_size = plt.rcParams["figure.figsize"]
                print("Debugging")
                im_rec = np.power(np.abs(ifft2(f_ih)), 2)
                ax1.get_figure().canvas.draw()
                ax2.get_figure().canvas.draw()
                ax3.get_figure().canvas.draw()
                ax4.get_figure().canvas.draw()
                ax1.imshow(pupil, cmap=plt.get_cmap('gray'))
                ax2.imshow(im_rec, cmap=plt.get_cmap('gray'))
                ax3.imshow(Im, cmap=plt.get_cmap('gray'))
                array = np.log10(np.abs(f_ih))
                array *= (1.0/array.max())
                array = fftshift(array)
                im = Image.fromarray(np.uint8(array*255), 'L')
                ax4.imshow(im, cmap=plt.get_cmap('gray'))
                # fig_size[0] = 20
                # fig_size[1] = 20

                plt.show(block=False)
                # f.set_size_inches(20, 20)

                # plt.rcParams["figure.figsize"] = fig_size



                # time.sleep(.5)
    return np.abs(np.power(ifft2(f_ih), 2))
