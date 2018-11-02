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

import pyfpm.data as dt

cfg = dt.load_config()


def image_center(image_size=None):
    """ Center coordinates given the image size.

    Args:
        image_size (list): list with the image sizes

    Returns:
        (int): integers with each dimension's mean size
    """
    if image_size is not None:
        yc, xc = np.array(image_size)/2
    return int(xc), int(yc)

def translate(value, input_min, input_max, output_min, output_max):
    """ Measuremente value rescaled by the selected span.

    Args:
        value (float): the value to be translated

    Returns:
        (float): the final linearly translated value
    """
    # Figure out how 'wide' each range is
    input_span = input_max - input_min
    output_span = output_max - output_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - input_min) / float(input_span)

    # Convert the 0-1 range into a value in the right range.
    return output_min + (value_scaled * output_span)

def get_acquisition_pars(theta=None, phi=None, shift=None, nx=None, ny=None, cfg=None):
    """ Returns illumination and camera acquisition parameters. It calculates
    them acording to the incident angles and illumination type (specified in cfg).

    Args:
        theta (float)
        phi (float)
        cfg (named tuple): configuration and calibration information

    Returns:
        (list) [iso, shutter_speed, led_power] acording to given position
    """
    # ss_rect_map = {(13, 13): 1E7, (13, 14): 1E7, (13, 15): 1E7, (13, 16): 1E7, (13, 17): 1E7,
    #                (14, 13): 1E7, (14, 14): 1E5, (14, 15): 1E5, (14, 16): 1E5, (14, 17): 1E7,
    #                (15, 13): 1E7, (15, 14): 1E5, (15, 15): 5E4, (15, 16): 1E5, (15, 17): 1E7,
    #                (16, 13): 1E7, (16, 14): 1E5, (16, 15): 1E5, (16, 16): 1E5, (16, 17): 1E7,
    #                (17, 13): 1E7, (17, 14): 1E7, (17, 15): 1E7, (17, 16): 1E7, (17, 17): 1E7}
    nmeans_dict = {(15, 15): 1,
(16, 15): 1,
(16, 16): 1,
(15, 16): 1,
(14, 16): 1,
(14, 15): 1,
(14, 14): 1,
(15, 14): 1,
(16, 14): 1,
(17, 14): 2,
(17, 15): 1,
(17, 16): 1,
(17, 17): 2,
(16, 17): 1,
(15, 17): 1,
(14, 17): 1,
(13, 17): 1,
(13, 16): 2,
(13, 15): 1,
(13, 14): 1,
(13, 13): 5,
(14, 13): 2,
(15, 13): 1,
(16, 13): 5,
(17, 13): 5,
(18, 13): 5,
(18, 14): 5,
(18, 15): 5,
(18, 16): 5,
(18, 17): 5,
(18, 18): 5,
(17, 18): 5,
(16, 18): 2,
(15, 18): 1,
(14, 18): 1,
(13, 18): 5,
(12, 18): 5,
(12, 17): 5,
(12, 16): 5,
(12, 15): 5,
(12, 14): 5,
(12, 13): 5,
(12, 12): 5,
(13, 12): 5,
(14, 12): 5,
(15, 12): 5,
(16, 12): 5,
(17, 12): 5,
(18, 12): 5,
(19, 12): 5,
(19, 13): 5,
(19, 14): 5,
(19, 15): 5,
(19, 16): 5,
(19, 17): 5,
(19, 18): 5,
(19, 19): 5,
(18, 19): 5,
(17, 19): 5,
(16, 19): 5,
(15, 19): 5,
(14, 19): 5,
(13, 19): 5,
(12, 19): 5,
(11, 19): 5,
(11, 18): 5,
(11, 17): 5,
(11, 16): 5,
(11, 15): 5,
(11, 14): 5,
(11, 13): 5,
(11, 12): 5,
(11, 11): 5,
(12, 11): 5,
(13, 11): 5,
(14, 11): 5,
(15, 11): 5,
(16, 11): 5,
(17, 11): 5,
(18, 11): 5,
(19, 11): 5}



    # led_center = 15
    # led_disp = (int(cfg.array_size)+1)//2
    # led_range = range(led_center-led_disp, led_center+led_disp)
    # ledmap = product(led_range, led_range)
    #
    # ss_dict = {}
    # for led in ledmap:
    #     # if led == [15, 15]:
    #     #     ss_dict[(led[0], led[1])] = 60E4
    #     # else:
    #     dist = (np.abs(led[0]-15)**2+np.abs(led[1]-15))
    #     ss = 5.E5*(1+.5*dist)
    #     ss_dict[(led[0], led[1])] = ss
    #     if ss >3E6:
    #         ss_dict[(led[0], led[1])] = 3E6

    power = 255
    # Camera parameters
    if nx is not None:
        try:
            # shutter_speed = ss_dict[nx, ny]
            shutter_speed = 500000
            nmeans = nmeans_dict[nx, ny]
        except:
            shutter_speed = 1E5
            nmeans = 1
        return float(cfg.iso), shutter_speed, power, nmeans

    shutter_speed_min = cfg.shutter_speed[0]
    shutter_speed_max = cfg.shutter_speed[0]
    if phi == None:
        if shift == None:
            raise Exception("Must assign a value either for phi or shift.")
        shutter_speed = translate(phi, 0, cfg.shift_max,
                                  shutter_speed_min, shutter_speed_max)
    else:
        shutter_speed = translate(phi, 0, 90,
                                  shutter_speed_min, shutter_speed_max)
    # Led parameters
    led_power = cfg.max_led_power
    return cfg.iso, shutter_speed, led_power, nmeans


def angles_to_k(theta, phi, kdsc):
    """ Returns kx, ky coordinates for the given theta and phi.

    Args:
        theta (float)
        phi (float)

    Returns:
        (list) [kx, ky] wave numbers
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    coords = np.array([np.sin(phi_rad)*np.cos(theta_rad),
                       np.sin(phi_rad)*np.sin(theta_rad)])
    #[kx, ky] = (1/wavelength)*coords*(pixel_size*npx)
    [kx, ky] = coords*kdsc

    return kx, ky

def set_iterator(cfg=None):
    shift_min, shift_max, shift_step = cfg.shift
    phi_min, phi_max, phi_step = cfg.phi
    theta_min, theta_max, theta_step = cfg.theta
    itertype = cfg.sweep

    if itertype == 'radial':
        """ The iterator is moved.
        """
        # yield 0, 0, 0, 0
        index = 0
        for phi in np.arange(phi_min, phi_max, phi_step):
            for theta in range(theta_min, theta_max, theta_step):
                if phi == 0 and index > 0:
                    continue
                power = 100
                yield index, theta, phi, power
                index += 1

    elif itertype == 'radial_efficient':
        """ Increments radius of the circle and alternates clockwise and
        anticlockwise movement when completing the circle.
        """
        # yield 0, 0, 0, 0
        index = 0
        direction_flag = 1
        phi_list = np.arange(phi_min, phi_max, phi_step)
        theta_list = list(range(theta_min, theta_max, theta_step))
        theta_list.extend(theta_list[-2:0:-1])
        theta_cycle = cycle(theta_list)
        phi_iter = iter(phi_list)
        ixx, iyy = [-1, -1]
        # list(enumerate(it.product(range(3), range(8))))

        for t in theta_cycle:
            ixx += 1
            if t == min(theta_list) or t == max(theta_list):
                p = next(phi_iter)
                iyy += 1
                ixx = 0
            try:
                acqpars = get_acquisition_pars(theta=t, phi=p, cfg=cfg)
            except:
                print('Old cfg had no acqpars.')
                acqpars = [0, 0, 0]
            yield {'index': index, 'theta': t, 'phi': p, 'acqpars': acqpars, 'indexes': (ixx, iyy)}
            # yield index, t, p, acqpars
            index += 1

    elif itertype == 'led_matrix':
        """ Iterates over led matrix.
        """
        asize = int(cfg.array_size)
        led_gap = float(cfg.led_gap)
        height = float(cfg.sample_height)

        xx = range(int(-(asize-1)/2), int((asize-1)/2)+1)
        zz = product(xx, xx)
        inditer = product(range(asize), range(asize))

        # x = np.arange(-(asize-1)/2+1, (asize-1)/2)
        # y = np.arange(-(asize-1)/2+1, (asize-1)/2)
        # xx, yy = np.meshgrid(x, y)
        # zz = -np.sin(np.atan(xx*led_gap(90))-np.sin(np.atan(xx*led_gap(90))
        # ziter = iter(zz.flaten)
        i = 0
        for x, y in zz:
            print(i)
            i += 1
            x *= led_gap
            y *= led_gap
            if x != 0:
                t = np.arctan(y/x)
                if x < 0:
                    t += np.pi
                if x > 0 and y < 0:
                    t = 2*np.pi + t
            else:
                t = np.pi/2 * np.sign(y)
            p = np.arctan(np.sqrt(x**2+y**2)/height)
            acqpars = get_acquisition_pars(theta=t, phi=p, cfg=cfg)
            print('x: %.1f y: %.1f theta: %.1f phi: %.1f' % (x, y, np.degrees(t), np.degrees(p)))

            yield {'indexes': next(inditer), 'theta': np.degrees(t), 'phi': np.degrees(p), 'acqpars': acqpars}
        # yield 0, 0, 0, 0

    elif itertype == 'led_matrix_rect':
        """ Iterates over led matrix in rectangular coordinates.
        """
        matsize = 32
        asize = int(cfg.array_size)
        xx = range(int(matsize/2-asize/2), int(matsize/2+asize/2))
        zz = product(xx, xx)


        for x, y in zz:
            acqpars = get_acquisition_pars(theta=0, phi=0, cfg=cfg)
            yield {'indexes': (x, y), 'nx': x, 'ny': y, 'acqpars': acqpars}


    elif itertype == 'led_matrix_ordered':
        """ Iterates over led matrix in rectangular coordinates, starting by the center
        and going through the square in spirals.
        """
        matsize = int(cfg.matsize)-1
        asize = int(cfg.array_size)
        lasti = (asize-1)/2
        ## The sequence will be +y, -x, -y, -x
        xc, yc = image_center([matsize, matsize])
        def out_dict(x, y):
            nx = xc+x
            ny = yc+y
            acqpars = get_acquisition_pars(nx=nx, ny=ny, cfg=cfg)
            return {'indexes': (int(nx), int(ny)), 'nx': nx, 'ny': ny, 'acqpars': acqpars}
        from random import shuffle

        yield out_dict(0, 0)
        for i in np.arange(1, lasti+1):
            increasing = np.arange(-i+1, i+1)
            decreasing= np.arange(i-1, -i-1, -1)
            # shuffle(increasing)
            # shuffle(decreasing)
            for y in increasing:
                yield out_dict(i, y)
            for x in decreasing:
                yield out_dict(x, i)
            for y in decreasing:
                yield out_dict(-i, y)
            for x in increasing:
                yield out_dict(x, -i)


    elif itertype == 'radial_efficient_shift':
        """ The same as radial_efficient but specifying shift in contrast to phi.
        """
        # yield 0, 0, 0, 0
        index = 0
        direction_flag = 1
        shift_list = np.arange(shift_min, shift_max, shift_step)
        theta_list = range(theta_min, theta_max, theta_step)
        theta_list.extend(theta_list[-2:0:-1])
        theta_list_max = max(theta_list)

        theta_cycle = cycle(theta_list)
        shift_iter = iter(shift_list)
        for t in theta_cycle:
            if t == min(theta_list) or t == max(theta_list):
                s = shift_iter.next()
            try:
                acqpars = get_acquisition_pars(theta=t, shift=s, cfg=cfg)
            except:
                print('Old cfg had no acqpars.')
                acqpars = [0, 0, 0]
            yield index, t, s, acqpars
            index += 1

def tidy(number):
    """ Rounding function to work under mechanical precission.
    """
    return np.around(number, 2)

def corrected_coordinates(theta=None, shift=None, phi=None, cfg=None):
    phi = np.degrees(np.arctan((1.*shift+1)/cfg.sample_height))
    return theta, phi

def phi_rot(od, phi):
    """ od rotation -phi angle- or along y axis
    """
    rot_mat = np.array([[np.cos(phi),  0, np.sin(phi)],
                        [0,            1,            0],
                        [-np.sin(phi), 0, np.cos(phi)]])
    return rot_mat.dot(od)


def theta_rot(od, theta):
    """ od rotation -theta angle- or along z axis
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta),  0],
                        [np.sin(theta),  np.cos(theta),  0],
                        [0,              0,              1]])
    return rot_mat.dot(od)


def rotate(prime_pos, theta, phi):
    """ Apply coordinate corrections as a 3d tranformation of the source
    """
    light_dir = np.array([0, 0, 1])
    origin_prime = np.array(prime_pos)
    light_dir = phi_rot(light_dir, phi)
    light_dir = theta_rot(light_dir, theta)
    # origin = phi_rot(origin_prime, phi)
    origin = theta_rot(origin_prime, theta)
    return origin, light_dir


def platform_to_cartesian(plat_coordinates, light_dir, source_center,
                          source_tilt, platform_tilt, height):
    """ Apply coordinate corrections as 3d tranformations of the source to
        convert platform coordinates to cartesian.

    Parameters:
    -----------
        source_position:    (array) source position in platform coordinates ([theta, shift]).
        light_dir:          (array) direction of light vector (default is [0, 0, 1]).
        source_center       (array) source center in cartesian coordinates (default is [0, 0])
                                    projected in the moving platform.
        source_tilt         (array) two parameters defining light source rotation ([theta, phi])
        platform_tilt       (array) two parameters defining platform rotation ([theta, phi])
    """
    # Translatig platform coordinates to 2D cartesian (x, y) in platform
    theta_rad = np.radians(plat_coordinates[0])
    x_plat = plat_coordinates[1] * np.cos(theta_rad)
    y_plat = plat_coordinates[1] * np.sin(theta_rad)
    source_position = [x_plat, y_plat, 0]  # All rotations made in z=0 plane
                                           # later I add the height
    # First correction: origin translation
    source_position[0] += source_center[0]
    source_position[1] += source_center[1]
    new_source_position = np.array(source_position)

    st_phi = np.radians(source_tilt[1])
    st_theta = np.radians(source_tilt[0])
    pt_phi = np.radians(platform_tilt[1])
    pt_theta = np.radians(platform_tilt[0])

    # direction rotation whith source_tilt and platform_tilt
    light_dir = phi_rot(light_dir, st_phi + pt_phi)
    light_dir = theta_rot(light_dir, st_theta + pt_theta)
    # source vector rotation whith platform_tilt
    source_position = phi_rot(new_source_position, pt_phi)
    source_position = theta_rot(source_position, pt_theta)
    source_position[2] += height
    return tidy(source_position), tidy(light_dir)


def platform_to_spherical(plat_coordinates, light_dir, source_center,
                          source_tilt, platform_tilt, height):
    """ Apply coordinate corrections as 3d tranformations of the source to
        convert platform coordinates to spherical.

    Parameters:
    -----------
        source_position:    (array) source position in platform coordinates ([theta, shift]).
        light_dir:          (array) direction of light vector (default is [0, 0, 1]).
        source_center       (array) source center in cartesian coordinates (default is [0, 0])
                                    projected in the moving platform.
        source_tilt         (array) two parameters defining light source rotation ([theta, phi])
        platform_tilt       (array) two parameters defining platform rotation ([theta, phi])

    Output:
    -------
        (array) spherical coordinates of the source (theta, phi, rho)
    """
    source_pos, light_dir = platform_to_cartesian(plat_coordinates, light_dir,
                            source_center, source_tilt, platform_tilt, height)
    x, y, z = source_pos
    rho = np.linalg.norm(source_pos)
    theta = np.arctan(y/x)
    phi = np.arccos(z/rho)
    return tidy(np.degrees(theta)), tidy(np.degrees(phi)), tidy(rho)


def input_angles_to_platform(theta, phi, height):
    """ Calculates platform coordinates from input angles (theta, phi). The 'input'
        word means this angles are **not corrected**.

    Parameters:
    -----------
        theta:    (array) source position in platform coordinates ([theta, shift]).
        phi:          (array) direction of light vector (default is [0, 0, 1]).
        height       (array) source center in cartesian coordinates (default is [0, 0])

    Output:
    -------
        (array) platform coordinates of the source (theta, shift, height)
    """
    shift = tidy(height*np.tan(np.radians(phi)))
    return theta, shift


def light_dir_from_angles(theta, phi, source_tilt):
    """ Calculates uncorrected light direction in cartesian coordinates. in other
    words, phi gives the direction to the center of the sample as if the platform
    had no positioning errors .

    Parameters:
    -----------
        phi:     (float) azimuthal angle.
        theta:   (float) azimuthal angle.

    Output:
    -------
        (array) platform coordinates of the source (theta, shift, height)
    """
    theta += source_tilt[0]
    phi += source_tilt[1]
    rho = 1
    phi_rad = np.radians(-phi)
    theta_rad = np.radians(theta)
    x = np.sin(phi_rad)*np.cos(theta_rad)*rho
    y = np.sin(phi_rad)*np.sin(theta_rad)*rho
    z = rho*np.cos(phi_rad)
    return x, y, z

def calculate_spot_center(source_position, light_dir):
    """ Center of the light spot on the sample plane.

    Returns
        (int array): [xcenter, ycenter] of the spot on the sample plane.
    """
    x_spot = source_position[0] + light_dir[0]*source_position[2]/light_dir[2]
    y_spot = source_position[1] + light_dir[1]*source_position[2]/light_dir[2]
    return x_spot, y_spot, 0


def spot_image(source_position, light_dir, radius=20, color='r'):
    """ Image of the spot of light on the sample plane.

    Returns
        (int array): [xcenter, ycenter] of the spot on the sample plane.
    """
    color_dict = {'r': 0, 'g': 1, 'b': 2}
    xsize, ysize = [400, 400]
    spot_center = calculate_spot_center(source_position, light_dir)
    spot_center += np.asarray([xsize/2, ysize/2, 0])
    image = np.zeros((xsize, ysize))
    xx, yy = np.meshgrid(range(xsize), range(ysize))
    c = (xx-spot_center[0])**2+(yy-spot_center[1])**2
    image_gray = [c < radius**2]
    image[:, :] = image_gray[0]
    return image

def n_to_krels(it=None, xoff=0, yoff=0, nx=15, ny=15, cfg=None, led_gap=None, height=None):
    """

    Parameters:
    -----------
        it:     (dict) iterator output.
        cfg:   (named array) azimuthal angle.

    Output:
    -------
        theta: (tuple) .
    """
    if cfg is not None:
        led_gap = float(cfg.led_gap)
        height = float(cfg.sample_height)
    else:
        led_gap = led_gap
    if it is not None:
        nx, ny = it['nx'], it['ny']
        indexes = it['indexes']
    else:
        nx=nx
        ny=ny
        indexes = (nx, ny)
    offset = np.array([xoff, yoff]) # offset
    mat_center = np.array([15, 15])-offset
    kx, ky = nx-mat_center[0], ny-mat_center[1]
    kx_rel= -np.sin(np.arctan(kx*led_gap/height))
    ky_rel= -np.sin(np.arctan(ky*led_gap/height))

    return indexes, kx_rel, ky_rel
