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

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from PIL import Image
from scipy import misc

from pyfpm.data import get_metadata


def laser_power(theta, phi):
    """ Returns power 0-255 given the theta, phi coordinates
    """
    return 255


def iter_laser(pupil_radius=50, ns=0.5, phi_max=90, image_size=(480, 640)):
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
        power = laser_power(theta, phi)
        yield index, np.degrees(theta), np.degrees(phi), power


def iter_leds(theta_max=180, phi_max=90, theta_step=10):
    """ Constructs an iterator of pupil center positions.

        Keywords:
        theta_max   radius of the pupil in the Fourier plane, given by NA
        phi_max     spherical angles in sexagesimal degrees
        theta_step  1-radial_overlap (radius segment overlap)
    """
    yield 0, -90, 0, 0
    index = 0
    phi_step = 20  # Already defined by the geometry
    phi_range = range(-phi_max, phi_max+1, phi_step)
    theta_range = range(0, theta_max, theta_step)
    for theta in theta_range:
        for phi in phi_range:
            index += 1
            power = laser_power(theta, phi)
            yield index, theta, phi, power


def normsort_iterator(iterator):
    """ Normalize coordinates for compatibility of different types of sweeps.
        It also selects non suitable (non repeated and equally spaced)
        coordinates for the reconstrution.
    """
    return True


def generate_pupil(theta, phi, power, pup_rad, image_size):
    rmax = image_size[1]/2  # Half image size  each side
    phi_rad = phi*np.pi/180
    if phi_rad < 0:
        theta = theta+180  # To make it compatible with the "leds" sweep
    theta_rad = theta*np.pi/180
    pup_matrix = np.zeros(image_size, dtype=np.uint8)
    nx, ny = image_size
    image_center = (nx/2,ny/2)
    # CHECK conversion from phi to r
    r = np.abs(rmax * np.sin(phi_rad))
    # Pupil positioning
    kx = np.floor(np.cos(theta_rad)*r)
    ky = np.floor(np.sin(theta_rad)*r)
    pup_pos = [image_center[0]+ky, image_center[1]+kx]
    # Put ones in a circle of radius defined in class
    coords = product(range(0,nx),range(0,ny))
    def dist(a,m):
        return np.linalg.norm(np.asarray(a)-m)
    c = list(ifilter(lambda a : dist(a, pup_pos) < pup_rad, coords))
    pup_matrix[[np.asarray(c)[:,0], np.asarray(c)[:,1]]] = 1
    return pup_matrix


def filter_by_pupil(image, theta, phi, power, pup_rad, image_size):
    pupil = generate_pupil(theta, phi, power, pup_rad, image_size)
    #pupil.astype(dtype=complex)
    im_array = misc.imread(StringIO(image),'RGB')
    print np.shape(im_array)
    f_ih = fft2(im_array)
    # Step 2: lr of the estimated image using the known pupil
    # f_il = ifft2(f_ih.*pupil_shift) # space pupil * fourier image
    shifted_pupil = fftshift(pupil)
    proc_array = np.multiply(shifted_pupil, f_ih)
    proc_array = ifft2(proc_array)
    #proc_array = shifted_pupil
    #proc_array = ifft2(f_ih)
    #proc_array *= (1.0/proc_array.max())
    # proc_array = proc_array *1.0/proc_array.max()
    proc_array = np.power(np.abs(proc_array),2)
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

def show_image(imtype = 'original', image = None, theta = 0, phi = 0, power = 0, pup_rad = 0):
    arg_dict = {'pupil'      : show_pupil(theta, phi, power, pup_rad),
                'filtered'   : show_filtered_image(image, theta, phi, power, pup_rad)}
    return bytearray(arg_dict[imtype])

def resample_image(image_array, new_size):
    return np.resize(image_array,new_size)

def recontruct(input_file, debug=False, ax=None):
    """ FPM reconstructon from pupil images
    """
    # image_dict: (theta, phi) -> (img, power)
    with h5py.File(input_file, "r") as hf:
        image_size, iterator_list, pupil_radius, ns, phi_max = get_metadata(hf)
        print pupil_radius
        pupil_radius = 80
        res_improvement = 1
        xsize_original = image_size[1]
        ysize_original = image_size[0]
        xsize = xsize_original * res_improvement
        ysize = ysize_original * res_improvement
        DATA_LENGTH = len(iterator_list)
        print(DATA_LENGTH)
        ## Step 1: initial estimation
        ## Ih = sqrt(Ih)*exp(i*ph)
        Ih_sq = 0.5 * np.ones(image_size) # Constant amplitude
        # Ih_sq = np.sqrt(hf[str(int(0))])
        Ph = np.ones_like(Ih_sq) # and null phase
        Ih = Ih_sq * np.exp(1j*Ph)
        f_ih = fft2(Ih) # unshifted transform, shift is afterwards applied to the pupil
        ## Steps 2-5
        iterations_number = 5 # Total number of iteration on the whole reconstruction
        if debug:
            f, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2)

        for l in range(iterations_number):
            for index, theta, phi, power in iterator_list:
                # Final step: squared inverse fft for visualization
                if phi > 20 or phi < -20 or index == 0 or index >30: # discarding noisy data
                    print("Discarding noisy data.")
                    continue
                print("i = %d, theta = %d, phi = %d"  % (index, theta, phi))
                im_array = hf[str(int(index))]
                pupil = generate_pupil(theta, phi, power, pupil_radius*res_improvement, image_size)
                pupil_shift = fftshift(pupil);
                # Step 2: lr of the estimated image using the known pupil
                f_il = ifft2(f_ih*pupil_shift) # space pupil * fourier image
                Il_sq = np.abs(f_il)
                Phl = np.angle(f_il)
                Expl = np.exp(1j*Phl)
                # Step 3: spectral pupil area replacement
                # OBS: OPTIMIZE
                # f_ih = fft2(im_array)
                Im = np.resize(im_array, image_size)
                Im_sq = np.sqrt(Im);
                # Il_sq update in the pupil area using Im_sq
                #Il_sq = Il_sq*(np.logical_not(pupil))+Im_sq*pupil;
                # Step 3 ()cont.: Fourier space hr image update
                Il = Im_sq * Expl; # spacial update
                f_il = fft2(Il);
                # f_ih = f_il * pupil_shift + f_ih * np.logical_not(pupil_shift)

                f_ih = f_il * pupil_shift + f_ih * (1-pupil_shift) # Fourier update

                if debug:
                    print("Debugging")
                    im_rec = np.power(np.abs(ifft2(f_ih)),2)
                    # ax = plt.gca() or plt.subplots(2)
                    # ax.imshow(pupil, cmap=plt.get_cmap('gray'))
                    # ax.imshow((np.abs(f_ih * (1-pupil_shift))), cmap=plt.get_cmap('gray'))
                    # ax.imshow(im_rec, cmap=plt.get_cmap('gray'))
                    ax1.get_figure().canvas.draw()
                    ax2.get_figure().canvas.draw()
                    ax3.get_figure().canvas.draw()
                    ax4.get_figure().canvas.draw()
                    ax1.imshow(pupil, cmap=plt.get_cmap('gray'))
                    ax2.imshow(im_rec, cmap=plt.get_cmap('gray'))
                    ax3.imshow(Im, cmap=plt.get_cmap('gray'))

                    array = np.abs(f_ih)
                    array *= (1.0/array.max())
                    array = fftshift(array)
                    im = Image.fromarray(np.uint8(array*255), 'L')
                    ax4.imshow(im, cmap=plt.get_cmap('gray'))
                    #plt.show()
                    plt.show(block=False)
                    time.sleep(.5)
                    #plt.close()

        return np.abs(np.power(ifft2(f_ih),2))

#
## class PupilFilter(object):
#     def __init__(self, image, size, overlap = 0.5, radius = 0):
#         # Iterating attributes
#         self.index    = 0
#         self.cycle    = 0 # number of cycles for the iteration
#         self.theta    = 0.
#         self.phi      = 0.
#         self.power    = 0.
#         self.end_flag = 0
#         # Fixed parameters
#         self.size     = size
#         self.image    = image
#         self.radius   = radius
#         self.overlap  = overlap
#         self.ns = 1 - self.overlap; # Non overlap factor
#         self.max_rad  = self.size[0]-self.radius
#         self.hr_image   = self.resample_image(4)
#         self.proc_image = self.filter_by_pupil()
#
#     def spectrum(self):
#         pupil = self.generate_pupil()
#         im_array = misc.imread(StringIO(self.image))
#         f_ih = fft2(im_array)
#         fshift = fftshift(f_ih)
#         spec = 20*np.log(np.abs(fshift))
#         spec *= (1.0/spec.max())
#         img = Image.fromarray(np.uint8((spec)*255))
#         with BytesIO() as output:
#              img.save(output, 'png')
#              spectrum = output.getvalue()
#         return spectrum
#
#     def filter_by_pupil(self):
#         pupil = self.generate_pupil()
#         pupil.astype(dtype=complex)
#         im_array = misc.imread(StringIO(self.image))
#         f_ih = fft2(im_array)
#         # Step 2: lr of the estimated image using the known pupil
#         # f_il = ifft2(f_ih.*pupil_shift) # space pupil * fourier image
#         shifted_pupil = fftshift(pupil)
#         proc_array = np.multiply(shifted_pupil, f_ih)
#         proc_array = ifft2(proc_array)
#
#         #proc_array = ifft2(f_ih)
#         proc_array *= (1.0/proc_array.max())
#         img = Image.fromarray(np.uint8((proc_array)*255))
#         with BytesIO() as output:
#              img.save(output, 'png')
#              proc_image = output.getvalue()
#         return proc_image
#
#     def step(self):
#         # Iterator update
#         r = 2*self.radius*self.ns*self.cycle
#         if r != 0:
#             del_theta = np.arctan(2*self.radius*self.ns/r)
#             self.theta = self.theta + del_theta
#         if self.theta > np.pi or  r == 0: # cycle ended
#             self.cycle = self.cycle + 1
#             r = 2*self.radius*self.ns*self.cycle
#             self.theta = 0
#         self.index = self.index + 1
#         if r > self.max_rad:
#             self.end_flag = 1
#             return
#
#     def generate_pupil(self):
#         pup_matrix = np.zeros(self.size, dtype=np.uint8)
#         nx, ny = self.size
#         image_center = (nx/2,ny/2)
#         r = 2*self.radius*self.ns*self.cycle;
#         # Pupil positioning
#         kx = np.floor(np.cos(self.theta)*r);
#         ky = np.floor(np.sin(self.theta)*r);
#         print kx, ky, r
#         pup_pos = [image_center[0]+ky, image_center[1]+kx];
#         # Put ones in a circle of radius defined in class
#         coords = product(range(0,nx),range(0,nx))
#         def dist(a,m):
#             return np.linalg.norm(np.asarray(a)-m)
#         c = list(ifilter(lambda a : dist(a, pup_pos) < self.radius, coords))
#         pup_matrix[[np.asarray(c)[:,0], np.asarray(c)[:,1]]] = 1
#         return pup_matrix
#
#     def resample_image(self, res_factor):
#         img = Image.open(StringIO(self.image))
#         output_size = (self.size[1]*res_factor, self.size[0]*res_factor)
#         img = img.resize(output_size, Image.ANTIALIAS)
#         with BytesIO() as output:
#              img.save(output, 'png')
#              return output.getvalue()
#
#     def show_pupil(self):
#         pup_matrix = self.generate_pupil()
#         # Converts the image to 8 bit png and stores it into ram
#         img = Image.fromarray(pup_matrix*255, 'L')
#         with BytesIO() as output:
#              img.save(output, 'png')
#              pupil = output.getvalue()
#         return pupil
#
#     def show_image(self, imtype = 'original'):
#         arg_dict = {'original'   : self.image,
#                     'hires'      : self.hr_image,
#                     'filtered'   : self.proc_image,
#                     'pupil'      : self.show_pupil()}
#         return bytearray(arg_dict[imtype])
#
#
#     def set_parameters(self, theta, phi, power):
#         self.index    = 0 # restarts the counter
#         self.theta    = 0.
#         self.phi      = 0.
#         self.power    = 0.
#
#     def get_theta(self):
#         return self.theta
#
#     def get_phi(self):
#         return self.phi
#
#     def get_power(self):
#         return self.power

# recontruct(load_from_folder('\kdkdkd'))
# recontruct(sim_iterator)
#
# def recontruct(data_it, debug=False):
#
#     # image_dict: (theta, phi) -> (img, power)
#
#     load('test_images.mat','test_images','pupil_step','pupil_diameter','iterator_array','res_factor','AQC_IMAGE_SIZE')
#
#     xsize_original = AQC_IMAGE_SIZE(2)
#     ysize_original = AQC_IMAGE_SIZE(1)
#     xsize = AQC_IMAGE_SIZE(2)/res_factor
#     ysize = AQC_IMAGE_SIZE(1)/res_factor
#     DATA_LENGTH = length(iterator_array)
#
#     ## Step 1: initial estimation
#     ## Ih = sqrt(Ih)*exp(i*ph)
#     Ih_sq = 0.5 * np.ones((ysize, xsize)) # Constant amplitude
#     Ph = np.ones_like(Ih_sq) # and null phase
#     Ih = Ih_sq * np.exp(1i*Ph)
#     f_ih = np.fft2(Ih) # unshifted transform, shift is afterwards applied to the pupil
#
#     ## Steps 2-5
#     END_IT = 1 # Total number of iteration on the whole reconstruction
#     #for l=1:END_IT
#     #tmp = imresize(test_images(:,:,4,4), 1/res_factor);
#     #original_resampled_RGB = zeros(size(tmp, 1), size(tmp, 2), 3);
#     #tmp(tmp<0) = 0;
#     #original_resampled_RGB(:, :, 1) = tmp / max(tmp(:));
#         i = 1
#         iterator = iterator_array(i,:)
#         for theta, phi, power, img in data_it:
#             pupil = build_pupil_image(image_size, theta, phi)## TODO
#
#             pupil_shift = fftshift(pupil);
#
#             # Step 2: lr of the estimated image using the known pupil
#             f_il = np.ifft2(f_ih.*pupil_shift) # space pupil * fourier image
#             Il_sq = np.abs(f_il)
#             Phl = np.angle(f_il)
#             Expl = np.exp(1i*Phl)
#
#             # Step 3: spectral pupil area replacement
#             # OBS: OPTIMIZE
#             Im = imresize(test_images(:,:,i), 1./res_factor, 'box') # oversampling image for the matrix product
#             Im_sq = np.sqrt(Im);
#
#             # Il_sq update in the pupil area using Im_sq
#             # Il_sq = Il_sq.*(not(pupil))+Im_sq.*pupil;
#
#             # Step 3 ()cont.: Fourier space hr image update
#             Il = Il_sq * Expl; % spacial update
#             f_il = np.fft2(Il);
#             f_ih = f_il * pupil_shift + f_ih * np.logical_not(pupil_shift) # Fourier update
#
#             # Final step: squared inverse fft for visualization
#             reconstructed_image = abs(ifft2(f_ih)).^2;
#
#             if debug:
#                 subplot(221), imagesc(reconstructed_image)
#                 subplot(222), imagesc(log10(abs(fftshift(f_ih))))
#                 subplot(223), imagesc(Im)
#                 subplot(224), imagesc(pupil)
#                 colormap gray
#                 pause(0.1)
#             i = i+1;
#         end
#     %end
