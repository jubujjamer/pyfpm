#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File optics_tools.py

Last update: 01/04/2017
Author: Juan Bujjamer

Tools for optical simulation and image processing.
Usage:

"""
from io import BytesIO
from itertools import ifilter, product, cycle
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


def annular_zernike(j, m, n, r):
    return np.sqrt(3)*(2*r**2 - 1)
