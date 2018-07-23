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
from scipy import misc
# import h5py

fig1, ax = plt.subplots(1, 1, figsize=(25, 15))
fig1.show()

colorArray = np.zeros((1900, 1900, 3))
colorArray[..., 0] = 1
colorArray[..., 1] = 1
colorArray[..., 2] = 0
ax.imshow(colorArray)

plt.show()
