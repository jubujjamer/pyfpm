#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File simulate.py

Last update: 28/10/2016
Use to locally simulate FPM.

Usage:

"""
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from pyfpm.fpmmath import *

print(dir())
print(image_center(image_size=[123, 130]))

pupil = generate_pupil(0, 90, 0, 50,[480, 640])
fig, ax = plt.subplots(1,1)
ax.imshow(pupil)
plt.show()
