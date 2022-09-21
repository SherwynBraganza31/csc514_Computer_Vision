#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:27:20 2022
CSC 514 - Project 0 - Q2
@author: sherwyn_b
"""

import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

I = io.imread('gigi.jpg').astype(np.ubyte)
original = I.copy()
temp = I >= 50 # check for values >= 50
I[temp] = I[temp] - 50 # for those that meet the condn, lower by 50
I[temp != 1] = 0 # for those that don't, cap to 0

stitched = np.concatenate((original,I), axis=1)
plt.imshow(stitched)
plt.show()

io.imsave('gigi_compared.jpg',stitched)