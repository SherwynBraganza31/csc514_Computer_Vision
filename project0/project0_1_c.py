#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:16:33 2022
CSC 514 - Project 0 - Q1_c
@author: sherwyn_b
"""

import skimage
from skimage import io
import time



img1 = io.imread('grizzlypeak.jpg', as_gray=False)
img1 = skimage.img_as_ubyte(img1, force_copy=True)
img2 = img1.copy()


start_original = time.time()
(m1,n1,k1) = img1.shape
for i in range(10):
    for i in range(m1):
        for j in range(n1):
            for k in range(k1):
                if img1[i,j,k] <= 10:
                    img1[i,j,k] = 0

end_original = time.time()
print('Original code runtime: {}'.format(end_original-start_original))

start_modified = time.time()

for i in range(10):
    temp = img1 <= 10
    img2[temp] = 0

end_modified = time.time()
print('Modified code runtime: {}'.format(end_modified-start_modified))
print('{} speedup factor achieved'.format(
    (end_original-start_original)/
    (end_modified-start_modified)
    ))


io.imsave('old_code_colored.jpg', img1)
io.imsave('modified_code_colored.jpg', img2)

