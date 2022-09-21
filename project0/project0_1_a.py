#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 09:58:11 2022
CSC 514 - Project 0 - Q1_a
@author: sherwyn_b
"""

import skimage
from skimage import io
import time



img1 = io.imread('grizzlypeakg.png', as_gray=True)  
img1 = skimage.img_as_ubyte(img1, force_copy=True)  # Make sure its in UByte fmt
img2 = img1.copy()  # create a deep copy of img1


"""
Old code running 2 for loops
"""
start_original = time.time()
(m1,n1) = img1.shape
for i in range(10):
    for i in range(m1):
        for j in range(n1):
            if img1[i,j] <= 10:
                img1[i,j] = 0
end_original = time.time()
print('Original code runtime: {}'.format(end_original-start_original))


"""
Modified Code using logical indexing
"""
start_modified = time.time()
for i in range(10):
    temp = img1 <= 10   # logical indexing
    img2[temp] = 0      # set elements that satisfy the condition

end_modified = time.time()
print('Modified code runtime: {}'.format(end_modified-start_modified))
print('{} speedup factor achieved'.format(
    (end_original-start_original)/
    (end_modified-start_modified)
    ))


io.imsave('old_code_gray.jpg', img1)
io.imsave('modified_code_gray.jpg', img2)
