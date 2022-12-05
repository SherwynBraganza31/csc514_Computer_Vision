# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 01:59:38 2022

@author: mange
"""

import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from typing import List
import cv2
from skimage import io
import matplotlib


image_list = [fn for fn in os.listdir("Archive/TrainingImages/TrainingImages/")]

def getImageSpace( folder_name ):
    image_space = np.zeros((128*128, 0))
    for i in range(128):
        img = io.imread(f"Archive/TrainingImages/TrainingImages/{folder_name}/UnProcessed/img_{i}.png")
        img = img.reshape((128*128, 1))
        image_space = np.hstack((image_space,img))
        
    #print(image_space.shape)
    return image_space


def ComputeER( X, u, val) :
    ratio_list = []
    de = (X * X).sum()
    ratio = 0
    for i in u:
        ratio += (i*i)/de
        ratio_list.append(ratio)
        if ratio >= val :
            return (X.shape[0], len(ratio_list)), ratio_list
        
    return (X.shape[0], len(ratio_list)), ratio_list
    

def createManifold( typ ):
    manifold = np.zeros((128*128, 0))
    for fn in image_list:
        X = getImageSpace(fn)
        if typ == "L" :
            U, S, V = np.linalg.svd(X, full_matrices=False)
            sh, li = ComputeER(X, S, 0.9)
            manifold = np.hstack((manifold, U[0:sh[0], 0:sh[1]])) 
        else :
            manifold = np.hstack((manifold, X)) 
         
    return manifold
    

if __name__ == '__main__':
    mf = createManifold( "G" )
    print(mf.shape)
    U, S, V = np.linalg.svd(mf, full_matrices=False)
    for i in range(10):
        eimg = U[:, i].reshape((128,128))
        plt.imshow(eimg, cmap="gray")
        plt.show()
    
'''
img_space = getImageSpace("Cup64")


avg = np.mean(img_space, axis=0)
#img_space = img_space - avg
#print(avg)
#img_space = np.outer(img_space, img_space.T) / 128
U, S, V = np.linalg.svd(img_space, full_matrices=False)
print(U.shape)
#w, v = np.linalg.eig(U)
#print(V.shape)

sh, li = ComputeER(img_space, S, 0.9)
img_space = img_space[0:sh[0],0:sh[1]]
print(img_space.shape)
#print(li)

print(S)
print( (S*S).sum())
for i in range(10):
    eimg = U[:, i].reshape((128,128))
    plt.imshow(eimg, cmap="gray")
    plt.show()
'''

#print(U.shape)
#print(V.shape)
#print(S.shape)
