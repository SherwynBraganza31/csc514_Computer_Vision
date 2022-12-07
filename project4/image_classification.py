# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 01:59:38 2022

@author: mange
@author: Sherwyn Braganza
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
import typing


path_list = [fn for fn in os.listdir("Archive/TrainingImages/TrainingImages/")]


def getImageSpace(folder_name: str)->np.ndarray:
    """
        Load all the train images from a folder and stack them together horizontally
        to create X matrix or image_space matrix

        :param folder_name: name of the parent folder
        :return: X or image_space matrix
    """
    image_space = np.zeros((128*128, 0))
    for i in range(128):
        img = io.imread(f"Archive/TrainingImages/TrainingImages/{folder_name}/UnProcessed/img_{i}.png")
        img = img.reshape((-1, 1))
        image_space = np.hstack((image_space,img))
        
    #print(image_space.shape)
    return image_space


def ComputeER(X: np.ndarray, singular_vals, thresh: float) -> tuple:
    """
        Computes the order of eigenvalues required to achieve a
        energy reconstruction specified by thresh.

        :param X: The image space
        :param singular_vals: list of singular values
        :param thresh: minimum energy threshold needed
        :return: (int, list) -> tuple of the minimum # of eigenvalues and the % recon achieved.
    """
    ratio_list = []
    frob_norm = (X * X).sum()
    ratio = 0
    for sigma in singular_vals:
        ratio += (sigma**2)/frob_norm
        ratio_list.append(ratio)
        if ratio >= thresh:
            break

    return len(ratio_list), ratio_list
    

def createManifold(image_list, typ):
    """

        :param typ: L for local manifold construction, G for global
        :return:
    """
    manifold = np.zeros((128*128, 0))
    for fn in image_list:
        X = getImageSpace(fn)
        if typ == "L":
            U, S, V = np.linalg.svd(X, full_matrices=False)
            k, recon_list = ComputeER(X, S, 0.9)
            manifold = np.hstack((manifold, U[:, 0:k]))
        else:
            manifold = np.hstack((manifold, X)) 
         
    return manifold


def plotEnergyRecovery(imagelist):
    subplot_nums = np.sqrt(len(imagelist))
    fig, ax = plt.subplots(subplot_nums, subplot_nums)


if __name__ == '__main__':
    mf = createManifold(path_list, "L")
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
