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
from skimage import io
import matplotlib


path_list = [fn for fn in os.listdir("Archive/TrainingImages/TrainingImages/")]


def getImageSpace(folder_name: str, grab_entire=False) -> np.ndarray:
    """
        Load all the train images from a folder and stack them together horizontally
        to create X matrix or image_space matrix

        :param folder_name: name of the parent folder
        :return: X or image_space matrix
    """
    image_space = np.zeros((128*128, 0))
    if grab_entire:
        for file in path_list:
            for i in range(128):
                img = skimage.io.imread(f"Archive/TrainingImages/TrainingImages/{file}/UnProcessed/img_{i}.png",
                                        as_gray=True)
                img = img.reshape((-1, 1))
                image_space = np.hstack((image_space, img))
    else:
        for i in range(128):
            img = io.imread(f"Archive/TrainingImages/TrainingImages/{folder_name}/UnProcessed/img_{i}.png")
            img = img.reshape((-1, 1))
            image_space = np.hstack((image_space, img))

    #print(image_space.shape)
    return image_space


def loadTestImage(folder_name: str, grab_entire=False) -> np.ndarray:
    """
        Load all the train images from a folder and stack them together horizontally
        to create X matrix or image_space matrix

        :param folder_name: name of the parent folder
        :return: X or image_space matrix
    """
    image_space = np.zeros((128*128, 0))
    if grab_entire:
        for file in path_list:
            for i in range(128):
                img = skimage.io.imread(f"Archive/TestImages/TestImages/{file[0:-2]}32/UnProcessed/img_{i}.png",
                                        as_gray=True)
                img = img.reshape((-1, 1))
                image_space = np.hstack((image_space, img))
    else:
        for i in range(128):
            img = io.imread(f"Archive/TestImages/TestImages/{folder_name[0:-2]}32/UnProcessed/img_{i}.png")
            img = img.reshape((-1, 1))
            image_space = np.hstack((image_space, img))

    #print(image_space.shape)
    return image_space


def computeER(X: np.ndarray, thresh: float) -> tuple:
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
    _, singular_vals, _ = np.linalg.svd(X, full_matrices=False)

    for sigma in singular_vals:
        ratio += (sigma**2)/frob_norm
        ratio_list.append(ratio)
        if ratio >= thresh:
            break

    return len(ratio_list), ratio_list


def createManifold(image_space):
    k, _ = computeER(image_space, 0.9)
    eigen_vectors = getBasisVectors(image_space, k)
    manifold = np.dot(image_space.T, eigen_vectors)

    return manifold, eigen_vectors


def getUnbiasedDataset(image_space: np.ndarray) -> np.ndarray:
    """
        Gets the unbiased image_space matrix from the biased image_space.
        Calculates the mean pixel value for each pixel and then subtracts it
        from the biased_image space

        :param image_space: The biased image space
        :return: the unbiased image space
    """
    mean_image = np.sum(image_space, axis=1)
    return image_space - mean_image


def getBasisVectors(image_space: np.ndarray, k: int) -> np.ndarray:
    """
        Gets the first 'k' eigenvectors of the imagespace
        :param image_space: The image space matrix
        :param k: The rank of eigenvectors to be chosen
        :return: k row matrix in which each row corresponds to an eigenvector
    """
    U, S, V = np.linalg.svd(image_space, full_matrices=False)

    return U[:, 0:k]

# TODO
def plotEnergyRecovery(imagelist):
    subplot_nums = np.sqrt(len(imagelist))
    fig, ax = plt.subplots(subplot_nums, subplot_nums)


def findClass(manifolds, eigen_vectors,  test_image):
    test_projection = np.dot(test_image.T, eigen_vectors['global'])
    min_distance = float('-inf')
    min_idx = 0

    for idx in range(manifolds['global'].shape[0]):
        temp_val = np.sum((test_projection - manifolds['global'][idx]) ** 2)
        if temp_val < min_distance:
            min_distance = temp_val
            min_idx = idx

    return path_list[min_idx//128]


def plotEigVector(eig_vec, name: str, show=False):
    eig_vec = (eig_vec - eig_vec.min())/(eig_vec.max() - eig_vec.min())
    eig_vec = skimage.img_as_uint(eig_vec.reshape(128, 128))
    if show:
        plt.title(f'EigenVector of {name}')
        plt.imshow(eig_vec, cmap='gray')
        plt.show()
    skimage.io.imsave(f'eig_{name}.jpg', eig_vec)


if __name__ == '__main__':
    manifolds = {}
    eigen_vectors = {}
    for obj in path_list:
        mani, eig = createManifold(getImageSpace(obj))
        manifolds.update({obj: mani})
        eigen_vectors.update({obj: eig})

    mani, eig = createManifold(getImageSpace("", grab_entire=True))
    manifolds.update({'global': mani})
    eigen_vectors.update({'global': eig})

    # plotEigVector(eigen_vectors['Boat64'][:, 0], 'Boat64')
    # plotEigVector(eigen_vectors['Keyboard64'][:, 0], 'Keyboard64')
    # plotEigVector(eigen_vectors['Cup64'][:, 0], 'Cup64')
    # plotEigVector(eigen_vectors['Cup64'][:, 1], 'Cup64_2')
    # plotEigVector(eigen_vectors['Cup64'][:, 2], 'Cup64_3')
    # plotEigVector(eigen_vectors['global'][:, 0], 'global_first')
    # plotEigVector(eigen_vectors['global'][:, 3], 'global_other')

    test_images = loadTestImage(path_list[4])
    obj = findClass(manifolds, eigen_vectors, test_images[6])

