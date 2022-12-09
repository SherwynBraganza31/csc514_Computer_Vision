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
import math
from skimage import io
import json
import csv

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

    return image_space


def loadTestImage(folder_name: str, grab_entire=False) -> np.ndarray:
    """
        Load all the test from a folder and stack them together horizontally
        to create X matrix or test image space matrix

        :param folder_name: name of the parent folder
        :return: X or image_space matrix
    """
    image_space = np.zeros((128*128, 0))
    if grab_entire:
        for file in path_list:
            for i in range(64):
                img = skimage.io.imread(f"Archive/TestImages/TestImages/{file[0:-2]}32/UnProcessed/img_{i}.png",
                                        as_gray=True)
                img = img.reshape((-1, 1))
                image_space = np.hstack((image_space, img))
    else:
        for i in range(64):
            img = io.imread(f"Archive/TestImages/TestImages/{folder_name[0:-2]}32/UnProcessed/img_{i}.png")
            img = img.reshape((-1, 1))
            image_space = np.hstack((image_space, img))

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
    """
        Compute or create a manifold based on the equation

                    M = X.T * eig

        :param image_space: The image space matrix
        :return: The manifold created
    """
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


def findClass(manifolds: dict, eigen_vectors: dict, test_image: np.ndarray):
    """
        Calculates the expected squared error between the
        test image's manifold computed with the eigenvectors
        of the global manifold, and the global manifold.
        Finds the manifold point with the least expected square
        error and returns that as the closest associated class.

        :param manifolds: The dictionary of manifolds
        :param eigen_vectors: The dictionary of eigen vectors
        :param test_image: The test image data space.
        :return: The class to whic hthe image belongs to
    """
    test_projection = np.dot(test_image.T, eigen_vectors['global'])
    min_distance = float('inf')
    min_idx = 0

    for idx in range(manifolds['global'].shape[0]):
        temp_val = np.sum((test_projection - manifolds['global'][idx]) ** 2)
        if temp_val <= min_distance:
            min_distance = temp_val
            min_idx = idx

    return path_list[min_idx//128]


def findPose(manifolds: dict, eigen_vectors: dict, test_image: np.ndarray, image_class: str):
    """
        Calculates the expected square error between the test image's manifold
        generated with the eigenvectors of the local manifold, and the local manifold
        of image_class.
        It then calculates the weighted average angle between the two closest associated
        poses and returns the indices of the two closet poses along with the approximated
        angle.

        :param manifolds: The dictionary of manifolds
        :param eigen_vectors: The dictionary of eigen vectors
        :param test_image: The image space matrix of the test image
        :param image_class: The class to which the test image belongs to
        :return: 3-tuple of top two nearest poses and approximated angle
    """
    test_projection = np.dot(test_image.T, eigen_vectors[image_class])
    min_distance = float('inf')
    min_idx = 0

    for idx in range(manifolds[image_class].shape[0]):
        temp_val = np.sum((test_projection - manifolds[image_class][idx]) ** 2)
        if temp_val <= min_distance:
            min_distance = temp_val
            min_idx = idx

    # Best approx angle calculation by computing the weighted mean between found pose bounds.
    min_idx2 = min_idx - 1 if np.sum((test_projection - manifolds[image_class][min_idx - 1]) ** 2) >= \
                              np.sum((test_projection - manifolds[image_class][min_idx + 1]) ** 2) else \
                              min_idx + 1

    temp_sum = np.sum((test_projection - manifolds[image_class][min_idx2]) ** 2) + \
               np.sum((test_projection - manifolds[image_class][min_idx2]) ** 2)

    alpha = (2 * math.pi * min_idx / 128) + \
            (np.sum((test_projection - manifolds[image_class][min_idx - 1]) ** 2) / temp_sum) * \
            ((2 * math.pi * min_idx / 128) - (2 * math.pi * min_idx2 / 128))

    return min_idx, min_idx2, alpha


def plotEigVector(eig_vec, name: str, show=False):
    eig_vec = (eig_vec - eig_vec.min())/(eig_vec.max() - eig_vec.min())
    eig_vec = skimage.img_as_uint(eig_vec.reshape(128, 128))
    if show:
        plt.title(f'EigenVector of {name}')
        plt.imshow(eig_vec, cmap='gray')
        plt.show()
    else:
        skimage.io.imsave(f'eig_{name}.jpg', eig_vec)


def saveToJSON(data: dict, filename: str):
    for x in data.keys():
        data[x] = data[x].tolist()

    with open(f"{filename}.json", "w") as outfile:
        json.dump(data, outfile)

    for x in data.keys():
        data[x] = np.asarray(data[x])


def loadFromJSON(filename: str):
    with open(f'{filename}.json', 'r') as f:
        data = json.load(f)

    for x in data.keys():
        data[x] = np.asarray(data[x])

    return data


def plotImageSeries(test_image, left_image, right_image, pose, image_class):
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(left_image.reshape(128, 128), cmap='gray')
    ax[1].imshow(test_image.reshape(128, 128), cmap='gray')
    ax[1].set_title(f'Test Image : {np.round(pose,decimals=4)}')
    ax[2].imshow(right_image.reshape(128, 128), cmap='gray')

    fig.savefig(f'image_series_{image_class}', bbox_inches='tight')


def matchTestImage(test_image, manifolds, eigen_vectors):
    image_class = findClass(manifolds, eigen_vectors, test_image)
    idx, idx2, alpha = findPose(manifolds, eigen_vectors, test_image, image_class)

    imageSpace = getImageSpace(image_class)
    if idx < idx2:
        plotImageSeries(test_image, imageSpace[:, idx], imageSpace[:, idx2], alpha, image_class)
    else:
        plotImageSeries(test_image, imageSpace[:, idx2], imageSpace[:, idx], alpha, image_class)


if __name__ == '__main__':
    manifolds = {}
    eigen_vectors = {}
    test_images = loadTestImage("", grab_entire=True)

    if 'manifolds.json' in os.listdir() and 'eigen_vectors.json' in os.listdir():
        manifolds = loadFromJSON('manifolds')
        eigen_vectors = loadFromJSON('eigen_vectors')

    else:
        for obj in path_list:
            mani, eig = createManifold(getImageSpace(obj))
            manifolds.update({obj: mani})
            eigen_vectors.update({obj: eig})

        mani, eig = createManifold(getImageSpace("", grab_entire=True))
        manifolds.update({'global': mani})
        eigen_vectors.update({'global': eig})

        saveToJSON(manifolds, 'manifolds')
        saveToJSON(eigen_vectors, 'eigen_vectors')

    # plotEigVector(eigen_vectors['Boat64'][:, 0], 'Boat64')
    # plotEigVector(eigen_vectors['Keyboard64'][:, 0], 'Keyboard64')
    # plotEigVector(eigen_vectors['Cup64'][:, 0], 'Cup64')
    # plotEigVector(eigen_vectors['Cup64'][:, 1], 'Cup64_2')
    # plotEigVector(eigen_vectors['Cup64'][:, 2], 'Cup64_3')
    # plotEigVector(eigen_vectors['global'][:, 0], 'global_first')
    # plotEigVector(eigen_vectors['global'][:, 3], 'global_other')

    with open('Archive/TestImages/TestImages/RandAng.txt', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)[0]

    for i in range(5):
        rand_idx = int(np.random.randint(0, test_images.shape[1]))
        test_img = test_images[:, rand_idx]
        matchTestImage(test_img, manifolds, eigen_vectors)

        print(f'Image Class - {path_list[rand_idx//64][0:-2]} | Angle = {data[rand_idx%64]}')


