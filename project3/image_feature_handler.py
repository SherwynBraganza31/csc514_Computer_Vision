import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import cv2


def loadImage(dir_name: str) -> List[np.ndarray]:
    """
        Grabs all the images from the directory 'dir_name' and returns a list of np.ndarrays
        that correspond to the image.

        Author: Sherwyn Braganza

        :param dir_name: The folder from which images have to be loaded
        :return: list of loaded images
    """
    image_array = []
    if os.path.exists(dir_name):
        os.chdir(dir_name)
        image_names = os.listdir()
        image_names.sort()
        for x in image_names:
            # check if the file listed is an image file, if it is, load else not.
            if '.jpg' in x.lower():
                image_array.append(skimage.io.imread(x))

        os.chdir("../../")
        return image_array
    else:
        print('Directory name doesn\'t exist')
        os.chdir("../../")



def grabUsrClicks(image1: np.ndarray, image2: np.ndarray) -> (np.ndarray, np.ndarray):
    """
        Horizontally concatenates both the images and plots them in a clickable window and gets user
        input when the user clicks inside the window.
        Times out if no clicks are received within 60 secs. Records only left clicks, right clicks undo
        the last left click. Mouse Button 3 prematurely ends the input capture.

        The aim of this function is to get sequential feature or interest point pairs from the user when two
        images are presented besides each other. Each click has to be followed by its corresponding feature
        pair in the other image.

        Author: Sherwyn Braganza

        :param image1: The first image
        :param image2: The second image
        :return: Tuple of lists image clicks in one image with the corresponding image clicks in the second.
    """
    joined_image = np.hstack((image1, image2))
    plt.imshow(joined_image)
    click_locs = plt.ginput(-1, timeout=30, show_clicks=True)
    image1_clicks, image2_clicks = [], []

    for x in click_locs:
        shape = image1.shape
        if x[0] >= shape[1]:
            # image2_clicks.append((shape[0] - x[1], x[0]%shape[1]))
            image2_clicks.append([x[0] % shape[1], x[1]])
        else:
            # image1_clicks.append((shape[0] - x[1], x[0]%shape[1]))
            image1_clicks.append([x[0] % shape[1], x[1]])

    image1_clicks = np.asarray(image1_clicks)
    image2_clicks = np.asarray(image2_clicks)
    plt.close()

    return image1_clicks, image2_clicks


def getFeatures(image1: np.ndarray, image2: np.ndarray):
    """
        Performs SIFT on the two images to get corresponding features.
        Sorts the found features based on distance and then returns the
        top 30 feature pairs.

        :param image1: First image
        :param image2: Second image
        :return: Two-Tuple of coordinates of corresponding features
    """
    sift = cv2.xfeatures2d.SIFT_create()
    image1 = cv2.cvtColor(skimage.img_as_ubyte(image1), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(skimage.img_as_ubyte(image2), cv2.COLOR_BGR2GRAY)
    features1, image1_descriptors = sift.detectAndCompute(image1, None)
    features2, image2_descriptors = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(image1_descriptors, image2_descriptors)
    matches = list(sorted(matches, key=lambda x: x.distance))[0:30]

    image1_features, image2_features = [], []

    for match in matches:
        image1_features.append(list(features1[match.queryIdx].pt)[-1::-1])
        image2_features.append(list(features2[match.trainIdx].pt)[-1::-1])

    return np.asarray(image1_features), np.asarray(image2_features)
