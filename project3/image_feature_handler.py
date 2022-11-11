import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


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
        for x in image_names:
            # check if the file listed is an image file, if it is, load else not.
            if '.jpg' in x.lower():
                image_array.append(skimage.io.imread(x))
        return image_array
    else:
        print('Directory name doesn\'t exist')


def grabUsrClicks(image1: np.ndarray, image2: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
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
    click_locs = plt.ginput(-1, timeout=90, show_clicks=True)
    image1_clicks, image2_clicks = [], []

    for x in click_locs:
        shape = image1.shape
        if x[0] >= shape[1]:
            # image2_clicks.append((shape[0] - x[1], x[0]%shape[1]))
            image2_clicks.append((x[0] % shape[1], x[1]))
        else:
            # image1_clicks.append((shape[0] - x[1], x[0]%shape[1]))
            image1_clicks.append((x[0] % shape[1], x[1]))

    plt.close()
    return image1_clicks, image2_clicks

