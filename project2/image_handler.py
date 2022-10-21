import skimage.io
import skimage.color
import numpy as np
import os


def readImage(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        raise Exception('Image "{}" not found'.format(filename))

    return skimage.io.imread(filename, as_gray=False)


def convertToGrayscale(image: np.ndarray) -> np.ndarray:
    return skimage.color.rgb2gray(image)
