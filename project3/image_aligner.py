import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
from image_feature_handler import loadImage, grabUsrClicks
import cv2

def computeHomography(image1_features: np.ndarray, image2_features: np.ndarray) -> np.ndarray:
    """
        Calculates the H matrix or the homography for a given set of points based on the
        equation

            p_2 = H * p_1           where p_1 is the source vector and
                                    p_2 is the destination vector

        Performs a least squares fit on the points to get the best possible homography
        by picking the singular vector corresponding to the singular value that has the
        least magnitude.

        Author: Sherwyn Braganza

        :param image1_features: Feature points of image 1
        :param image2_features: Feature points of image 2
        :return: the homography matrix
    """
    p_matrix_rows = 2 * min([len(image1_features), len(image2_features)])
    point_matrix = np.zeros((p_matrix_rows, 9))
    for idx in range(int(p_matrix_rows/2)):
        x_d, y_d = image2_features[idx]
        x_s, y_s = image1_features[idx]
        point_matrix[2*idx, :] = np.asarray([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        point_matrix[2*idx+1, :] = np.asarray([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    U, S, V = np.linalg.svd(point_matrix)
    # print(np.sum(V[-1] ** 2))
    # h_matrix, status = cv2.findHomography(image1_features, image2_features)

    h_matrix = V[-1].reshape(3,3)
    return h_matrix


def computeProjection(h_matrix: np.ndarray, p_source: np.ndarray) -> np.ndarray:
    """
        Computes p_destination using the homography matrix based on the equation

                p_dest = h_matrix  x  p_source

        :param h_matrix: The homography
        :param p_source: The source point
        :return: The destinantion point
    """
    pt_projection = np.matmul(h_matrix, p_source)
    pt_projection = pt_projection / pt_projection[-1]

    return pt_projection


def inverseWarp(image1, image2, image1_features, image2_features):
    """
        Performs an inverse warp of image2 to the plane of image1.
        Computes the edge projections of image2 on the plane of image1
        and pads image1 accordingly.
        Then performs an inverse warp using interpolation

        Author: Sherwyn Braganza

        :param image1: The image that acts as the base plane
        :param image2: The image that is warped to the base plane
        :param image1_features: features in image1
        :param image2_features: features in image2
        :return: The warped image
    """
    h_matrix = computeHomography(image1_features, image2_features)
    h_inv = computeHomography(image2_features, image1_features)

    x_bound, y_bound = image2.shape[0], image2.shape[1]  # get size of the second image

    # Get positional values for the extremities of image2 in the image2 plane
    edge_pts = np.asarray([[0, 0, 1],  # TopLeft
                           [0, y_bound, 1],  # TopRight
                           [x_bound, y_bound, 1],  # BottomRight
                           [x_bound, 0, 1]]).T  # Bottom Left

    edge_projection = computeProjection(h_inv, edge_pts).round(decimals=0).astype('int32')

    bound_min = np.min(edge_projection, axis=1)[0:2]
    bound_max = np.max(edge_projection, axis=1)[0:2]

    # compute pad parameters and pad the image
    left_pad = -bound_min[1] if bound_min[1] < 0 else 0
    right_pad = bound_max[1] - y_bound if bound_max[1] > y_bound else 0
    top_pad = -bound_min[0] if bound_min[0] < 0 else 0
    bottom_pad = bound_max[0] - x_bound if bound_max[0] > x_bound else 0
    pad_params = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    image1 = np.pad(image1, pad_params, mode='constant', constant_values=0)

    for x in range(bound_min[0], bound_max[0]):
        for y in range(bound_min[1], bound_max[1]):
            projection = computeProjection(h_matrix,
                                           np.asarray([x, y, 1]).reshape(3, 1))
            rounded_proj = projection.round(decimals=0).astype('int32')
            trunc_proj = projection.astype('int32')

            if 1 <= projection[0] < x_bound - 1 and 1 <= projection[1] < y_bound - 1:
                # pix_val = interpPixels(image2, trunc_proj[0], trunc_proj[1])
                pix_val = image2[rounded_proj[0], rounded_proj[1], :]
                image1[x + top_pad, y + left_pad, :] = np.mean(pix_val, axis=0)

    return image1


def forwardWarp(image1, image2, image1_features, image2_features) -> np.ndarray:
    """
        Melds two images together by performing a forward warp from image2 to image1.
        Computes the edge projections of image2 on the plane of image1
        and pads image1 accordingly.

        Author: Sherwyn Braganza

        :param image1: The image that acts as the base plane
        :param image2: The image that is warped to the base plane
        :param image1_features: features in image1
        :param image2_features: features in image2
        :return: The warped image
    """
    h_matrix = computeHomography(image2_features, image1_features)
    x_bound, y_bound = image2.shape[0], image2.shape[1]  # get size of the second image

    # Get positional values for the extremities of image2 in the image2 plane
    edge_pts = np.asarray([[0, 0, 1],  # TopLeft
                [0, y_bound, 1],  # TopRight
                [x_bound, y_bound, 1],  # BottomRight
                [x_bound, 0, 1]]).T    # Bottom Left

    edge_projection = computeProjection(h_matrix, edge_pts).round(decimals=0).astype('int32')

    bound_min = np.min(edge_projection, axis=1)[0:2]
    bound_max = np.max(edge_projection, axis=1)[0:2]

    # compute pad parameters and pad the image
    left_pad = -bound_min[1] if bound_min[1] < 0 else 0
    right_pad = bound_max[1] - y_bound if bound_max[1] > y_bound else 0
    top_pad = -bound_min[0] if bound_min[0] < 0 else 0
    bottom_pad = bound_max[0] - x_bound if bound_max[0] > x_bound else 0
    pad_params = ((top_pad, bottom_pad), (left_pad, right_pad), (0,0))
    image1 = np.pad(image1, pad_params, mode='constant', constant_values=0)

    for x in range(image2.shape[0]):
        for y in range(image2.shape[1]):
            projection = computeProjection(h_matrix,
                                           np.asarray([x, y, 1]).reshape(3, 1)).round(decimals=0).astype('int32')
            image1[projection[0] + top_pad, projection[1] + left_pad, :] = image2[x, y, :]

    return image1


def interpPixels(image, x, y):
    """
        Interpolates the pixels
        :param image:
        :param x:
        :param y:
        :return:
    """
    proj_x, proj_y = np.arange(x - 1, x + 2, 1), np.arange(y - 1, y + 2, 1)
    xx, yy = np.meshgrid(x, y)
    values = image[xx, yy, :]
    # gaussian = [0.25, 0.5, 0.25]

    return np.mean(values, axis=0)




