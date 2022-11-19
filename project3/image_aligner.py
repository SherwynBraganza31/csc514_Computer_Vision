import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    h_matrix = V[-1].reshape(3,3)
    return h_matrix


def testHomography(image1: np.ndarray, image1_features: np.ndarray,
                   image2_features: np.ndarray, plot=False, h_matrix=None):
    """
        Tests the homography or h-matrix by plotting the original feature points from image1
        and the projection of the feature points in image2, on image1.
        Also prints out the mean squared error

        Author: Sherwyn Braganza

        :param image1: the first image on which the points are to be plotted
        :param image1_features: feature points of the first image
        :param image2_features: feature points of the second image
        :param plot: Whether to plot the result or just return the MSE
        :param h_matrix: optional variable if the user wants to provide the homography
        :return None
    """
    # get the homography matrix of projecting points from image2 to image1
    if h_matrix is None:
        h_matrix = computeHomography(image2_features, image1_features)

    source_pts = np.vstack((image2_features.T, np.ones(image2_features.shape[0])))
    projections = computeProjection(h_matrix, source_pts)
    squared_errors = np.sum((image1_features.T - projections[0:2,:]) ** 2, axis=0)

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(image1)
        for i in range(0, len(image1_features)):
            ax.scatter(image1_features[i][1], image1_features[i][0], marker='^', c='red')
            ax.scatter(projections[1, i], projections[0, i], marker='x', c='green')
        plt.show()

        print('MSE = {}'.format(np.mean(squared_errors)))

    return np.mean(squared_errors)


def ransac(image1_features: np.ndarray, image2_features: np.ndarray, max_ite=100) -> np.ndarray:
    best_homography = None
    best_mse = 10000
    iterations = 0

    while iterations < max_ite:
        random_idx = np.random.permutation(len(image1_features))[0:4]
        homography = computeHomography(image1_features[random_idx], image2_features[random_idx])
        mse = testHomography(np.zeros((10, 10)), image1_features, image2_features, h_matrix=homography)
        if mse < best_mse:
            best_homography = homography
            best_mse = mse

    if best_homography is None or best_mse > 10:
        return computeHomography(image1_features, image2_features)
    else:
        return best_homography


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


def inverseWarp(image1, image2, image1_features, image2_features) -> np.ndarray:
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
    # h_matrix = ransac(image1_features, image2_features)
    # h_inv = ransac(image2_features, image1_features)

    h_matrix, status = cv2.findHomography(image1_features, image2_features, cv2.RANSAC, 5.0)
    h_inv, status = cv2.findHomography(image2_features, image1_features, cv2.RANSAC, 5.0)

    ################# Image1 Padding based on projections of image2 ############################
    x_bound, y_bound = image2.shape[0], image2.shape[1] # get size of the second image
    original_shape = image1.shape

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
    right_pad = bound_max[1] - image1.shape[1] + 1 if bound_max[1] > image1.shape[1]  else 0
    top_pad = -bound_min[0] if bound_min[0] < 0 else 0
    bottom_pad = bound_max[0] - image1.shape[0] + 1 if bound_max[0] > image1.shape[0] else 0
    pad_params = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    image1 = np.pad(image1, pad_params, mode='constant', constant_values=0)
    reshape_params = top_pad, image2.shape
    #############################################################################################

    #################### Advanced Indexing ###########################
    x_start = original_shape[0] if bound_min[0] > 0 else bound_min[0]
    x_end = 0 if bound_max[0] < original_shape[0] else bound_max[0]
    y_start = original_shape[1] if bound_min[1] > 0 else bound_min[1]
    y_end = 0 if bound_max[1] < original_shape[1] else bound_max[1]
    x = np.arange(x_start, x_end, 1)
    y = np.arange(y_start, y_end, 1)
    x, y = np.meshgrid(x, y)
    orig_shape = x.shape

    x, y = x.reshape(1, -1), y.reshape(1, -1)
    ones = np.ones((1, x.shape[1]))
    point_matrix = np.vstack((x, y, ones))

    projections = computeProjection(h_matrix, point_matrix).round(decimals=0).astype('int32')
    x, y = x.reshape(orig_shape), y.reshape(orig_shape)
    xx, yy = projections[0, :], projections[1, :]
    xx, yy = xx.reshape(orig_shape), yy.reshape(orig_shape)
    x = x + top_pad
    y = y + left_pad

    #################### Padding image2 #############################
    bound_min = xx.min(), yy.min()
    bound_max = xx.max(), yy.max()

    # compute pad parameters and pad the image
    left_pad = -bound_min[1] if bound_min[1] < 0 else 0
    right_pad = bound_max[1] - image2.shape[1] + 1 if bound_max[1] > image2.shape[1] else 0
    top_pad = -bound_min[0] if bound_min[0] < 0 else 0
    bottom_pad = bound_max[0] - image2.shape[0] + 1 if bound_max[0] > image2.shape[0] else 0
    pad_params = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    image2 = np.pad(image2, pad_params, mode='constant', constant_values=0)

    xx, yy = xx + top_pad, yy + left_pad
    value = image2[xx, yy, :]
    image1[x, y, :] = value
    image1 = image1[reshape_params[0]: reshape_params[0] + reshape_params[1][0], :]

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
    # h_matrix = ransac(image2_features, image1_features)
    h_matrix, status = cv2.findHomography(image1_features, image2_features, cv2.RANSAC, 5.0)

    #################### Advanced Indexing ###########################
    x = np.arange(0, image2.shape[0], 1)
    y = np.arange(0, image2.shape[1], 1)
    x, y = np.meshgrid(x, y)

    x, y = x.reshape(1, -1), y.reshape(1, -1)
    ones = np.ones((1, x.shape[1]))
    point_matrix = np.vstack((x, y, ones))

    projections = computeProjection(h_matrix, point_matrix)
    xx, yy = projections[0].round(decimals=0).astype('int32'), projections[1].round(decimals=0).astype('int32')
    x = x.reshape(image2.shape[1], image2.shape[0])
    y = y.reshape(image2.shape[1], image2.shape[0])
    xx = xx.reshape(x.shape)
    yy = yy.reshape(x.shape)

    ################# Image1 Padding based on projections of image2 ############################
    bound_min = xx.min(), yy.min()
    bound_max = xx.max(), yy.max()

    # compute pad parameters and pad the image
    left_pad = -bound_min[1] if bound_min[1] < 0 else 0
    right_pad = bound_max[1] - image1.shape[1] + 1 if bound_max[1] > image1.shape[1] else 0
    top_pad = -bound_min[0] if bound_min[0] < 0 else 0
    bottom_pad = bound_max[0] - image1.shape[0] + 1 if bound_max[0] > image1.shape[0] else 0
    pad_params = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    image1 = np.pad(image1, pad_params, mode='constant', constant_values=0)

    xx = (xx + top_pad).reshape(x.shape)
    yy = (yy + left_pad).reshape(y.shape)

    image1[xx, yy, :] = image2[x, y, :]
    image1 = image1[top_pad: top_pad + image2.shape[0], :]


    ############### Regular slow indexing ###################
    # for x in range(image2.shape[0]):
    #     for y in range(image2.shape[1]):
    #         projection = computeProjection(h_matrix,
    #                                        np.asarray([x, y, 1]).reshape(3, 1)).round(decimals=0).astype('int32')
    #         if 0 <= projection[0] + top_pad < image1.shape[0] and 0 <= projection[1] + top_pad < image1.shape[1]:
    #             image1[projection[0] + top_pad, projection[1] + left_pad, :] = image2[x, y, :]

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




