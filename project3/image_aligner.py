import numpy as np
from matplotlib import pyplot as plt
from image_feature_handler import loadImage, grabUsrClicks
import cv2

def computeHomography(image1_features, image2_features):
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
        :return: the homography
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

def inverseWarp(image1, image2, image1_features, image2_features):
    h_matrix = computeHomography(image1_features, image2_features)
    h_inv = np.linalg.inv(h_matrix)
    rows, cols = image2.shape[0], image2.shape[1]

    # The extremities of image 2 in the image 2 plane
    edge_pts = np.asarray([[0, 0, 1],  # TopLeft
                [0, cols, 1],  # TopRight
                [rows, cols, 1],  # BottomRight
                [rows, 0, 1]]).T    # Bottom Left

    edge_projection = np.matmul(h_inv, edge_pts)
    edge_projection = (edge_projection/edge_projection[-1, :])
    edge_projection = np.round(edge_projection, decimals=0).astype('int32')

    min_x = -min(edge_projection[0, :]) if min(edge_projection[0, :]) < 0 else 0
    max_x = max(edge_projection[0, :]) - image1.shape[0] if max(edge_projection[0, :]) > image1.shape[0] else 0
    min_y = -min(edge_projection[1, :]) if min(edge_projection[1, :]) < 0 else 0
    max_y = max(edge_projection[1, :]) - image1.shape[1] if max(edge_projection[1, :]) > image1.shape[1] else 0
    pad_params = ((min_x, max_x), (min_y, max_y), (0, 0))
    image1 = np.pad(image1, pad_params, mode='constant', constant_values=0)

    x_span = np.linspace(min(edge_projection[0, :]), max(edge_projection[0, :]),
                         num=max(edge_projection[0, :]) - min(edge_projection[0, :]), dtype='int32',
                         endpoint=False)
    y_span = np.linspace(min(edge_projection[1, :]), max(edge_projection[1, :]),
                         num=max(edge_projection[1, :]) - min(edge_projection[1, :]), dtype='int32',
                         endpoint=False)

    for x in x_span:
        for y in y_span:
            temp_p = np.matmul(h_matrix, np.asarray([x, y, 1]).reshape(3,1))
            temp_p = temp_p/temp_p[-1]
            try:
                value = image2[int(temp_p[0]), int(temp_p[1]), :]
                image1[x + min_x, y + min_y, :] = value
            except IndexError:
                value = 0

    plt.imshow(image1)
    plt.show()

    return

def forwardWarp(image1, image2, image1_features, image2_features):
    h_inv = computeHomography(image2_features, image1_features)
    x_bound, y_bound = image2.shape[0], image2.shape[1]

    # The extremities of image 2 in the image 2 plane
    edge_pts = np.asarray([[0, 0, 1],  # TopLeft
                [0, y_bound, 1],  # TopRight
                [x_bound, y_bound, 1],  # BottomRight
                [x_bound, 0, 1]]).T    # Bottom Left

    edge_projection = np.matmul(h_inv, edge_pts)
    edge_projection = ((edge_projection/edge_projection[-1]).round(decimals=0)).astype('int32')

    bound_min = np.min(edge_projection, axis=1)[0:2]
    bound_max = np.max(edge_projection, axis=1)[0:2]
    left_pad = -bound_min[1] if bound_min[1] < 0 else 0
    right_pad = bound_max[1] - y_bound if bound_max[1] > y_bound else 0
    top_pad = -bound_min[0] if bound_min[0] < 0 else 0
    bottom_pad = bound_max[0] - x_bound if bound_max[0] > x_bound else 0
    pad_params = ((top_pad, bottom_pad), (left_pad, right_pad), (0,0))
    image1 = np.pad(image1, pad_params, mode='constant', constant_values=0)

    for x in range(image2.shape[0]):
        for y in range(image2.shape[1]):
            projection = np.matmul(h_inv, np.asarray([x, y, 1]).reshape(3, 1))
            projection /= projection[-1]
            projection = np.round(projection, 0).astype('int32')
            image1[projection[0] + top_pad, projection[1] + left_pad, :] = image2[x, y, :]

    plt.imshow(image1)
    plt.show()

class Tests:
    """
        Tests Class
    """
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def testUsrClicks(self) -> None:
        """
            Tests the user input capturing module. Computes the homography and tests it by projecting
            corresponding features from image2 onto image1

            Author: Sherwyn Braganza
            :return: None
        """
        images = loadImage(self.dir_name)
        img1_clicks, img2_clicks = grabUsrClicks(images[0], images[1])
        h_matrix = computeHomography(img1_clicks, img2_clicks)
        self.testHomography(images[0], img1_clicks, img2_clicks)

    def testPresetFeatures(self) -> None:
        """
            Tests the homography computation module
            Author: Sherwyn Braganza
            :return: None
        """
        images = loadImage(self.dir_name)
        img1_clicks = np.asarray([[355.90143369, 1372.25985663],
                                  [852.3172043,  1384.804659],
                                  [504.64695341, 1680.50358423],
                                  [463.42831541, 1024.58960573],
                                  [823.64336918, 1078.35304659],
                                  [669.52150538, 1521.00537634],
                                  [658.7688172, 1859.71505376],
                                  [920.41756272, 1685.87992832],
                                  [1187.44265233, 1709.17741935]])

        img2_clicks = np.asarray([[368.44623656, 598.06630824],
                                  [863.06989247, 601.65053763],
                                  [527.94444444, 886.59677419],
                                  [454.46774194, 223.51433692],
                                  [841.56451613, 280.86200717],
                                  [680.27419355, 734.26702509],
                                  [676.68996416, 1047.88709677],
                                  [918.62544803, 886.59677419],
                                  [1165.93727599,909.89426523]])

        h_matrix = computeHomography(img1_clicks, img2_clicks)
        # self.testHomography(images[0], img1_clicks, img2_clicks)
        forwardWarp(images[0], images[1], img1_clicks, img2_clicks)

    def testHomography(self, image1, image1_features, image2_features) -> None:
        """
            Tests the homography or h-matrix by plotting the original feature points from image1
            and the projection of the feature points in image2 on image1.
            Also prints out the mean absolute error.

            Author: Sherwyn Braganza

            :param h_matrix: the homography
            :param image1: the first image on which the points are to be plotted
            :param image1_features: feature points of the first image
            :param image2_features: feature points of the second image
        """
        squared_errors = []
        projected_pts = []
        h_matrix = computeHomography(image2_features, image1_features)
        for idx, pt in enumerate(image2_features):
            temp_p_matrix = np.asarray([pt[0], pt[1], 1]).reshape(3, 1)
            pt_proj = np.matmul(h_matrix, temp_p_matrix)
            pt_proj = pt_proj/pt_proj[-1]
            projected_pts.append(pt_proj)
            difference = pt_proj[0:2] - image1_features[idx].reshape(2,1)
            squared = difference ** 2
            squared_errors.append(np.sum(squared))

        fig, ax = plt.subplots()
        ax.imshow(image1)
        for i in range(0, len(image1_features)):
            ax.scatter(image1_features[i][1], image1_features[i][0], marker='^', c='red')
            ax.scatter(projected_pts[i][1], projected_pts[i][0], marker='x', c='green')
        plt.show()

        print('MSE = {}'.format(np.mean(squared_errors)))


if __name__ == '__main__':
    tests = Tests('P2_Benchmarks/test')
    # tests.testUsrClicks()
    tests.testPresetFeatures()


