import numpy as np
from matplotlib import pyplot as plt
from image_feature_handler import loadImage, grabUsrClicks


def computeHomography(image1_pts, image2_pts):
    """
        Calculates the H matrix or the homography for a given set of points.
        Performs a least squares fit on the points to get the best possible homography
        by picking the singular vector corresponding to the smallest magnitude singular
        vector.

        Author: Sherwyn Braganza

        :param image1_pts: Feature points of image 1
        :param image2_pts: Feature points of image 2
        :return: the homography
    """
    p_matrix_rows = 2 * min([len(image1_pts), len(image2_pts)])
    point_matrix = np.zeros((p_matrix_rows, 9))
    for idx in range(int(p_matrix_rows/2)):
        x_d, y_d = image1_pts[idx]
        x_s, y_s = image2_pts[idx]
        point_matrix[2*idx, :] = np.asarray([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        point_matrix[2*idx+1, :] = np.asarray([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    U, S, V = np.linalg.svd(point_matrix)
    # print(np.sum(V[-1] ** 2))
    return V[-1].reshape((3, 3))


def testHomography(h_matrix, image1, image1_pts, image2_pts) -> None:
    """
        Tests the homography or h-matrix by plotting the original feature points from image1
        and the projection of the feature points in image2 on image1.
        Also prints out the mean absolute error.

        Author: Sherwyn Braganza

        :param h_matrix: the homography
        :param image1: the first image on which the points are to be plotted
        :param image1_pts: feature points of the first image
        :param image2_pts: feature points of the second image
    """
    abs_errors = []
    projected_pts = []
    for idx, pt in enumerate(image2_pts):
        temp_p_matrix = np.asarray([pt[0], pt[1], 1]).reshape(3,1)
        pt_proj = np.matmul(h_matrix, temp_p_matrix)
        pt_proj = pt_proj/pt_proj[-1]
        projected_pts.append(pt_proj)
        abs_errors.append(np.sum(np.abs(pt_proj[0:2]-image1_pts[idx][0:2])))

    fig, ax = plt.subplots()
    ax.imshow(image1)
    for i in range(0, len(image1_pts)):
        ax.scatter(image1_pts[i][0], image1_pts[i][1], marker='^', c='red')
        ax.scatter(projected_pts[i][0], projected_pts[i][1], marker='x', c='green')
    plt.show()

    print('MAE = {}'.format(np.mean(abs_errors)))


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
        img1_clicks, img2_clicks = grabUsrClicks(images[2], images[3])
        h_matrix = computeHomography(img1_clicks, img2_clicks)
        testHomography(h_matrix, images[2], img1_clicks, img2_clicks)

    def testPresetFeatures(self) -> None:
        """
            Tests the homography computation module
            Author: Sherwyn Braganza
            :return: None
        """
        images = loadImage(self.dir_name)
        img1_clicks = [(1178.510752688172, 225.82774193548357),
                       (1052.6612903225805, 1616.795483870967),
                       (708.2311827956987, 1636.6664516129026),
                       (2682.08064516129, 2173.1825806451607),
                       (1138.7688172043008, 702.7309677419353),
                       (3331.198924731183, 1146.5159139784942),
                       (3563.0268817204296, 1053.784731182795),
                       (1536.1881720430106, 1259.1180645161285),
                       (1350.7258064516127, 2623.5911827956984),
                       (966.5537634408602, 1795.6341935483865),
                       (794.338709677419, 1782.3868817204293),
                       (2933.779569892473, 2961.3976344086013)]
        img2_clicks = [(2390.6397849462373, 298.68795698924714),
                       (2304.532258064517, 1669.784731182795),
                       (1973.3494623655915, 1702.9030107526874),
                       (3960.4462365591407, 2199.6772043010747),
                       (2357.521505376344, 775.5911827956988),
                       (4589.693548387097, 1139.8922580645158),
                       (4828.145161290322, 1020.6664516129026),
                       (2768.18817204301, 1298.8599999999992),
                       (2615.844086021505, 2669.9567741935475),
                       (2218.4247311827967, 1855.2470967741929),
                       (2052.833333333334, 1841.999784946236),
                       (4218.768817204302, 2987.8922580645153)]
        h_matrix = computeHomography(img1_clicks, img2_clicks)
        testHomography(h_matrix, images[3], img1_clicks, img2_clicks)


if __name__ == '__main__':
    tests = Tests('P2_Benchmarks/DP_3')
    tests.testPresetFeatures()



