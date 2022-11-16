from image_aligner import *
from image_feature_handler import *


class Tests:
    """
        Tests Class
    """
    def __init__(self):
        self.dir_name = 'P2_Benchmarks/test'

    def testHomography(self, image1, image1_features, image2_features) -> None:
        """
            Tests the homography or h-matrix by plotting the original feature points from image1
            and the projection of the feature points in image2, on image1.
            Also prints out the mean squared error

            Author: Sherwyn Braganza

            :param image1: the first image on which the points are to be plotted
            :param image1_features: feature points of the first image
            :param image2_features: feature points of the second image
            :return None
        """
        squared_errors = []  # list to store the individual sqaured errors
        projected_pts = []  # list to store the projections of features from image2

        # get the homography matrix of projecting points from image2 to image1
        h_matrix = computeHomography(image2_features, image1_features)

        for idx, pt in enumerate(image2_features):
            pt_source = np.asarray([pt[0], pt[1], 1]).reshape(3, 1)
            pt_projection = computeProjection(h_matrix, pt_source)
            projected_pts.append(pt_projection)
            squared_errors.append(np.sum((pt_projection[0:2] - image1_features[idx].reshape(2,1))) ** 2)

        fig, ax = plt.subplots()
        ax.imshow(image1)
        for i in range(0, len(image1_features)):
            ax.scatter(image1_features[i][1], image1_features[i][0], marker='^', c='red')
            ax.scatter(projected_pts[i][1], projected_pts[i][0], marker='x', c='green')
        plt.show()

        print('MSE = {}'.format(np.mean(squared_errors)))

    def testUsrClicks(self) -> None:
        """
            Tests the user input capturing module. Runs the testHomography function to verify features are matched.

            Author: Sherwyn Braganza
            :return: None
        """
        images = loadImage(self.dir_name)
        img1_clicks, img2_clicks = grabUsrClicks(images[0], images[1])
        self.testHomography(images[0], img1_clicks, img2_clicks)

    def testPresetFeatures(self) -> None:
        """
            Tests the homography computation module
            Author: Sherwyn Braganza
            :return: None
        """
        images = loadImage(self.dir_name)

        # preselected feature pairs
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

        self.testHomography(images[0], img1_clicks, img2_clicks)

    def testForwardWarp(self) -> None:
        """
            Combines both the images by forward warping.
            :return: None
        """
        images = loadImage(self.dir_name)

        # preselected feature pairs
        img2_clicks = np.asarray([[355.90143369, 1372.25985663],
                                  [852.3172043, 1384.804659],
                                  [504.64695341, 1680.50358423],
                                  [463.42831541, 1024.58960573],
                                  [823.64336918, 1078.35304659],
                                  [669.52150538, 1521.00537634],
                                  [658.7688172, 1859.71505376],
                                  [920.41756272, 1685.87992832],
                                  [1187.44265233, 1709.17741935]])

        img1_clicks = np.asarray([[368.44623656, 598.06630824],
                                  [863.06989247, 601.65053763],
                                  [527.94444444, 886.59677419],
                                  [454.46774194, 223.51433692],
                                  [841.56451613, 280.86200717],
                                  [680.27419355, 734.26702509],
                                  [676.68996416, 1047.88709677],
                                  [918.62544803, 886.59677419],
                                  [1165.93727599, 909.89426523]])

        image1 = forwardWarp(images[0], images[1], img1_clicks, img2_clicks)
        plt.imshow(image1)
        plt.show()

        return

    def testInverseWarp(self) -> None:
        """
            Combines both the images by forward warping.
            :return: None
        """
        images = loadImage(self.dir_name)

        # preselected feature pairs
        img2_clicks = np.asarray([[355.90143369, 1372.25985663],
                                  [852.3172043, 1384.804659],
                                  [504.64695341, 1680.50358423],
                                  [463.42831541, 1024.58960573],
                                  [823.64336918, 1078.35304659],
                                  [669.52150538, 1521.00537634],
                                  [658.7688172, 1859.71505376],
                                  [920.41756272, 1685.87992832],
                                  [1187.44265233, 1709.17741935]])

        img1_clicks = np.asarray([[368.44623656, 598.06630824],
                                  [863.06989247, 601.65053763],
                                  [527.94444444, 886.59677419],
                                  [454.46774194, 223.51433692],
                                  [841.56451613, 280.86200717],
                                  [680.27419355, 734.26702509],
                                  [676.68996416, 1047.88709677],
                                  [918.62544803, 886.59677419],
                                  [1165.93727599, 909.89426523]])

        image1 = inverseWarp(images[0], images[1], img1_clicks, img2_clicks)
        plt.imshow(image1)
        plt.show()

        return

    def siftTest(self):
        images = loadImage(self.dir_name)
        image1_features, image2_features = getFeatures(images[0], images[1])
        # self.testHomography(images[0], image1_features, image2_features)
        image1 = forwardWarp(images[0], images[1], image1_features, image2_features)
        plt.imshow(image1)
        plt.show()

    def quadPanoramaTest(self):
        self.dir_name = 'P2_Benchmarks/quad'
        images = loadImage(self.dir_name)
        image1 = images[0]

        for idx in range(1, len(images)-10):
            image2 = images[idx]
            image1_features, image2_features = getFeatures(image1, image2)
            image1 = forwardWarp(image1, image2, image1_features, image2_features)

        plt.imshow(image1)
        plt.show()


if __name__ == '__main__':
    tests = Tests()
    # tests.testUsrClicks()
    # tests.testPresetFeatures()
    # tests.testForwardWarp()
    # tests.testInverseWarp()
    # tests.siftTest()
    tests.quadPanoramaTest()