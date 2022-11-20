from image_aligner import *
from image_feature_handler import *


class Tests:
    """
        Tests Class
    """
    def __init__(self):
        self.dir_name = 'P2_Benchmarks/test'

    def testUsrClicks(self) -> None:
        """
            Tests the user input capturing module. Runs the testHomography function to verify features are matched.

            Author: Sherwyn Braganza
            :return: None
        """
        images = loadImage(self.dir_name)
        img1_clicks, img2_clicks = grabUsrClicks(images[0], images[1])
        testHomography(images[0], img1_clicks, img2_clicks, plot=True)

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

        testHomography(images[4], img1_clicks, img2_clicks, plot=True)

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

        image1 = inverseWarp(images[3], images[4], img1_clicks, img2_clicks)
        plt.imshow(image1)
        plt.show()

        return

    def siftTest(self):
        images = loadImage(self.dir_name)
        image1_features, image2_features = getFeatures(images[0], images[1])
        image1 = forwardWarp(images[0], images[1], image1_features, image2_features)
        plt.imshow(image1)
        plt.show()

    def panoramaTest(self):
        self.dir_name = 'P2_Benchmarks/test'
        images = loadImage(self.dir_name)
        mid = int(len(images)/2)
        image1 = images[mid]

        for idx in range(mid-1, 0, -1):
            print('Merging main image and {}'.format(idx))
            image2 = images[idx]
            image1_features, image2_features = getFeatures(image1, image2)
            image1 = inverseWarp(image1, image2, image1_features, image2_features)
            plt.imshow(image1)

        for idx in range(mid+1, len(images)-1):
            print('Merging main image and {}'.format(idx))
            image2 = images[idx]
            image1_features, image2_features = getFeatures(image1, image2)
            image1 = inverseWarp(image1, image2, image1_features, image2_features)

        skimage.io.imsave('quad_merged.jpg', image1)

    def movieProjectionTest(self):
        images = loadImage('P2_Benchmarks/screen_warp')
        img1_clicks, img2_clicks = grabUsrClicks(images[0], skimage.img_as_ubyte(np.ones(images[0].shape)))
        img2_clicks = np.asarray([[0,0], [0, images[1].shape[1]], [images[1].shape[0], images[1].shape[1]],
                       [images[1].shape[0], 0]])
        image1 = inverseWarp(images[0], images[1], img1_clicks, img2_clicks)
        plt.imshow(image1)
        plt.show()



if __name__ == '__main__':

    # RUN ONLY ONE TEST AT TIME
    # BUGS IN DIRECTORY INDEXING

    tests = Tests()
    # tests.testUsrClicks()
    # tests.testPresetFeatures()
    # tests.testForwardWarp()
    # tests.testInverseWarp()
    # tests.siftTest()
    tests.panoramaTest()
    # tests.movieProjectionTest()