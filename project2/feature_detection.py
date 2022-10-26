import numpy as np
import skimage.io
import image_handler
from scipy import signal

def generateGaussian(sigma, size) -> np.ndarray:
    """
        Generates a Gaussian using std_dev sigma of size (size x size)
        :param sigma: The standard deviation
        :param size: The size or shape of the Guassian
        :return: np.ndarray: The guassian that is generated
    """
    # enforce an odd sized kernel
    if not size % 2:
        raise Exception('Even Guassian size given')

    center = size // 2
    kernel = np.zeros(size)

    # Generate Gaussian blur.
    for x in range(size):
        diff = (x - center) ** 2
        kernel[x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = np.asarray(kernel.reshape(-1, 1).T * kernel.reshape(-1, 1))
    kernel = kernel / np.sum(kernel)

    return kernel

def rotateMatrix()

class FeatureDetection:

    def __init__(self, image: np.ndarray = None, features=None, response=None):
        self.image = image
        self.grayscale = image_handler.convertToGrayscale(image)
        self.features = features
        self.response = response
 
    def getFeatures(self):
        """
            Implements a corner strength detection method to identify interest points.
            The performs non-Maximal Suppresion on the identified points to remove
            duplicates of features really close to each other.

            Author Sherwyn Braganza

            :return: None
        """
        SOBEL_X = np.asarray([[-1,-1, 0, 1, 1],
                      [-2,-1, 0, 1, 2],
                      [-4, -2, 0, 2, 4],
                      [-2, -1, 0, 1, 2],
                      [-1,-1, 0, 1, 1]])
        SOBEL_Y = SOBEL_X.T
        I_X = signal.convolve2d(self.grayscale, SOBEL_X, mode='same')
        I_Y = signal.convolve2d(self.grayscale, SOBEL_Y, mode='same')

        # Pick a feature Detection Method
        self.computeHarrisStephens(I_X, I_Y)

        self.features = []
        bounds = int(10 / 100 * image.shape[0]), int(10 / 100 * image.shape[1])
        for row, x in enumerate(self.response):
            for col, y in enumerate(x):
                if row < bounds[0] or row > (image.shape[0] - bounds[0]) \
                        or col < bounds[1] or col > (image.shape[1] - bounds[1]):
                    continue
                if y > 30:
                    self.features.append([(row, col), self.getOrientation(I_X, I_Y, (row, col))])

        self.nonMaximalSuppression((-2,2))

    def getOrientation(self, I_X, I_Y, loc):
        """
            Calculates the orientaion of the gradient at point loc.
            Does this by taking the tan^(-1) of the ratio of gradient
            in the Y direction to the gradient in the X.

            Author: Sherwyn Braganza

            :param I_X: The gradient in the X direction
            :param I_Y: The gradient in the Y direction
            :param loc: The point at which the orientation should be found
            :return: The angle in degrees.
        """
        return np.degrees(np.arctan2(I_Y[loc],I_X[loc]))

    def computeHarrisStephens(self, I_X, I_Y):
        """
            Computes the corner strength based on the Harris-Stephens
            Algorithm.

            Author: Sherwyn Braganza

            :param I_X: The derivative of the image in the X direction
            :param I_Y: The derivative of the image in the Y direction
            :return: The corner strength response
        """
        w_1d = np.asarray([1, 4, 7, 4, 1]).reshape(-1, 1)
        w_2d = np.matmul(w_1d, w_1d.T)
        w = w_2d / np.sum(w_2d)
        I_XX = signal.convolve2d(I_X ** 2, w, mode='same')
        I_YY = signal.convolve2d(I_Y ** 2, w, mode='same')
        I_XY = signal.convolve2d(I_X * I_Y, w, mode='same')
        alpha = 0.04

        # determinant
        det = I_XX * I_YY - I_XY * I_XY
        # trace
        trace = I_XX + I_YY

        self.response = det - alpha * (trace ** 2)

    def nonMaximalSuppression(self, suppression_window) -> None:
        """
            Implements non maximal suppression in a specified window size (suppression_window).
            Manipulates the feature array internally and deletes points that are supposed to
            be suppressed.

            Author: Sherwyn Braganza

            :param suppression_window: The window size
            :return: None
        """
        count = 0
        idx = 0
        while idx < len(self.features):
            feature = self.features[idx]
            if self.response[feature[0][0], feature[0][1]] < \
                    np.max(
                        self.response[feature[0][0] + suppression_window[0]: feature[0][0] + suppression_window[1]+1,
                        feature[0][1] + suppression_window[0]: feature[0][1] + suppression_window[1]+1]
                    ) and feature:
                del self.features[idx]
                count += 1
            else:
                idx += 1

        print('{} features suppressed'.format(count))

    def showFeatures(self, loc: tuple, scale: float = 1, orientation: float = 0) -> None:
        """
            Shows (overlays) the features detected on the image. Plots them in the form of boxes with
            orientation.

            :param loc: the centroid of where the feature was detected
            :param scale:
            :param orientation: The orientation of the feature.
            :return: None
        """
        # orientation += 90
        buffer = self.constructBufferSquare(loc, scale)
        rotated_buffer = self.rotateBufferSquare(loc, buffer, orientation)
        plotpoints = self.generatePlotPoints(loc, rotated_buffer, scale)

        for x in plotpoints:
            self.image[x[0], x[1], :] = np.asarray([255, 0, 0])

    def constructBufferSquare(self, loc: tuple, scale: float) -> np.ndarray:
        """
            Gets 4 points around loc that corresponds to the edges of the
            buffer square. The points are listed in the bottomLeft, bottomRight,
            topRight, topLeft, bottomTopRightCenter order
        :param loc: The centroid of the buffer square
        :param scale:
        :return:
        """
        sideLength = int((scale - 1)*2 + 1)
        bottomLeft = np.asarray([loc[0] - sideLength // 2, loc[1] - sideLength // 2])
        bottomRight = np.asarray([loc[0] + sideLength // 2, loc[1] - sideLength // 2])
        topRight = np.asarray([loc[0] + sideLength // 2, loc[1] + sideLength // 2])
        topLeft = np.asarray([(loc[0] - sideLength // 2, loc[1] + sideLength // 2)])
        bottomTopRightCenter = np.asarray([(loc[0] + sideLength // 2, loc[1])])

        return np.vstack((bottomLeft,bottomRight,topRight,topLeft, bottomTopRightCenter))

    def rotateBufferSquare(self, loc: tuple, points: np.ndarray, orientation: float) -> np.ndarray:
        """
            Rotate the buffer square according to the orientation given.

            Author: Sherwyn Braganza

            :param loc: The center of the feature
            :param points: The extremities of the buffer square
            :param orientation: The dominant orientation of the feature
            :return: The new rotated buffer square's endpoints
        """
        loc = np.asarray(loc)
        cob_points = points - loc
        theta = np.radians(orientation)
        c, s = np.cos(theta), np.sin(theta)
        rotMatrix = np.array(((c, s),
                              (-s, c)))

        cob_points = np.matmul(cob_points, rotMatrix)

        return np.rint(cob_points + loc).astype(int, casting='unsafe')

    def generatePlotPoints(self, loc: tuple, points: np.ndarray, scale) -> np.ndarray:
        """
            Interpolate between extremities and get points along the edges of the buffer
            square for plotting

            Author: Sherwyn Braganza

            :param loc: The center of the feature
            :param points: The endpoints of edges of the feature
            :param scale: The span of the feature
            :return: Points along the edges of the feature buffer
        """
        plotpoints = np.asarray([0, 0])

        for i in range(len(points)):
            temp_x = np.linspace(points[i % 4, 0], points[(i+1)%4, 0], num=(scale-1)*2 + 1, endpoint=True, dtype=int)
            temp_y = np.linspace(points[i % 4, 1], points[(i+1)%4, 1], num=(scale-1)*2 + 1, endpoint=True, dtype=int)
            plotpoints = np.vstack((plotpoints, np.hstack((temp_x.reshape(-1, 1), temp_y.reshape(-1, 1)))))

        plotpoints = np.vstack((
            plotpoints,
            np.hstack((
                np.linspace(loc[0], points[-1, 0], num=scale, endpoint=True, dtype=int).reshape(-1, 1),
                np.linspace(loc[1], points[-1, 1], num=scale, endpoint=True, dtype=int).reshape(-1, 1),
            ))
        ))

        return np.delete(plotpoints, 0, axis=0)

class FeatureMatching:

    def __init__(self, featureImage1: FeatureDetection, featureImage2: FeatureDetection):
        self.featureImage1 = featureImage1
        self.featureImage2 = featureImage2
        self.combinedImage = np.hstack((featureImage1, featureImage2))
        self.shift_amount = featureImage1.image.shape

    def rotate
    def grabGradientChunk(self, location, scale, orientation):


    def getMatches(self):
        for x in self.featureImage1.features:
            x =


if __name__ == '__main__':

    # ######### Function tests ##########
    # imageFeature = FeatureDetection()
    # center = 3,4
    # scale = 4
    # square = imageFeature.constructBufferSquare(center, scale)
    # print(square)
    # points = imageFeature.rotateBufferSquare(center, square, 360)
    # points = imageFeature.generatePlotPoints(center, points, scale)
    # plt.plot(points[:,0], points[:,1])
    # plt.show()

    # ########## Feature Plotter Test ###############
    # image = image_handler.readImage('datasets/Yosemite/Yosemite1.jpg')
    # features = FeatureDetection(image)
    # features.showFeatures((88, 96), 20, 45)
    # features.showFeatures((250, 450), 20, -36)
    # skimage.io.imshow(image)
    # skimage.io.show()

    image = image_handler.readImage('datasets/Yosemite/Yosemite1.jpg')
    image2 = image_handler.readImage('datasets/Yosemite/Yosemite2.jpg')
    image1_features = FeatureDetection(image, match_image=image2)
    image2_features = FeatureDetection(image2)
    image1_features.getFeatures()
    image2_features.getFeatures()
    for x in image1_features.features:
        image1_features.showFeatures(x[0], 5, x[1])
    for x in image2_features.features:
        image2_features.showFeatures(x[0], 5, x[1])
    skimage.io.imshow(np.hstack((image, image2)))
    skimage.io.show()

    # image = image_handler.readImage('datasets/box.jpg')
    # image1_features = FeatureDetection(image)
    # for x in image1_features.features:
    #     image1_features.showFeatures(x[0], 10, x[1])
    # skimage.io.imshow(image)
    # skimage.io.show()


