import numpy as np
import skimage.io
import image_handler
from scipy import signal, ndimage
import matplotlib.pyplot as plt

def generateGaussian(sigma, size) -> np.ndarray:
    """
        Generates a Gaussian of std_dev sigma and of size (size x size)
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

    # matrix multiply the 1D Guassian with its transpose to create a 2-D version
    kernel = np.asarray(kernel.reshape(-1, 1).T * kernel.reshape(-1, 1))
    kernel = kernel / np.sum(kernel)
    return kernel

def rotateMatrix(center: tuple[int, int], matrix: np.ndarray, orientation):
    """
        Rotates a matrix about a given center. From a broad level view,
        uses the Change of Basis Theorem to bring the matrix to (0, 0) and
        uses the standard rotation matrix about (0, 0) to rotate.

        :param center: The centre about which the rotation should be performed
        :param matrix: The matrix to be rotated
        :param orientation: The angle by which it should be rotated by
        :return: The rotated matrix.
    """
    loc = np.asarray(center)
    cob_points = matrix - center  # change of basis points
    theta = np.radians(orientation)
    c, s = np.cos(theta), np.sin(theta)

    # rotation Matrix about 0
    rotMatrix = np.array(((c, s),
                          (-s, c)))

    cob_points = np.matmul(cob_points, rotMatrix)

    return np.rint(cob_points + loc).astype(int, casting='unsafe')

def normalizeMatrix(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.min())/(matrix.max() - matrix.min())

def getGradientOrientation(g_x, g_y, loc):
    """
        Grabs the orientation in degrees of the 2-D Gradient Vector at
        loc.

        Author: Sherwyn Braganza

        :param g_x: The gradient in the X direction
        :param g_y: The gradient in the Y direction
        :param loc: The point at which the orientation should be found
        :return: The angle in degrees.
    """
    return np.degrees(np.arctan2(g_y[loc], g_x[loc]))

def grabBufferSquare(loc: tuple, size: int) -> np.ndarray:
    """
        Grabs the end points of a hypothetical buffer square, centered
        at loc. The size of the square is determined by size

        The 4 end points corresponds to the edges of the buffer square.
        The points are listed in the bottomLeft, bottomRight,
        topRight, topLeft order.

        Additionally, it grabs a 5th point that corresponds to the projection
        of loc on the Right Edge of the buffer square.

        Author: Sherwyn Braganza

        :param loc: The centroid of the buffer square
        :param size: The edge size of the square
        :return: np.array corresponding to the end points.
    """
    sideLength = size
    bottomLeft = np.asarray([loc[0] - sideLength // 2, loc[1] - sideLength // 2])
    bottomRight = np.asarray([loc[0] + sideLength // 2, loc[1] - sideLength // 2])
    topRight = np.asarray([loc[0] + sideLength // 2, loc[1] + sideLength // 2])
    topLeft = np.asarray([(loc[0] - sideLength // 2, loc[1] + sideLength // 2)])
    bottomTopRightCenter = np.asarray([(loc[0] + sideLength // 2, loc[1])])

    return np.vstack((bottomLeft,bottomRight,topRight,topLeft, bottomTopRightCenter))

class FeatureDetection:

    def __init__(self, image: np.ndarray = None, features=None, response=None):
        """

            :param image:
            :param features:
            :param response:
        """
        self.image = image
        self.grayscale = image_handler.convertToGrayscale(image) if image is not None else None
        self.features = features
        self.response = response
        self.x_grad = None
        self.y_grad = None
 
    def getFeatures(self, response_threshold: float = 0):
        """
            Implements a corner strength detection method to identify interest points.
            The performs non-Maximal Suppression on the identified points to remove
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

        # convolve the sobel with the image to get gradients in each dir
        self.x_grad = g_x = signal.convolve2d(self.grayscale, SOBEL_X, mode='same')
        self.y_grad = g_y = signal.convolve2d(self.grayscale, SOBEL_Y, mode='same')

        # Pick a feature Detection Method
        self.computeBrownSzeliskiWinder(g_x, g_y)
        self.response = normalizeMatrix(self.response)

        self.features = []

        # Set exclusion bounds so we dont pick up corners as features
        # set to default 5% exclusion on each edge
        bounds = int(5 / 100 * self.image.shape[0]), int(5 / 100 * self.image.shape[1])

        # iterate through the response matrix and grab features
        scale = 1
        for row, x in enumerate(self.response):
            for col, y in enumerate(x):
                if row < bounds[0] or row > (self.image.shape[0] - bounds[0]) \
                        or col < bounds[1] or col > (self.image.shape[1] - bounds[1]):
                    continue
                if y > response_threshold:
                    self.features.append([(row, col), scale, getGradientOrientation(g_x, g_y, (row, col))])

        # suppress non maximums in a 5x5 window
        self.suppressNonMaximals((-10,10))

    # def adaptiveNonMaximalSuppression(self, num_points):
    #     bounds = int(5 / 100 * self.image.shape[0]), int(5 / 100 * self.image.shape[1])
    #     bounded_response = np.reshape(self.response[bounds[0]:-bounds[0], bounds[1]:-bounds[1]]), (1, -1)
    #     bounded_response = np.sort(bounded_response)
    #     suppression_radius = 0
    #     max_idx = np.where(self.response == bounded_response[0], self.response)
    #
    #     for i in range(1, bounded_response):
    #         if bounded_response[0] < 0.9 * bounded_response[i]:
    #             temp_idx = np.where(self.response == bounded_response[i], self.response)
    #             suppression_radius = min(
    #                 ((max_idx[0] - temp_idx[0])**2 + (max_idx[1] - temp_idx[1])**2) ** 0.5
    #             )
    #
    #     points = grabBufferSquare((max_idx[0], max_idx[1]), 2*suppression_radius)
    #     self.features.append(max_idx)
    #     polygon_list = [shapely.geometry.polygon(points+points[0])]
    #
    #     for i in range(1, bounded_response):
    #         temp_idx = np.where(self.response == bounded_response[i], self.response)
    #         contains = False
    #         for x in polygon_list:
    #             if x.contains(shapely.geometry.point(temp_idx)):
    #                 contains = True
    #                 break
    #         if not contains:
    #             self.features.append(temp_idx)
    #             points = grabBufferSquare(temp_idx, 2*suppression_radius)
    #             polygon_list.append(shapely.geometry.polygon(points+points[0]))
    #
    #

    def computeHarrisStephensResponse(self, g_x, g_y):
        """
            Computes the corner strength based on the Harris-Stephens
            Algorithm.

            Author: Sherwyn Braganza

            :param g_x: The derivative of the image in the X direction
            :param g_y: The derivative of the image in the Y direction
            :return: The corner strength response
        """
        w_1d = np.asarray([1, 4, 7, 4, 1]).reshape(-1, 1)
        w_2d = np.matmul(w_1d, w_1d.T)
        w = w_2d / np.sum(w_2d)
        g_xX = signal.convolve2d(g_x ** 2, w, mode='same')
        g_yY = signal.convolve2d(g_y ** 2, w, mode='same')
        g_xY = signal.convolve2d(g_x * g_y, w, mode='same')
        alpha = 0.04

        # determinant
        det = g_xX * g_yY - g_xY * g_xY
        # trace
        trace = g_xX + g_yY

        self.response = det - alpha * (trace ** 2)

    def computeBrownSzeliskiWinder(self,  g_x, g_y):
        w_1d = np.asarray([1, 4, 7, 4, 1]).reshape(-1, 1)
        w_2d = np.matmul(w_1d, w_1d.T)
        w = w_2d / np.sum(w_2d)
        g_xX = signal.convolve2d(g_x ** 2, w, mode='same')
        g_yY = signal.convolve2d(g_y ** 2, w, mode='same')
        g_xY = signal.convolve2d(g_x * g_y, w, mode='same')
        # determinant
        det = g_xX * g_yY - g_xY * g_xY
        # trace
        trace = g_xX + g_yY

        self.response = det/trace

    def suppressNonMaximals(self, suppression_window) -> None:
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

    def plotFeature(self, loc: tuple, span: int = 5, orientation: float = 0) -> None:
        """
            Shows (overlays) the features detected on the image. Plots them in the form of boxes with
            orientation.

            :param loc: the centroid of where the feature was detected
            :param span: The span of the feature
            :param orientation: The orientation of the feature.
            :return: None
        """
        buffer_size = span
        buffer = grabBufferSquare(loc, buffer_size)
        rotated_buffer = rotateMatrix(loc, buffer, orientation)
        plotpoints = self.generatePlotPoints(loc, rotated_buffer, buffer_size)

        for x in plotpoints:
            self.image[x[0], x[1], :] = np.asarray([255, 0, 0])

    def generatePlotPoints(self, loc: tuple, points: np.ndarray, size: int) -> np.ndarray:
        """
            Interpolate between extremities and get points along the edges of the buffer
            square for plotting

            Author: Sherwyn Braganza

            :param loc: The center of the feature
            :param points: The endpoints of edges of the feature
            :param size: The span of the feature
            :return: Points along the edges of the feature buffer
        """
        plotpoints = np.asarray([0, 0])

        density_factor = 3
        for i in range(len(points)):
            temp_x = np.linspace(points[i % 4, 0], points[(i+1)%4, 0], num=size*density_factor+1,
                                 endpoint=True, dtype=int)
            temp_y = np.linspace(points[i % 4, 1], points[(i+1)%4, 1], num=size*density_factor+1,
                                 endpoint=True, dtype=int)
            plotpoints = np.vstack((plotpoints, np.hstack((temp_x.reshape(-1, 1), temp_y.reshape(-1, 1)))))

        plotpoints = np.vstack((
            plotpoints,
            np.hstack((
                np.linspace(loc[0], points[-1, 0], num=size*density_factor+1, endpoint=True,
                            dtype=int).reshape(-1, 1),
                np.linspace(loc[1], points[-1, 1], num=size*density_factor+1, endpoint=True,
                            dtype=int).reshape(-1, 1),
            ))
        ))

        return np.delete(plotpoints, 0, axis=0)

    def showFeatures(self) -> None:
        """
            Goes through the each elemenent of the feature array and
            calls plotFeature to plot the feature on the image.

            Author: Sherwyn Braganza

            :return: None
        """
        for x in self.features:
            self.plotFeature(x[0], int(np.rint(5 * x[1])), x[2])

class FeatureMatching:

    def __init__(self, featureImage1: FeatureDetection, featureImage2: FeatureDetection):
        self.featureImage1 = featureImage1
        self.featureImage2 = featureImage2
        self.combinedImage = np.hstack((featureImage1.image, featureImage2.image))
        self.shift_amount = (0, featureImage1.image.shape[1])
        self.match_pairs = []
        self.ssd_distance = []
        self.ratio_test = -1

    def grabDescriptor(self, featureImage: FeatureDetection, loc, scale, orientation):
        side_length = int(np.rint(scale * 15))
        min_x, max_x = loc[0] - side_length, loc[0] + side_length
        min_y, max_y = loc[1] - side_length, loc[1] + side_length

        x_chunk = featureImage.x_grad[min_x: max_x + 1, min_y: max_y + 1]
        y_chunk = featureImage.y_grad[min_x: max_x + 1, min_y: max_y + 1]

        x_chunk = ndimage.rotate(x_chunk, angle=-orientation, reshape=False)
        y_chunk = ndimage.rotate(y_chunk, angle=-orientation, reshape=False)

        center = [x_chunk.shape[0]//2, x_chunk.shape[1]//2]
        min_x, max_x = center[0] - side_length//2, center[0] + side_length//2
        min_y, max_y = center[1] - side_length//2, center[1] + side_length//2

        x_chunk = x_chunk[min_x:max_x + 1, min_y:max_y+1]
        y_chunk = y_chunk[min_x:max_x + 1, min_y:max_y+1]

        # fig, axs = plt.subplots(2)
        # axs[0].imshow(x_chunk, cmap='gray')
        # axs[1].imshow(y_chunk, cmap='gray')
        plt.show()

        return normalizeMatrix(x_chunk), normalizeMatrix(y_chunk)

    def getMatches(self):
        gradient_tolerance = 10
        min_dist = 0
        for idx_x, x in enumerate(self.featureImage1.features):
            for idx_y, y in enumerate(self.featureImage2.features):
                image_a_grad = self.grabDescriptor(self.featureImage1, x[0], x[1], x[2])
                image_b_grad = self.grabDescriptor(self.featureImage2, y[0], y[1], y[2])
                grad_diff = np.sqrt((image_a_grad[0] - image_b_grad[0]) ** 2 +
                                    (image_a_grad[1] - image_b_grad[1]) ** 2)

                if np.sum(grad_diff)/(2*len(image_a_grad)) < gradient_tolerance:
                    self.match_pairs.append((x,y))
                    min_dist = min((grad_diff.min(), min_dist))
                    self.ssd_distance.append(np.sum(grad_diff))
                    del self.featureImage1.features[idx_x]
                    del self.featureImage2.features[idx_y]
                    break

        sortedSSD = np.sort(self.ssd_distance)
        self.ratio_test = sortedSSD[0]/sortedSSD[1]
        print(sortedSSD[0], self.ratio_test)
        print(len(self.match_pairs))

    def plotMatches(self):
        for match in self.match_pairs:
            point_density = int(np.rint(2 * np.sqrt(
                (match[0][0][0] - match[1][0][0] + self.shift_amount[0])**2 +
                (match[0][0][1] - match[1][0][1] + self.shift_amount[1])**2
            )))
            x_points = np.linspace(match[0][0][0], match[1][0][0] + self.shift_amount[0] + 1,
                                   num=point_density, dtype='int32')
            y_points = np.linspace(match[0][0][1], match[1][0][1] + self.shift_amount[1] + 1,
                                   num=point_density, dtype='int32')
            for i in range(point_density):
                self.combinedImage[x_points[i], y_points[i], :] = np.asarray([0,255, 0])

class Tests:
    def __init__(self):
        return

    def bufferSquareTest(self):
        imageFeature = FeatureDetection()
        center = 3,4
        scale = 4
        square = grabBufferSquare(center, scale)
        print(square)
        points = rotateMatrix(center, square, 360)
        points = imageFeature.generatePlotPoints(center, points, scale)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

    def bufferSquareImageTest(self):
        image = image_handler.readImage('datasets/Yosemite/Yosemite1.jpg')
        features = FeatureDetection(image)
        features.getFeatures(response_threshold=0.2)
        features.showFeatures()
        skimage.io.imshow(image)
        skimage.io.imsave('feature_square_example.png', image)
        skimage.io.show()

    def featureGrabPlotTest(self):
        image = image_handler.readImage('datasets/box.jpg')
        image1_features = FeatureDetection(image)
        image1_features.getFeatures()
        image1_features.showFeatures()
        skimage.io.imshow(image)
        skimage.io.show()

    def featureMatchTest(self):
        # image = image_handler.readImage('datasets/Yosemite/Yosemite1.jpg')
        # image2 = image_handler.readImage('datasets/Yosemite/Yosemite2.jpg')
        image = image_handler.readImage('datasets/graf/img1.ppm')
        image2 = image_handler.readImage('datasets/graf/img2.ppm')
        # image = image_handler.readImage('datasets/wall/img5.ppm')
        # image2 = image_handler.readImage('datasets/wall/img2.ppm')
        # image = image_handler.readImage('datasets/bikes/img1.ppm')
        # image2 = image_handler.readImage('datasets/bikes/img3.ppm')
        image1_features = FeatureDetection(image)
        image2_features = FeatureDetection(image2)
        image1_features.getFeatures(response_threshold=0.3)
        image2_features.getFeatures(response_threshold=0.3)
        image1_features.showFeatures()
        image2_features.showFeatures()
        matches = FeatureMatching(image1_features, image2_features)
        matches.getMatches()
        matches.plotMatches()
        skimage.io.imshow(matches.combinedImage)
        skimage.io.imsave('graf_rotation.png', matches.combinedImage)
        skimage.io.show()


if __name__ == '__main__':
    new_test = Tests()
    # new_test.bufferSquareImageTest()
    new_test.featureMatchTest()


