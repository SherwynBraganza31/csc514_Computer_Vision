import numpy as np
import matplotlib.pyplot as plt
import image_handler

class FeatureDetection:
    def __init__(self, image:np.ndarray = None):
        self.image = image

    def showFeatures(self, loc: tuple[int, int], scale: float = 1, orientation: float = 0) -> None:
        orientation += 90
        buffer = self.constructBufferSquare(loc, scale)
        rotated_buffer = self.rotateBufferSquare(loc, buffer, orientation)
        plotpoints = self.generatePlotPoints(loc, rotated_buffer, scale)

        for x in plotpoints:
            self.image[x[0], x[1], 0] = 255
            self.image[x[0], x[1], 1] = 0
            self.image[x[0], x[1], 2] = 0

    def constructBufferSquare(self, loc: tuple[int,int], scale: float) -> np.ndarray:
        """
            Gets 4 points around loc that corresponds to the edges of the
            buffer square. The points are listed in the bottomLeft, bottomRight,
            topRight, topLeft, bottomTopRightCenter
        :param loc:
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

    def rotateBufferSquare(self, loc: tuple[int, int], points: np.ndarray, orientation: float) -> np.ndarray:
        loc = np.asarray(loc)
        cob_points = points - loc
        theta = np.radians(orientation)
        c, s = np.cos(theta), np.sin(theta)
        rotMatrix = np.array(((c, s),
                              (-s, c)))

        cob_points = np.matmul(cob_points, rotMatrix)

        return np.rint(cob_points + loc).astype(int, casting='unsafe')

    def generatePlotPoints(self, loc: tuple[int, int], points: np.ndarray, scale) -> np.ndarray:
        plotpoints = np.asarray([0, 0])

        for i in range(len(points)):
            temp_x = np.linspace(points[i % 4, 0], points[(i+1)%4, 0], num=(scale-1)*2 + 1, endpoint=True, dtype=int)
            temp_y = np.linspace(points[i % 4, 1], points[(i+1)%4, 1], num=(scale-1)*2 + 1, endpoint=True, dtype=int)
            plotpoints = np.vstack((plotpoints, np.hstack((temp_x.reshape(-1, 1), temp_y.reshape(-1, 1)))))

        plotpoints = np.vstack((
            plotpoints,
            np.hstack((
                np.linspace(loc[0], points[-1,0], num=scale, endpoint=True, dtype=int).reshape(-1, 1),
                np.linspace(loc[1], points[-1,1], num=scale, endpoint=True, dtype=int).reshape(-1, 1),
            ))
        ))

        return np.delete(plotpoints, 0, axis=0)


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

    ########## Feature Plotter Test ###############
    image = image_handler.readImage('datasets/Yosemite/Yosemite1.jpg')
    features = FeatureDetection(image)
    features.showFeatures((88, 96), 20, 45)
    features.showFeatures((182, 44), 20, -36)
    plt.imshow(image)
    plt.show()
