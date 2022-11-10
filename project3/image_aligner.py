import numpy as np
from matplotlib import pyplot as plt
import os
import skimage

def loadImage(folder_name) -> list[np.ndarray]:
    image_array = []

    if os.path.exists(folder_name):
        os.chdir(folder_name)
        fnames = os.listdir()
        for x in fnames[1:]:
            image_array.append(skimage.io.imread(x))

        return image_array

    else:
        print('Directory name doesn\'t exist')

def grabUsrClicks(image1, image2):
    joined_image = np.hstack((image1, image2))
    plt.imshow(joined_image)
    click_locs = plt.ginput(-1, timeout=120, show_clicks=True)
    image1_clicks, image2_clicks = [], []

    for x in click_locs:
        shape = image1.shape
        if x[0] >= shape[1]:
            # image2_clicks.append((shape[0] - x[1], x[0]%shape[1]))
            image2_clicks.append((x[0]%shape[1], x[1]))
        else:
            # image1_clicks.append((shape[0] - x[1], x[0]%shape[1]))
            image1_clicks.append((x[0]%shape[1], x[1]))

    return image1_clicks, image2_clicks

def computeHomography(image1_pts, image2_pts):
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

def testHomography(h_matrix, image1, image2, image1_pts, image2_pts):
    squared_errors = []
    projected_pts = []
    for idx, pt in enumerate(image2_pts):
        temp_p_matrix = np.asarray([pt[0], pt[1], 1]).reshape(3,1)
        pt_proj = np.matmul(h_matrix, temp_p_matrix)
        pt_proj = pt_proj/pt_proj[-1]
        projected_pts.append(pt_proj)
        squared_errors.append(np.sum(np.abs(pt_proj[0:2]-image1_pts[idx][0:2])))

    fig, ax = plt.subplots()
    ax.imshow(image1)
    for i in range(0, len(image1_pts)):
        ax.scatter(image1_pts[i][0], image1_pts[i][1], marker='^')
        ax.scatter(projected_pts[i][0], projected_pts[i][1], marker='x')
    plt.show()
    return np.mean(squared_errors)

# def forwardWarp(image1, image2, p, p2):
#
#     for x in p2:
#         image1[]


if __name__ == '__main__':
    images = loadImage('P2_Benchmarks/DP_3')
    clicks = grabUsrClicks(images[2], images[3])
    print(clicks)

    # clicks = ([(3061.62, 1178.5107526881718),
    #            (1644.1576344086025, 1041.6218637992831),
    #            (1635.3260931899645, 706.0232974910393),
    #            (2558.2221505376347, 1138.768817204301),
    #            (2456.659426523297, 1182.926523297491),
    #            (2010.666594982079, 1540.6039426523296),
    #            (2138.72394265233, 3328.9910394265225),
    #            (2227.03935483871, 3567.442652329749),
    #            (1105.4336200716848, 2701.9516129032254),
    #            (1776.6307526881724, 560.3028673835126),
    #            (655.0250179211475, 1350.725806451613),
    #            (319.42645161290375, 2935.9874551971325),
    #            (429.82071684587845, 1973.3494623655913)],
    #           [(2990.967670250896, 2388.431899641577),
    #            (1604.4156989247313, 2304.532258064515),
    #            (1560.2579928315413, 1977.7652329749108),
    #            (2496.4013620071682, 2361.9372759856633),
    #            (2408.0859498207883, 2406.094982078852),
    #            (1953.2615770609318, 2777.0197132616486),
    #            (2138.72394265233, 4583.069892473119),
    #            (2253.5339784946236, 4825.937275985663),
    #            (1083.35476702509, 3978.109318996416),
    #            (1705.9784229390684, 1832.0448028673836),
    #            (597.6200000000003, 2618.0519713261638),
    #            (292.93182795698976, 4229.808243727599),
    #            (394.4945519713265, 3240.6756272401435)])

    h_matrix = computeHomography(clicks[0], clicks[1])
    print(testHomography(h_matrix, images[2], images[3], clicks[0], clicks[1]))

