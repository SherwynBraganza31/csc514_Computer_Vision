import time
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage.transform
import my_imfilter
from scipy.signal import convolve2d
from skimage import io
import numpy as np

if __name__ == '__main__':
    kernel_stack = [my_imfilter.generateGaussianKernel(0.25)]
    image_stack = [io.imread('data/gigi.jpg', as_gray=False)]
    max_step = 7
    time_convolutional = np.zeros((max_step, max_step))
    time_fourier = np.zeros((max_step, max_step))

    for i in range(1,max_step):
        image_stack.append(skimage.transform.rescale(image_stack[0], (1/2*i), anti_aliasing=True, channel_axis=2))
        kernel_stack.append(my_imfilter.generateGaussianKernel(0.25*(i+1)))

    for idx1, x in enumerate(image_stack):
        for idx2, y in enumerate(kernel_stack):
            begin = time.time()
            for i in range(3):
                convolve2d(x[:, :, i], y)
            end = time.time()
            time_convolutional[idx1, idx2] = 1000 *(end-begin)

            begin = time.time()
            my_imfilter.fourierDomain(x, y)
            end = time.time()
            time_fourier[idx1, idx2] = 1000 *(end-begin)

    fig, axs = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    fig.tight_layout(h_pad=2)
    plt.subplots_adjust(top=0.9)
    image_sizes = (['8', '4', '2', '1', '0.5', '0.25', '0.125'])[-1::-1]
    kernel_sizes = ['3x3','5x5','7x7','9x9','11x11','13x13','15x15']
    x,y = np.meshgrid(np.arange(1,8), np.arange(1,8))
    ax.scatter(x,y, time_convolutional, color='g', label='Convolution')
    ax.scatter(x,y, time_fourier, color='r', label='Fourier')
    ax.legend()
    ax.set_xticks(np.arange(1,8), image_sizes)
    ax.set_yticks(np.arange(1,8), kernel_sizes)
    ax.set_xlabel('Image Sizes in MPix')
    ax.set_ylabel('Square Kernel Sizes')
    ax.set_zlabel('Time in ms')
    plt.savefig('metrics.jpg', dpi=300)
    plt.show()








