import time
import matplotlib.pyplot as plt
import skimage.transform
import my_imfilter
from scipy.signal import convolve2d
from skimage import io
import numpy as np

if __name__ == '__main__':
    kernel_stack = [my_imfilter.generateGaussianKernel(0.25)]
    image_stack = [io.imread('data/bird.bmp', as_gray=False)]
    max_step = 7
    time_convolutional = np.zeros((max_step, max_step))
    time_fourier = np.zeros((max_step, max_step))

    for i in range(1,max_step):
        image_stack.append(skimage.transform.rescale(image_stack[0], (1/2*i), anti_aliasing=True, channel_axis=2))
        kernel_stack.append(my_imfilter.generateGaussianKernel(0.25*(i+1)))

    # for idx1, x in enumerate(image_stack):
    #     for idx2, y in enumerate(kernel_stack):
    #         begin = time.time()
    #         for i in range(3):
    #             convolve2d(x[:, :, i], y)
    #         end = time.time()
    #         time_convolutional[idx1, idx2] = 1000 *(begin-end)
    #
    #         begin = time.time()
    #         my_imfilter.fourierDomain(x, y)
    #         end = time.time()
    #         time_convolutional[idx1, idx2] = 1000 *(bpeegin-end)

    for x in image_stack:
        begin = time.time()
        for i in range(3):
            convolve2d(x[:,:,i], kernel_stack[1])
        end = time.time()
        time_image_convolutional.append(1000*(end-begin))

        begin = time.time()
        my_imfilter.fourierDomain(x, kernel_stack[1])
        end = time.time()
        time_image_fourier.append(1000*(end-begin))

    for x in kernel_stack:
        begin = time.time()
        for i in range(3):
            convolve2d(image_stack[-3][:,:,i], x)
        my_imfilter.imConvolute(image_stack[4], x)
        end = time.time()
        time_kernel_convolutional.append(1000*(end-begin))

        begin = time.time()
        my_imfilter.fourierDomain(image_stack[-2], x)
        end = time.time()
        time_kernel_fourier.append(1000*(end-begin))

    fig, axs = plt.subplots(2)
    fig.tight_layout(h_pad=2)
    plt.subplots_adjust(top=0.9)
    image_sizes = (['8', '4', '2', '1', '0.5', '0.25', '0.125'])[-1::-1]
    axs[0].plot(image_sizes, time_image_convolutional, color='r',
                label='convolution')
    axs[0].plot(image_sizes, time_image_fourier, color='g', label='fourier')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('Time (in ms)')
    axs[0].set_xlabel('Image Sizes in MPix')
    kernel_sizes = ['3x3','5x5','7x7','9x9','11x11','13x13','15x15']
    axs[1].plot(kernel_sizes, time_kernel_convolutional, label='convolution')
    axs[1].plot(kernel_sizes, time_kernel_fourier, label='fourier')
    axs[1].legend(loc='upper right')
    axs[1].set_ylabel('Time (in ms)')
    axs[1].set_xlabel('Kernel Sizes')
    plt.savefig('metrics.jpg', dpi=300)
    plt.show()








