import time
import matplotlib.pyplot as plt
import skimage.transform
import my_imfilter
from skimage import io

if __name__ == '__main__':
    kernel_stack = []
    image_stack = [io.imread('data/gigi.jpg', as_gray=False)]
    time_kernel_convolutional = []
    time_kernel_fourier = []
    time_image_convolutional = []
    time_image_fourier = []

    for i in range(1,8):
        image_stack.append(skimage.transform.rescale(image_stack[0], (1/2*i), anti_aliasing=True, channel_axis=2))
        kernel_stack.append(my_imfilter.generateGaussianKernel(0.25*i))

    for x in image_stack:
        begin = time.time()
        my_imfilter.imConvolute(x, kernel_stack[1])
        end = time.time()
        time_image_convolutional.append(end-begin)

        begin = time.time()
        my_imfilter.fourierDomain(x, kernel_stack[1])
        end = time.time()
        time_image_fourier.append(end - begin)

    for x in kernel_stack:
        begin = time.time()
        my_imfilter.imConvolute(image_stack[4], x)
        end = time.time()
        time_kernel_convolutional.append(end - begin)

        begin = time.time()
        my_imfilter.fourierDomain(image_stack[-2], x)
        end = time.time()
        time_kernel_fourier.append(end - begin)

    fig, axs = plt.subplots(2,1)
    fig.tight_layout(h_pad=2)
    plt.subplots_adjust(top=0.9)
    axs[0,1].plot(range(1,7), time_image_convolutional)
    axs[0, 1].plot(range(1, 7), time_image_fourier[-1::-1])
    fig.set_title('Image Size vs. Processing time')
    axs.set_ylabel('Time (in ms)')
    axs.set_xlabel('Image Sizes')
    plt.savefig('imagesizes.jpg', dpi=300)
    plt.show()

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(h_pad=2)
    plt.subplots_adjust(top=0.9)
    axs[0, 1].plot(range(1, 7), time_image_convolutional)
    axs[0, 1].plot(range(1, 7), time_image_fourier[-1::-1])
    fig.set_title('Image Size vs. Processing time')
    axs.set_ylabel('Time (in ms)')
    axs.set_xlabel('Kernel Sizes')
    plt.savefig('kernelsizes.jpg', dpi=300)
    plt.show()








