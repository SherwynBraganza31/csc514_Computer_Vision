import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt


def my_imfilter(image: np.ndarray, kernel: np.ndarray) -> (np.ndarray, np.ndarray):
    """
        Image Convolution Filter

        Author: Sherwyn Braganza
        Sept 28, 2020 - Initial Creation
        Sept 28, 2020 - Added more descriptive comments.
        Sept 28, 2020 - Finished off most of convolve
        Sept 28, 2020 - Compared results with signal.convolve2D

        Function that implements convolution through correlation. Designed to work
        like signal.convolve2D from the scipy library. The kernel used for convolution
        is presented by the user. This function is designed to handle both rgb and grayscale images.

        It exploits the fact that convolution is basically correlation using a flipped kernel.
        It therefore flips the kernel and passes it to the function that gives the correlational
        version of an image with the flipped kernel.

        The function returns two np.matrix(s), one corresponding to the image convolved
        with an impulse response kernel and the other convolved with the kernel provided.

            @:param     image (np.matrix)   : Numpy matrix containing image data
            @:param     kernel (np.matrix)  : Numpy matrix containing data for an
                                              odd dimension kernel

            @:return    impulse (np.matrix) : Impulse Convolved Version of the Image (using impulse kernel)
            @:return    filter (np.matrix)  : Kernel Convolved Version of the Image
    """
    if kernel.shape[0]%2 == 0 or kernel.shape[1]%2 == 0:
        raise Exception('Kernel with even dimensions provided.')

    # flip the kernel along rows and cols
    kernel = np.flip(np.flip(kernel, axis=1),axis=0)

    # call correlation function and get the kernel correlated image as well as the impulse version of it
    impulse, filtered = im_correlate(image, kernel)

    return impulse, filtered


def im_correlate(image: np.ndarray, kernel: np.ndarray) -> (np.ndarray, np.ndarray):
    """
        Image Correlation Filter

        Author: Sherwyn Braganza
        Sept 28, 2020 - Initial Creation
        Sept 28, 2020 - Added more descriptive comments.
        Sept 28, 2020 - Finished off most of correlate
        Sept 28, 2020 - Compared results with signal.correlate2D

        Function that implements correlation image processing. Designed to work
        like signal.correlate2D from the scipy library. The kernel used for correlation
        is presented by the user.

        This function convolves the kernel with the image using matrix convolution. It
        initially converts the image to a float and then pads it with 0s along the borders
        according to the shape of the convolutional kernel. It finally clips the image pixel values to [0,1]
        before converting it back to ubyte format and returning it.

        The function returns two np.matrix(s), one corresponding to the image correlated
        with an impulse response kernel and the other correlated with the kernel provided.

        TODO - Try to use logical indexing insted of loops

            @:param     image (np.matrix)   : Numpy matrix containing image data
            @:param     kernel (np.matrix)  : Numpy matrix containing data for an
                                              odd dimension kernel

            @:return    impulse (np.matrix) : Impulse Correlated Version of the Image (using impulse kernel)
            @:return    filter (np.matrix)  : Kernel Correlated Version of the Image

    """
    if kernel.shape[0]%2 == 0 or kernel.shape[1]%2 == 0:
        raise Exception('Kernel with even dimensions provided.')

    image = skimage.img_as_float32(image) # convert to floats in [0,1] to make computations uniform

    # Padding section
    pad_row, pad_col = kernel.shape[0]//2, kernel.shape[1]//2 # calculate pad_width for rows and cols
    padded_img = np.pad(image,
                        ((pad_row, pad_row), (pad_col,pad_col), (0,0)),
                         mode='constant',
                        constant_values=0) # pad image along rows and cols but not channels (if it exists)

    # create containers for the impulse and kernel filtered image
    impulse = np.zeros(image.shape)
    filtered = np.zeros(image.shape)

    # create impulse kernel to be identical to fed in kernel in terms of shape.
    # It doesn't need to be this way but I like to keep things uniform
    impulse_kernel = np.zeros(kernel.shape)
    impulse_kernel[pad_row, pad_col] = 1

    # Check if it is an rgb or grayscale img
    channel = 3 if len(image.shape) > 2 else 1

    for i in range(pad_row, image.shape[0]+pad_row):
        for j in range(pad_col, image.shape[1]+pad_col):
            for k in range(channel):
                impulse[i - pad_row, j - pad_col, k] = np.sum(
                    padded_img[i-pad_row:i+pad_row+1, j-pad_col:j+pad_col+1, k] * impulse_kernel)  # convolution step
                filtered[i - pad_row, j - pad_col, k] = np.sum(
                    padded_img[i-pad_row:i+pad_row+1, j-pad_col:j+pad_col+1, k] * kernel)  # convolution step

    # clip images and convert them back to ubytes before returning
    return skimage.img_as_ubyte(impulse.clip(0,1)), skimage.img_as_ubyte(filtered.clip(0,1))


if __name__ == '__main__':
    img1 = io.imread('data/bicycle.bmp', as_gray=False)
    sobel = np.asarray([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    emboss = np.array([[-2, -1, 0],
                        [-1, 1, 1],
                        [0, 1, 2]])
    boxblur = (1 / 9.0) * np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])

    gaussian_1D = (1/36)*np.asarray([1,2,3,4,5,6,5,4,3,2,1]).reshape(-1,1)
    gaussian = gaussian_1D.T * gaussian_1D

    img_sobel = my_imfilter(img1, sobel)[1]
    img_sharpen = my_imfilter(img1, sharpen)[1]
    img_emboss = my_imfilter(img1, emboss)[1]
    img_boxblur = my_imfilter(img1, boxblur)[1]
    img_gaussian = my_imfilter(img1, gaussian)[1]

    joined = np.hstack((img1, img_sobel, img_sharpen, img_emboss, img_boxblur, img_gaussian))
    plt.title('Original --> Sobel --> Sharpen --> Emboss --> Boxblur --> Gaussian')
    plt.imshow(joined)
    plt.show()
