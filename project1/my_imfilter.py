import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt


def my_imfilter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
        Image Convolution Filter

        Author: Sherwyn Braganza
        Sept 28, 2022 - Initial Creation
        Sept 28, 2022 - Added more descriptive comments.
        Sept 28, 2022 - Finished off most of convolve
        Sept 28, 2022 - Compared results with signal.convolve2D
        Sept 30, 2022 - Changes made to return only filtered image

        Function that implements convolution through correlation. Designed to work
        like signal.convolve2D from the scipy library. The kernel used for convolution
        is presented by the user. This function is designed to handle both rgb and grayscale images.

        It exploits the fact that convolution is basically correlation using a flipped kernel.
        It therefore flips the kernel and passes it to the function that gives the correlational
        version of an image with the flipped kernel.

        The function returns an np.ndarray, corresponding to the image correlated with the kernel provided.

            :param image: Numpy matrix containing image data
            :param kernel: Numpy matrix containing data for an
                                              odd dimension kernel

            :return filter: Kernel Convolved Version of the Image
    """
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise Exception('Kernel with even dimensions provided.')

    # flip the kernel along rows and cols
    kernel = np.flip(np.flip(kernel, axis=1), axis=0)

    # call correlation function and get the kernel correlated image as well as the impulse version of it
    filtered = im_correlate(image, kernel)

    return filtered


def im_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
        Image Correlation Filter

        Author: Sherwyn Braganza
        Sept 28, 2022 - Initial Creation
        Sept 28, 2022 - Added more descriptive comments.
        Sept 28, 2022 - Finished off most of correlate
        Sept 28, 2022 - Compared results with signal.correlate2D
        Sept 30, 2022 - Changes made to return only filtered image
        Setp 30, 2022 - Increased compatibility with grayscales

        Function that implements correlation image processing. Designed to work
        like signal.correlate2D from the scipy library. The kernel used for correlation
        is presented by the user.

        This function convolves the kernel with the image using matrix convolution. It
        initially converts the image to a float and then pads it with 0s along the borders
        according to the shape of the convolutional kernel. It finally clips the image pixel values to [0,1]
        before converting it back to ubyte format and returning it.

        The function returns an np.ndarray, corresponding to the image correlated with the kernel provided.

        TODO - Try to use logical indexing insted of loops

            :param image: Numpy matrix containing image data
            :param kernel: Numpy matrix containing data for an odd dimension kernel

            :return filter: Kernel Correlated Version of the Image
    """
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise Exception('Kernel with even dimensions provided.')

    # if grayscale or colored
    color = True if len(image.shape) > 2 else False

    image = skimage.img_as_float32(image)  # convert to floats in [0,1] to make computations uniform

    # if grayscale, create a third dimension with only one channel
    if not color:
        image = image.reshape(image.shape[0],image.shape[1],1)

    # Padding section
    pad_row, pad_col = kernel.shape[0] // 2, kernel.shape[1] // 2  # calculate pad_width for rows and cols
    pad_params = ((pad_row, pad_row), (pad_col, pad_col), (0, 0))
    padded_img = np.pad(image,
                        pad_params,
                        mode='constant',
                        constant_values=0)  # pad image along rows and cols but not channels (if it exists)

    # create a container for the kernel filtered image
    filtered = np.zeros(image.shape)

    # Check if it is an rgb or grayscale img
    channel = 3 if color else 1

    for i in range(pad_row, image.shape[0] + pad_row):
        for j in range(pad_col, image.shape[1] + pad_col):
            for k in range(channel):
                filtered[i - pad_row, j - pad_col, k] = np.sum(
                    padded_img[i - pad_row:i + pad_row + 1, j - pad_col:j + pad_col + 1,
                    k] * kernel)  # convolution step

    # clip images and convert them back to ubytes before returning

    return skimage.img_as_ubyte(filtered.clip(0, 1)) if color \
        else skimage.img_as_ubyte(filtered.clip(0, 1))[:, :, 0]


def hybridise(image1: np.ndarray, image2: np.ndarray, sigma1: int, sigma2: int, fourier: bool):
    """
        Hybrid Image Generator
        Generates a hybrid image according to the process described by Oliva, Torralba and Schyns
        in Siggraph(2006) (http://olivalab.mit.edu/hybrid/Talk_Hybrid_Siggraph06.pdf).

        The basis of this process is to take the high pass version of one image and
        superimpose it on the low pass version of the other. These images have to be normalized
        and centered for best results. This function implements the same process
        using 2 approaches - Spatial Convolution and Fourier Domain Multiplication (Hadamard Product)

        The function generates 2 different Gaussian Filters based on the sigmas provided and
        uses them to get a low pass and high pass filtered images

        The generated hybrid image is rescaled 4 times and stacked horizontally to get the
        final image that is then returned.

        Author: Sherwyn Braganza
        29 Sept, 2022 - Implemented a rudimentary hybrid image generator
                        that uses Spatial Convolution
        30 Sept, 2022 - Implemented hybrid image rescaling and stacking
        30 Sept, 2022 - Expanded it to use Spatial Convolution and
                        Fourier Domain Hadamard (fourier domain still need to be coded up)

        :param image1: The low pass intended image
        :param image2: The high pass intended image
        :param sigma1: The sigma value corresponding to the low pass image
        :param sigma2: The sigma value corresponding to the high pass image
        :param fourier: False if you want to use Spatial Convolution, True
                        if you want to use Fourier Domain Processing

        :return: The stacked hybrid image
    """
    hybrid = np.asarray([])
    gaussian_low = generateGaussianKernel(sigma1)
    gaussian_high = generateGaussianKernel(sigma2)

    if not fourier:
        lowpass_image = my_imfilter(image1, gaussian_low)
        lowpass_image2 = my_imfilter(image2, gaussian_high)

        hybrid = skimage.img_as_ubyte((
                                      skimage.img_as_float32(lowpass_image) +
                                      skimage.img_as_float32(image2) -
                                      skimage.img_as_float32(lowpass_image2)
                                      ).clip(0, 1))
    #else:
        # TODO - Implement fourier based hybrid image generation



    #######################################
    # Image stacking and padding section
    #######################################
    image_stack = [hybrid]

    # create scale down versions of the original
    for i in range(0, 4):
        image_stack.append(skimage.img_as_ubyte(
            skimage.transform.rescale(image_stack[i], 0.5, anti_aliasing=True, channel_axis=2)
        ))

    # padding the rescaled images along axis 0 to make them the same size vertically
    for i in range(1, 5):
        image_stack[i] = np.pad(
            image_stack[i],
            ((image_stack[0].shape[0] - image_stack[i].shape[0], 0),
             (0, 0),
             (0, 0)),
            mode='constant',
            constant_values=255
        )

    # padding along axis 1
    for i in range(1, 5):
        image_stack[i] = np.pad(
            image_stack[i],
            ((0, 0),
             (5, 0),
             (0, 0)),
            mode='constant',
            constant_values=255
        )

    return np.hstack(image_stack)


def generateGaussianKernel(sigma: float) -> np.ndarray:
    """
        Generates a 2-D Gaussian Kernel

        Author: Sherwyn Braganza
        Sept 29, 2020 - Added function and base code for generating it
        Sept 29, 2020 - Implemented a weighted mean based kernel generator
        Sept 30, 2020 - Changed implementation to generate a true gaussian
                        based kernel

        Generates a 1D Gaussian distribution from the Gaussian equation
        using the value of sigma. Matrix multiplies the transpose of
        the 1D Gaussian with itself to form a 2D square Gaussian kernel

        :param sigma: The standard deviation of the gaussian
        :return: kernel: The 2D Gaussian kernel generated
    """
    size = int(8 * sigma + 1)
    # enforce an odd sized kernel
    if not size % 2:
        size = size + 1

    center = size // 2
    kernel = np.zeros(size)

    # Generate Gaussian blur.
    for x in range(size):
        diff = (x - center) ** 2
        kernel[x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = np.asarray(kernel.reshape(-1, 1).T * kernel.reshape(-1, 1))
    kernel = kernel / np.sum(kernel)

    return kernel


if __name__ == '__main__':

    img1 = io.imread('data/dog.bmp', as_gray=False)
    img2 = skimage.img_as_ubyte(io.imread('data/marilyn.bmp', as_gray=True))
    impulse = np.asarray([[0,0,0],
                          [0,1,0],
                          [0,0,0]])
    sobel = np.asarray([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    emboss = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])

    gaussian = generateGaussianKernel(0.75)

    img_impulse = my_imfilter(img1, impulse)
    img_sobel = my_imfilter(img1, sobel)
    img_sharpen = my_imfilter(img1, sharpen)
    img_emboss = my_imfilter(img1, emboss)
    img_gaussian = my_imfilter(img1, gaussian)

    joined = np.hstack((img1, img_impulse, img_sobel, img_sharpen, img_emboss, img_gaussian))
    plt.title('Original --> Impulse --> Sobel --> Sharpen --> Emboss -->Gaussian')
    plt.imshow(joined)
    skimage.io.imsave('my_filter_colored.jpg', joined)
    plt.show()

    img_impulse = my_imfilter(img2, impulse)
    img_sobel = my_imfilter(img2, sobel)
    img_sharpen = my_imfilter(img2, sharpen)
    img_emboss = my_imfilter(img2, emboss)
    img_gaussian = my_imfilter(img2, gaussian)

    joined = np.hstack((img2, img_impulse, img_sobel, img_sharpen, img_emboss, img_gaussian))
    plt.title('Original --> Impulse --> Sobel --> Sharpen --> Emboss --> Gaussian')
    plt.imshow(joined, cmap='gray')
    skimage.io.imsave('my_filter_gray.jpg', joined)
    plt.show()


    # new_image = hybridise(img1, img2, sigma1=7, sigma2=5, fourier=False)
    # plt.imsave('hybrid.jpg', new_image)
