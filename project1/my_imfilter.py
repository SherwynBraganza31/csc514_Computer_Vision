import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

"""
    Image Convolution Filter 
    
    Author: Sherwyn Braganza
    Sept 28, 2020 - Initial Creation
    
    Function that implements convolution through correlation. Designed to work 
    like signal.convolve2D from the scipy library. The kernel used for convolution 
    is presented by the user.
    
    The function returns two np.matrix(s), one corresponding to the image convolved
    with an impulse response kernel and the other convolved with the kernel provided.
    
        @:param     image (np.matrix)   : Numpy matrix containing image data
        @:param     kernel (np.matrix)  : Numpy matrix containing data for an 
                                          odd dimension kernel
        
        @:return    impulse (np.matrix) : Impulse Convolved Version of the Image (using impulse kernel)
        @:return    filter (np.matrix)  : Kernel Convolved Version of the Image
"""
def my_imfilter(image:np.ndarray, kernel:np.ndarray) -> (np.ndarray, np.ndarray):
    if kernel.shape[0]%2 == 0 or kernel.shape[1]%2 == 1:
        raise Exception('Kernel with even dimensions provided.')

    image = np.flip(np.flip(image, axis=1),axis=0)

    impulse, filtered = im_correlate(np.matrix(image), kernel)
    impulse, filtered = np.matrix(np.flip(np.flip(impulse, axis=1),axis=0)), \
                        np.matrix(np.flip(np.flip(filtered, axis=1),axis=0))

    return impulse, filtered


"""
    Image Correlation Filter 
    
    Author: Sherwyn Braganza
    Sept 28, 2020 - Initial Creation
    
    Function that implements correlation image processing. Designed to work 
    like signal.correlate2D from the scipy library. The kernel used for correlation
    is presented by the user.
    
    The function returns two np.matrix(s), one corresponding to the image correlated
    with an impulse response kernel and the other correlated with the kernel provided.
    
        @:param     image (np.matrix)   : Numpy matrix containing image data
        @:param     kernel (np.matrix)  : Numpy matrix containing data for an 
                                          odd dimension kernel
        
        @:return    impulse (np.matrix) : Impulse Correlated Version of the Image (using impulse kernel)
        @:return    filter (np.matrix)  : Kernel Correlated Version of the Image

"""
def im_correlate(image:np.ndarray, kernel:np.ndarray) -> (np.ndarray, np.ndarray):
    image = skimage.img_as_float32(image)
    pad_row, pad_col = kernel.shape[0]//2, kernel.shape[1]//2
    padded_img = np.pad(image,
                        ((pad_row, pad_row), (pad_col,pad_col), (0,0)),
                         mode='constant',
                        constant_values=0)
    impulse = np.zeros(image.shape)
    filtered = np.zeros(image.shape)

    impulse_kernel = np.zeros(kernel.shape)
    impulse_kernel[pad_row, pad_col] = 1

    channel = 3 if len(image.shape) > 2 else 1

    for i in range(pad_row, image.shape[0]+pad_row):
        for j in range(pad_col, image.shape[1]+pad_col):
            for k in range(channel):
                impulse[i - pad_row, j - pad_col, k] = np.sum(
                    padded_img[i-pad_row:i+pad_row+1, j-pad_col:j+pad_col+1, k] * impulse_kernel
                )
                filtered[i - pad_row, j - pad_col, k] = np.sum(
                    padded_img[i-pad_row:i+pad_row+1, j-pad_col:j+pad_col+1, k] * kernel
                )

    return skimage.img_as_ubyte(impulse.clip(0,1)), skimage.img_as_ubyte(filtered.clip(0,1))


if __name__ == '__main__':
    img1 = io.imread('data/bicycle.bmp', as_gray=False)
    kernel = np.asarray([[-1,0,1],
                        [-2,0,2],
                        [-1,0,-1]])

    impulse, filtered = im_correlate(img1, kernel)
    signal_filtered = np.stack(
        (correlate2d(img1[:,:,0], kernel, mode='full'),
         correlate2d(img1[:, :, 1], kernel, mode='full'),
         correlate2d(img1[:, :, 2], kernel, mode='full')),
        axis=2).clip(0,255)

    plt.imshow(impulse)
    plt.show()
    plt.imshow(filtered, cmap='Greys')
    plt.show()
    plt.imshow(signal_filtered, cmap='Greys')
    plt.show()

