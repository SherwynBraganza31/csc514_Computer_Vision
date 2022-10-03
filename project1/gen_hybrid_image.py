from my_imfilter import hybridise
import skimage.io as io
import os

def generateCog():
    img2 = io.imread('data/cat.bmp', as_gray=False)
    img1 = io.imread('data/dog.bmp', as_gray=False)
    new_image = hybridise(img1, img2, sigma1=8, sigma2=3, fourier=False)
    io.imsave('hybrid_images/cog.jpg', new_image)


def generateAlbertMonroe():
    img2 = io.imread('data/einstein.bmp', as_gray=False)
    img1 = io.imread('data/marilyn.bmp', as_gray=False)
    new_image = hybridise(img1, img2, sigma1=3, sigma2=2, fourier=False)
    io.imsave('hybrid_images/albert_monroe.jpg', new_image)


if __name__ == '__main__':
    if not os.path.exists('hybrid_images'):
        os.makedirs('hybrid_images')

    generateCog()
    generateAlbertMonroe()

