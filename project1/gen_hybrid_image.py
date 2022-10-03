from my_imfilter import hybridise
import skimage.io as io
import os

def generateCog():
    img2 = io.imread('data/cat.bmp', as_gray=False)
    img1 = io.imread('data/dog.bmp', as_gray=False)
    new_image = hybridise(img1, img2, sigma1=8, sigma2=3, fourier=False)
    io.imsave('hybrid_images/cog.jpg', new_image)


def generateAlbertMonroe():
    img2 = io.imread('data/einstein.bmp', as_gray=True)
    img1 = io.imread('data/marilyn.bmp', as_gray=True)
    new_image = hybridise(img1, img2, sigma1=4, sigma2=2, fourier=True)
    io.imsave('hybrid_images/albert_monroe.jpg', new_image)

def generateBiMotorCycle():
    img2 = io.imread('data/bicycle.bmp', as_gray=False)
    img1 = io.imread('data/motorcycle.bmp', as_gray=False)
    new_image = hybridise(img1, img2, sigma1=5, sigma2=2, fourier=True)
    io.imsave('hybrid_images/bimotorcycle.jpg', new_image)

def generateFishMarine():
    img2 = io.imread('data/submarine.bmp', as_gray=False)
    img1 = io.imread('data/fish.bmp', as_gray=False)
    new_image = hybridise(img1, img2, sigma1=5, sigma2=4, fourier=True)
    io.imsave('hybrid_images/fishmarine.jpg', new_image)

def generatePlird():
    img2 = io.imread('data/plane.bmp', as_gray=False)
    img1 = io.imread('data/bird.bmp', as_gray=False)
    new_image = hybridise(img1, img2, sigma1=5, sigma2=4, fourier=True)
    io.imsave('hybrid_images/plird.jpg', new_image)


if __name__ == '__main__':
    if not os.path.exists('hybrid_images'):
        os.makedirs('hybrid_images')

    generateCog()
    generateAlbertMonroe()
    generateBiMotorCycle()
    generateFishMarine()
    generatePlird()

