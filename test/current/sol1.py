import timeit

import numpy as np
from scipy.misc import imread as imread
from scipy.interpolate import interp1d
from skimage.color import rgb2gray, rgb2yiq as skimage_rgb2yiq, yiq2rgb as \
    skimage_yiq2rgb
from matplotlib import pyplot as plt

# Constants
PIXEL_INTENSITY_MAX = 255
PIXEL_INTENSITIES = PIXEL_INTENSITY_MAX + 1
PIXEL_INTENSITIES_LIST = [i for i in range(0, PIXEL_INTENSITY_MAX + 1)]
PIXEL_INTENSITIES_NORMALIZED = [i / PIXEL_INTENSITY_MAX for i in
                                PIXEL_INTENSITIES_LIST]
PIXEL_RANGE = (0, PIXEL_INTENSITY_MAX)
PIXEL_RANGE_NORMALIZED = (0, 1)

# Picture representation modes
MODE_GRAYSCALE = 1
MODE_RGB = 2

# Magical matrices
MATRIX_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
MATRIX_YIQ2RGB = np.linalg.inv(MATRIX_RGB2YIQ)
INDEX_Y = 0


# Methods
def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB
                            image (2).
    :return:                returns an image represented by a matrix of type
    np.float64 with
                            intensities normalized to the range [0,1]
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    if (representation == MODE_GRAYSCALE):
        im_float = rgb2gray(im_float)
    return im_float / PIXEL_INTENSITY_MAX


def display_image(image, cmap=None):
    clipped = np.clip(image[:, :, :], 0, 1)
    plt.imshow(clipped, cmap=cmap)
    plt.show()


def imdisplay(filename, representation):
    """
    reads an image file and displays the loaded image in converted
    representation.
    figure.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB
                            image (2).
    """
    im = read_image(filename, representation)
    cmap = None
    if (representation == MODE_GRAYSCALE):
        cmap = plt.cm.gray
    display_image(im, cmap)


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into YIQ color space.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imRGB, MATRIX_RGB2YIQ.T)


def yiq2rgb(imYIQ):
    """
    Transforms an RGB image into YIQ color space.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imYIQ, MATRIX_YIQ2RGB.T)


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: Grayscale or RGB float64 image with values in [0, 1].
    :return:        list [im_eq, hist_orig, hist_eq] where:
                    im_eq - is the equalized image. grayscale or RGB float64
                    image with values in [0, 1].
                    hist_orig - is a 256 bin histogram of the original image
                    (array with shape (256,) ).
                    hist_eq - is a 256 bin histogram of the equalized image
                    (array with shape (256,) ).
    """
    # Convert image to YIQ for equalizing Y values
    yiq = skimage_rgb2yiq(im_orig) * PIXEL_INTENSITY_MAX
    y = yiq[..., INDEX_Y].astype(np.uint8)

    # Build cumulative histogram of Y values
    hist_orig, bins = np.histogram(y, 256, (0, 255))
    cdf = np.cumsum(hist_orig)
    cdf_n = (cdf * PIXEL_INTENSITY_MAX / cdf[-1]).astype(np.uint8)

    # Map original y values to their equalized values
    y_eq = np.array(list(map(lambda y: cdf_n[y], y))).astype(np.uint8)

    # Calculate histogram of equalized Y values
    hist_eq, bins = np.histogram(y_eq, 256)

    # Update Y with equalized values
    yiq[..., INDEX_Y] = y_eq

    # Back to RGB
    im_eq = skimage_yiq2rgb(yiq / PIXEL_INTENSITY_MAX)

    # Done!
    return [im_eq, hist_orig, hist_eq]


TEST_IMAGE = "external\\Low Contrast.jpg"
im = read_image(TEST_IMAGE, MODE_RGB)
im_eq = histogram_equalize(im)[0]
display_image(im_eq)
