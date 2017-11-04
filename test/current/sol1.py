import numpy as np
from scipy.misc import imread as imread
from scipy.interpolate import interp1d
from skimage.color import rgb2gray
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
    im_float /= PIXEL_INTENSITY_MAX

    if (representation == MODE_RGB):
        return im_float
    elif (representation == MODE_GRAYSCALE):
        return rgb2gray(im_float)
    else:
        # Whoops! shouldn't get here (invalid input)
        return None


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
    plt.imshow(im, cmap=cmap)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into YIQ color space.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imRGB, MATRIX_RGB2YIQ)


def yiq2rgb(imYIQ):
    """
    Transforms an RGB image into YIQ color space.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imYIQ, MATRIX_YIQ2RGB)


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
    yiq = rgb2yiq(im_orig)
    hist_orig, bins = np.histogram(yiq[..., INDEX_Y], 256,
                                   (0, 1))
    cdf = np.cumsum(hist_orig)
    cdf = PIXEL_INTENSITY_MAX * cdf / cdf[-1]
    #yiq_eq = np.interp(yiq, bins[:-1], cdf) / PIXEL_INTENSITY_MAX
    hist_eq, bins = np.histogram(a=yiq, bins=256)
    im_eq = yiq2rgb(yiq_eq)
    return [im_eq, hist_orig, hist_eq]


TEST_IMAGE = "external\\jerusalem.jpg"
im = read_image(TEST_IMAGE, MODE_RGB)
plt.imshow(histogram_equalize(im)[0])
plt.show()
