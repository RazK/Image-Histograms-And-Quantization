import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.color import rgb2gray

# Constants
PIXEL_INTENSITY_MAX = 255
PIXEL_INTENSITIES = PIXEL_INTENSITY_MAX + 1
PIXEL_RANGE = (0, PIXEL_INTENSITIES)
PIXEL_RANGE_NORMALIZED = (0, 1)
PIXEL_CHANNELS_RGB = 3
PIXEL_CHANNELS_RGBA = 4

# Picture representation modes
MODE_GRAYSCALE = 1
MODE_RGB = 2

# Magic matrices
MATRIX_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
MATRIX_YIQ2RGB = np.linalg.inv(MATRIX_RGB2YIQ)

# Picture dimension roperties
DIMENSIONS_GREYSCALE = 2  # dimensions of GREYSCALE image
DIMENSIONS_RGB = 3  # dimensions of RGB image

# RGB, RGBA properties
RED = 0
GREEN = 1
BLUE = 2
ALPHA = 3
RGB_CHANNELS = [RED, GREEN, BLUE]
RGBA_CHANNELS = [RED, GREEN, BLUE, ALPHA]

# YIQ properties
Y = 0
I = 1
Q = 2
YIQ_CHANNELS = [Y, I, Q]


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB image (2).
    :return:                returns an image represented by a matrix of type
                            .float64 with intensities normalized to the
                            range [0,1]
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    if (representation == MODE_GRAYSCALE):
        im_float = rgb2gray(im_float)
    return im_float / PIXEL_INTENSITY_MAX


def display_image(image, cmap=None):
    clipped = np.clip(image[..., :], 0, 1)
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


def is_greyscale(image):
    """
    Returns true if image is greyscale, otherwise false.
    :param image:   an image to check
    :return:        true if image is greyscale, otherwise false
    """
    # Image is just (width * height) with no pixel axis
    return (len(image.shape) == DIMENSIONS_GREYSCALE)


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into YIQ color space.
    Removes alpha channel if exists.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imRGB[:, :, :len(RGB_CHANNELS)], MATRIX_RGB2YIQ.T)


def yiq2rgb(imYIQ):
    """
    Transforms an RGB image into YIQ color space.
    Removes alpha channel if exists.
    :param imRGB:   an RGB image.
    :return:        an YIQ image with the same dimensions as the input.
    """
    return np.dot(imYIQ[:, :, :len(RGB_CHANNELS)], MATRIX_YIQ2RGB.T)


def intensity_histogram_translated(intensities, translate=True):
    """
    Returns a histogram of the given intensities translated to [0, 255] range.
    Translates the intensities from [0,1] to [0, 255] if translate=True.
    :param intensities: An array of intensities, in [0, 1] range by default.
    :param translate:   Flag if the intensities are given in [0, 1] range.
                        Should be False if the given intensities are already
                        in [0, 255] range.
    :return:    A histogram of the intensities, in [0, 255] range.
    """
    # Translate [0,1] intensity range to [0,255] integer range
    if (translate):
        intensities = np.round(intensities * PIXEL_INTENSITY_MAX).astype(
            np.uint8)

    # Build cumulative histogram of pixel intensities
    hist, bins = np.histogram(a=intensities, bins=PIXEL_INTENSITIES,
                              range=PIXEL_RANGE)

    # Don't give a damn about bins, return hist
    return hist
