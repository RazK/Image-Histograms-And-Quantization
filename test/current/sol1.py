import numpy as np
from builtins import len
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from scipy.misc import imread as imread

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


# Methods
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
    if (representation == MODE_GRAYSCALE):
        im = rgb2gray(imread(filename))
    else:
        im = imread(filename, "RGB")
    im_float = im.astype(np.float64)
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


def intensity_equalize(intensities):
    """
    Performs histogram equalization of the given pixel intensities.
    :param intensities: float64 image intensities with values in [0, 1].
    :return:            list [im_eq, hist_orig, hist_eq] where:
                        intensity_eq - is the equalized intensities. np.array
                        with values in [0, 1].
                        hist_orig - is a 256 bin histogram of the original
                        intensities (array with shape (256,) ).
                        hist_eq - is a 256 bin histogram of the
                        equalized intensities (array with shape (256,) ).
    """
    # Translate [0,1] intensity range to [0,255] integer range
    translated_intensities = np.round(
        intensities * PIXEL_INTENSITY_MAX).astype(
        np.uint8)

    # Build cumulative histogram of pixel intensities
    hist_orig = intensity_histogram_translated(translated_intensities, False)
    cdf = np.cumsum(hist_orig)

    # Normalize cumulative histogram:
    # C[k] = round(((C[k] / NUM_OF_PIXELS) * PIXEL_INTENSITY_MAX))
    cdf_n = (cdf * PIXEL_INTENSITY_MAX / cdf[-1]).astype(np.uint8)

    # Map original intensity values to their equalized values
    intensity_eq = np.array(list(map(lambda i: cdf_n[i],
                                     translated_intensities))).astype(np.uint8)

    # Calculate histogram of equalized intensity values
    hist_eq, bins = np.histogram(intensity_eq, PIXEL_INTENSITIES)

    return (intensity_eq / PIXEL_INTENSITY_MAX), hist_orig, hist_eq


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
    # GREYSCALE Image
    if (is_greyscale(im_orig)):
        # Equalize image intensities
        intensities_eq, hist_orig, hist_eq = intensity_equalize(im_orig)
        im_eq = intensities_eq

    # RGB Image
    else:
        # Convert RGB [0-1] to YIQ [0-255] for equalizing Y values
        yiq = rgb2yiq(im_orig)
        y_eq, hist_orig, hist_eq = intensity_equalize(yiq[..., Y])

        # Update Y with equalized values and go back to RGB
        yiq[..., Y] = y_eq
        im_eq = yiq2rgb(yiq)

    return [im_eq, hist_orig, hist_eq]


def _quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image.
    :param im_orig: input grayscale or RGB image to be quantized (float64
                    image with values in [0, 1])
    :param n_quant: number of intensities the output im_quant image should
                    have.
    :param n_iter:  maximum number of iterations of the optimization
                    procedure (may converge earlier.)
    :return:        list [im_quant, error, z_arr] where:
                    im_quant - is the quantized output image.
                    error - is an array with shape (n_iter,) (or less) of
                    the total intensities error for each iteration of the
                    quantization procedure.
                    z_arr - an array of intensity segments mapped to
                    respective q values
    """
    # Build histogram of pixel intensities
    if (is_greyscale(im_orig)):
        hist_orig = intensity_histogram_translated(im_orig)
    else:
        yiq = rgb2yiq(im_orig)
        hist_orig = intensity_histogram_translated(yiq[..., Y])

    # Distribute pixels ranges by equal cumulative sum
    z_arr = np.arange(n_quant + 1)
    cdf = hist_orig.cumsum()
    z_space = cdf[-1] / n_quant
    z_cumsums = np.linspace(z_space, cdf[-1], n_quant)
    for i in range(n_quant):
        z_arr[i + 1] = np.argmin(cdf < z_cumsums[i])

    # Initial guess: q are medians of each range
    q_arr = np.arange(n_quant)
    for i in range(n_quant):
        start, end = z_arr[i], z_arr[i + 1] + 1
        q_arr[i] = (start + end) / 2

    # Initialize errors array
    error = np.array([0] * n_iter)

    # Iterate until n_iter exceeded or z_arr did not change
    for j in range(n_iter):
        # Reset iteration error
        error_j = 0

        # Store previous z values to check for convergence
        z_arr_prev = z_arr.copy()

        # Calculate q values for current z
        for i in range(len(z_arr) - 1):
            start, end = z_arr[i], z_arr[i + 1] + 1
            szp = sum(hist_orig[start:end] * np.arange(start, end))
            sp = sum(hist_orig[start:end])
            q_arr[i] = szp / sp
            error_j += np.sum((hist_orig[start:end] - q_arr[i]) ** 2)

        # Calculate z values by updated q
        for i in range(1, len(q_arr)):
            z_arr[i] = (q_arr[i - 1] + q_arr[i]) / 2

        # Stop iterating upon convergence
        if (np.array_equal(z_arr, z_arr_prev)):
            # Yay! we have converged! :-D
            break

        # Record iteration error
        error[j] = error_j

    # Build quantization lookup table
    lut = np.arange(PIXEL_INTENSITIES)
    for i in range(len(z_arr) - 1):
        start, end = z_arr[i], z_arr[i + 1]
        lut[start:end + 1] = q_arr[i]

    if (is_greyscale(im_orig)):
        intensities = im_orig
    else:
        yiq = rgb2yiq(im_orig)
        intensities = yiq[..., Y]

    # Translate [0,1] intensity range to [0,255] integer range
    intensities = np.round(intensities * PIXEL_INTENSITY_MAX).astype(
        np.uint8)

    # Map intensity values to their quantized values
    intensities_quant = np.array(list(map(lambda i: lut[i],
                                          intensities))).astype(np.uint8)
    # Translate [0,255] intensity range back to [0,1]
    intensities_quant = intensities_quant / PIXEL_INTENSITIES

    if (is_greyscale(im_orig)):
        im_quant = intensities_quant
    else:
        yiq[..., Y] = intensities_quant
        im_quant = yiq2rgb(yiq)

    # Woohoo!
    return im_quant, error, z_arr


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image.
    :param im_orig: input grayscale or RGB image to be quantized (float64
                    image with values in [0, 1])
    :param n_quant: number of intensities the output im_quant image should
                    have.
    :param n_iter:  maximum number of iterations of the optimization
                    procedure (may converge earlier.)
    :return:        list [im_quant, error] where:
                    im_quant - is the quantized output image.
                    error - is an array with shape (n_iter,) (or less) of
                    the total intensities error for each iteration of the
                    quantization procedure.
    """
    return _quantize(im_orig, n_quant, n_iter)[0:2]  # drop the z_arr


def quantize_rgb(im_orig, n_quant, n_iter):
    """
        Performs optimal RGB quantization of a given greyscale or RGB image.
        :param im_orig: input grayscale or RGB image to be quantized (float64
                        image with values in [0, 1])
        :param n_quant: number of intensities the output im_quant image should
                        have.
        :param n_iter:  maximum number of iterations of the optimization
                        procedure (may converge earlier.)
        :return:        list [im_quant, error] where:
                        im_quant - is the quantized output image.
                        error - is an array with shape (n_iter,) (or less) of
                        the total intensities error for each iteration of the
                        quantization procedure.
        """
    # Get array of z segments after quantization of yiq intensities
    z_arr = _quantize(im_orig, n_quant, n_iter)[2]

    # Calculate histogram of each RGB channel
    histograms = np.arange(len(RGB_CHANNELS))
    for channel in RGB_CHANNELS:
        histograms[channel] = intensity_histogram_translated(
            im_orig[..., channel])

    # For each channel in RGB, calculate q values for each z segment
    for channel in range(DIMENSIONS_RGB):
        for i in range(len(z_arr) - 1):
            start, end = z_arr[i], z_arr[i + 1] + 1
            szp = sum(histograms[channel, start:end] * np.arange(start, end))
            sp = sum(histograms[channel, start:end])

