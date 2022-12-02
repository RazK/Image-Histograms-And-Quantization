import numpy as np
from matplotlib import pyplot as plt

from utils import is_greyscale, rgb2yiq, yiq2rgb, intensity_histogram_translated, PIXEL_INTENSITY_MAX, \
    PIXEL_INTENSITIES, DIMENSIONS_RGB, RGB_CHANNELS, Y, read_image, MODE_GRAYSCALE, display_image

FIFTY_SHADES_ORIGINAL = "demo/optimal_quantization/fifty_shades_original.jpg"
FIFTY_SHADES_OUTPUT = "demo/optimal_quantization/fifty_shades_output_{}.jpg"
SHADES = 3


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
    error, z_arr = _quantize(im_orig, n_quant, n_iter)[1:3]

    # Calculate histogram of each RGB channel
    histograms = list(RGB_CHANNELS)
    for channel in RGB_CHANNELS:
        histograms[channel] = intensity_histogram_translated(
            im_orig[..., channel])

    # For each channel in RGB, calculate q values for each z segment
    q_values = list(RGB_CHANNELS)
    for channel in range(DIMENSIONS_RGB):
        q_values[channel] = np.arange(n_quant)
        for i in range(len(z_arr) - 1):
            start, end = z_arr[i], z_arr[i + 1] + 1
            szp = sum(histograms[channel][start:end] * np.arange(start, end))
            sp = sum(histograms[channel][start:end])
            q_values[channel][i] = szp / sp

    # Build quantization lookup table
    luts = list(RGB_CHANNELS)
    for channel in range(DIMENSIONS_RGB):
        luts[channel] = np.arange(PIXEL_INTENSITIES)
        for i in range(len(z_arr) - 1):
            start, end = z_arr[i], z_arr[i + 1]
            luts[channel][start:end + 1] = q_values[channel][i]

    # Translate [0,1] intensity range to [0,255] integer range
    im_quant = np.round(im_orig * PIXEL_INTENSITY_MAX).astype(
        np.uint8)

    # Map intensity values to their quantized values
    for channel in range(DIMENSIONS_RGB):
        im_quant[..., channel] = np.array(list(map(lambda i: luts[channel][i],
                                                   im_quant[..., channel]))).astype(
            np.uint8)

    # Translate [0,255] intensity range back to [0,1]
    im_quant = im_quant / PIXEL_INTENSITIES

    # Woohoo!
    return im_quant, error, z_arr


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
    error = np.array([0] * n_iter).astype(np.uint64)

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


def optimal_quantization_demo():
    demo_quantize_image = read_image(FIFTY_SHADES_ORIGINAL, MODE_GRAYSCALE)
    im_quant, errors = quantize(demo_quantize_image, SHADES, 20)
    display_image(im_quant, plt.cm.gray)
    plt.imsave(FIFTY_SHADES_OUTPUT.format(SHADES), im_quant, cmap=plt.cm.gray)


if __name__ == "__main__":
    optimal_quantization_demo()
