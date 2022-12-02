import numpy as np
from matplotlib import pyplot as plt

from utils import is_greyscale, rgb2yiq, yiq2rgb, intensity_histogram_translated, PIXEL_INTENSITY_MAX, \
    PIXEL_INTENSITIES, Y, read_image, MODE_GRAYSCALE, display_image

EQUALIZATION_DEMO_ORIGINAL = "demo/histogram_equalization/before_equalization.jpg"
EQUALIZATION_DEMO_OUTPUT = "demo/histogram_equalization/after_equalization.jpg"


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
    lookup = (cdf * PIXEL_INTENSITY_MAX / cdf[-1]).astype(np.uint8)

    # Map original intensity values to their equalized values
    intensity_eq = np.array(list(map(lambda i: lookup[i],
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


def histogram_equalization_demo():
    demo_equalize_image = read_image(EQUALIZATION_DEMO_ORIGINAL, MODE_GRAYSCALE)
    im_eq, hist_orig, hist_eq = histogram_equalize(demo_equalize_image)
    display_image(im_eq, plt.cm.gray)
    plt.imsave(EQUALIZATION_DEMO_OUTPUT, im_eq, cmap=plt.cm.gray)


if __name__ == "__main__":
    histogram_equalization_demo()
