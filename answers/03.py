import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

L = 256  # Number of intensity levels


def main():
    im = cv.imread('resources/runway.png', cv.IMREAD_GRAYSCALE)
    assert im is not None, "Image not found!"
    M, N = im.shape

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
    ax1.imshow(im, cmap='gray')
    ax1.set_title('Original Grayscale Image')
    ax1.axis('off')

    equalied_image = equalize_histogram(im, M, N)

    ax2.imshow(equalied_image, cmap='gray')
    ax2.set_title('Histogram Equalized Image')
    ax2.axis('off')

    ax3.hist(im.flatten(), bins=256, range=(0, 256), color='black')
    ax3.set_title('Histogram of Original Image')
    ax3.set_xlabel('Pixel Intensity')
    ax3.set_ylabel('Frequency')
    ax3.set_xlim([0, 255])

    ax4.hist(equalied_image.flatten(), bins=256, range=(0, 256), color='black')
    ax4.set_title('Histogram of Equalized Image')
    ax4.set_xlabel('Pixel Intensity')
    ax4.set_ylabel('Frequency')
    ax4.set_xlim([0, 255])

    plt.tight_layout()
    plt.show()

def equalize_histogram(image: np.ndarray, m: int, n: int) -> np.ndarray:
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()  # cumulative distribution function
    table = np.array([(L-1)/(m*n) * cdf[k] for k in range(256)], dtype=np.uint8)
    return table[image]

main()