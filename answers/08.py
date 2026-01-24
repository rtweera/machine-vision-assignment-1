import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

MANUAL_FUNCTION = True 
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA_X = 1.5
MEDIAN_KERNEL_SIZE = 5

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    Y, X = np.meshgrid(ax, ax)
    kernel = np.exp(-(X**2 + Y**2) / (2. * sigma**2))   # constant factor omitted due to division during normalization
    return kernel / np.sum(kernel)

def apply_gaussian_smoothing(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    return cv.filter2D(image, -1, kernel)

def apply_median_smoothing(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Median kernel size must be a positive odd integer.")

    pad = kernel_size // 2
    padded = np.pad(image, pad_width=pad, mode='reflect')   # opencv default is reflect
    output = np.empty_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = padded[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.median(window)

    return output

def main():
    im = cv.imread('resources/emma_salt_pepper.jpg', cv.IMREAD_GRAYSCALE)
    assert im is not None, "Image not found!"
    
    if MANUAL_FUNCTION:
        im_gaussian_smoothed = apply_gaussian_smoothing(im, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X)
        im_median_smoothed = apply_median_smoothing(im, MEDIAN_KERNEL_SIZE)
    else:
        im_gaussian_smoothed = cv.GaussianBlur(im, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), sigmaX=GAUSSIAN_SIGMA_X)
        im_median_smoothed = cv.medianBlur(im, ksize=MEDIAN_KERNEL_SIZE)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
    ax1.imshow(im, cmap='gray')
    ax1.set_title('Original Noisy Image')
    ax1.axis('off')
    ax2.imshow(im_gaussian_smoothed, cmap='gray')
    ax2.set_title(f'Gaussian Smoothed Image ({GAUSSIAN_KERNEL_SIZE}x{GAUSSIAN_KERNEL_SIZE}, σ={GAUSSIAN_SIGMA_X})')
    ax2.axis('off')
    ax3.imshow(im_median_smoothed, cmap='gray')
    ax3.set_title(f'Median Smoothed Image ({MEDIAN_KERNEL_SIZE}x{MEDIAN_KERNEL_SIZE})')
    ax3.axis('off')

    ax4.hist(im.flatten(), bins=256, range=(0, 256), color='black')
    ax4.set_title('Histogram of Original Noisy Image')
    ax4.set_xlabel('Pixel Intensity')
    ax4.set_ylabel('Frequency')
    ax4.set_xlim([0, 255])

    ax5.hist(im_gaussian_smoothed.flatten(), bins=256, range=(0, 256), color='black')
    ax5.set_title('Histogram of Gaussian Smoothed Image')
    ax5.set_xlabel('Pixel Intensity')
    ax5.set_ylabel('Frequency')
    ax5.set_xlim([0, 255])

    ax6.hist(im_median_smoothed.flatten(), bins=256, range=(0, 256), color='black')
    ax6.set_title('Histogram of Median Smoothed Image')
    ax6.set_xlabel('Pixel Intensity')
    ax6.set_ylabel('Frequency')
    ax6.set_xlim([0, 255])

    plt.tight_layout()
    plt.show()
main()