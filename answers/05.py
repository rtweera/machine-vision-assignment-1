import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

KERNEL_SIZE = 51
KERNEL_SIGMA = 10

def get_normalized_gaussian_kernal(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    Y, X = np.meshgrid(ax, ax)
    kernel = np.exp(-(X**2 + Y**2) / (2. * sigma**2))   # constant factor omitted due to division during normalization
    return kernel / np.sum(kernel)

def main():
    kernel_5x5 = get_normalized_gaussian_kernal(5, 2) 
    print(f"5x5 Gaussian Kernal (sigma=2): \n{np.round(kernel_5x5, 3)}")

    kernel_51x51 = get_normalized_gaussian_kernal(51, 10)

    # grid
    ax = np.linspace(-25, 25, 51)
    Y, X = np.meshgrid(ax, ax)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, kernel_51x51, cmap='viridis')
    ax1.set_title("3D Surface of 51x51 Gaussian Kernel")
    ax1.set_zlabel("Height (Coefficient)")

    im = cv.imread('resources/runway.png', cv.IMREAD_GRAYSCALE)
    assert im is not None, "Image not found!"

    fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(12, 6))

    ax2.imshow(im, cmap='gray')
    ax2.set_title("Original Grayscale Image")
    ax2.axis('off')

    manual_kernel = get_normalized_gaussian_kernal(KERNEL_SIZE, KERNEL_SIGMA)
    im_blurred = cv.filter2D(im, -1, manual_kernel)
    ax3.imshow(im_blurred, cmap='gray')
    ax3.set_title("Blurred Image - manual")
    ax3.axis('off')

    im_blurred_cv = cv.GaussianBlur(im, (KERNEL_SIZE, KERNEL_SIZE), KERNEL_SIGMA)
    ax4.imshow(im_blurred_cv, cmap='gray')
    ax4.set_title("Blurred Image - OpenCV")
    ax4.axis('off')

    plt.tight_layout()
    plt.show()
main()