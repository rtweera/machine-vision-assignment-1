from typing import Tuple
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_derivative_of_gaussian(kernel_size: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    Y, X = np.meshgrid(ax, ax)
    standard_gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) # constant factor omitted since we normalize later

    # DoG: dG/dZ = - (Z / sigma^2) * G(.)
    kernel_X = - (X / (sigma**2)) * standard_gaussian
    kernel_Y = - (Y / (sigma**2)) * standard_gaussian
    
    # L1 normalization (i.e., sum of absolute values)
    normalized_kernel_X = kernel_X / np.sum(np.abs(kernel_X))
    normalized_kernel_Y = kernel_Y / np.sum(np.abs(kernel_Y))
    return normalized_kernel_X, normalized_kernel_Y

def normalize_image(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) * (255.0 / (max_val - min_val))
    return normalized.astype(np.uint8)

def main():
    kernel_5x5_X, kernel_5x5_Y = get_derivative_of_gaussian(5, 2)
    print(f"5x5 DoG Kernel (sigma=2) - X direction: \n{np.round(kernel_5x5_X, 3)}")
    print(f"5x5 DoG Kernel (sigma=2) - Y direction: \n{np.round(kernel_5x5_Y, 3)}")

    kernel_51x51_X, kernel_51x51_Y = get_derivative_of_gaussian(51, 8)
    grid = np.linspace(-25, 25, 51)
    X, Y = np.meshgrid(grid, grid)
    fig1 = plt.figure(figsize=(18, 10))

    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, kernel_51x51_X, cmap='coolwarm')
    ax1.set_title("Derivative of Gaussian (X)")
    ax1.set_zlabel("Amplitude")

    im = cv.imread('resources/runway.png', cv.IMREAD_GRAYSCALE)
    assert im is not None, "Image not found!"
    # set ddepth to CV_64F to avoid overflow/underflow, if set -1, it will be converted to uint8 & below 0 
    # and above 255 will be truncated to 0 and 255 respectively -> so lose information
    nabla_X_manual = cv.filter2D(im, cv.CV_64F, kernel_5x5_X) 
    nabla_Y_manual = cv.filter2D(im, cv.CV_64F, kernel_5x5_Y)
    mag_manual = cv.magnitude(nabla_X_manual, nabla_Y_manual)

    nabla_X_cv = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)
    nabla_Y_cv = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)
    mag_cv = cv.magnitude(nabla_X_cv, nabla_Y_cv)

    fig2 = plt.figure(figsize=(18, 10))

    ax2 = fig2.add_subplot(2, 3, 1)
    ax2.imshow(normalize_image(nabla_X_manual), cmap='gray')
    ax2.set_title("Manual DoG (X-Grad)")
    ax2.axis('off')

    ax2 = fig2.add_subplot(2, 3, 2)
    ax2.imshow(normalize_image(nabla_Y_manual), cmap='gray')
    ax2.set_title("Manual DoG (Y-Grad)")
    ax2.axis('off')

    ax3 = fig2.add_subplot(2, 3, 3)
    ax3.imshow(normalize_image(mag_manual), cmap='gray')
    ax3.set_title("Manual DoG Magnitude")
    ax3.axis('off')

    ax4 = fig2.add_subplot(2, 3, 4)
    ax4.imshow(normalize_image(nabla_X_cv), cmap='gray')
    ax4.set_title("OpenCV Sobel (X-Grad)")
    ax4.axis('off')

    ax4 = fig2.add_subplot(2, 3, 5)
    ax4.imshow(normalize_image(nabla_Y_cv), cmap='gray')
    ax4.set_title("OpenCV Sobel (Y-Grad)")
    ax4.axis('off')

    ax5 = fig2.add_subplot(2, 3, 6)
    ax5.imshow(normalize_image(mag_cv), cmap='gray')
    ax5.set_title("OpenCV Sobel Magnitude")
    ax5.axis('off')

    plt.tight_layout()
    plt.show()
main()