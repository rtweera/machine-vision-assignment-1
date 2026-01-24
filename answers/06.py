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

def main():
    kernel_5x5_X, kernel_5x5_Y = get_derivative_of_gaussian(5, 2)
    print(f"5x5 DoG Kernel (sigma=2) - X direction: \n{np.round(kernel_5x5_X, 3)}")
    print(f"5x5 DoG Kernel (sigma=2) - Y direction: \n{np.round(kernel_5x5_Y, 3)}")

    kernel_51x51_X, kernel_51x51_Y = get_derivative_of_gaussian(51, 8)
    grid = np.linspace(-25, 25, 51)
    X, Y = np.meshgrid(grid, grid)
    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.plot_surface(X, Y, kernel_51x51_X, cmap='coolwarm')
    ax1.set_title("Derivative of Gaussian (X)")
    ax1.set_zlabel("Amplitude")

    plt.tight_layout()
    plt.show()
main()