import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_normalized_gaussian_kernal(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    Y, X = np.meshgrid(ax, ax)
    kernel = np.exp(-(X**2 + Y**2) / (2. * sigma**2))   # constant factor omitted due to division during normalization
    return kernel / np.sum(kernel)

def main():
    kernal_5x5 = get_normalized_gaussian_kernal(5, 2) 
    print(f"5x5 Gaussian Kernal (sigma=2): \n{np.round(kernal_5x5, 3)}")

    kernel_51x51 = get_normalized_gaussian_kernal(51, 10)

    # grid
    ax = np.linspace(-25, 25, 51)
    Y, X = np.meshgrid(ax, ax)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, kernel_51x51, cmap='viridis')
    ax1.set_title("3D Surface of 51x51 Gaussian Kernel")
    ax1.set_zlabel("Height (Coefficient)")

    plt.tight_layout()
    plt.show()
main()