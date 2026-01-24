from typing import Tuple
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_derivative_of_gaussian(kernel_size: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    Y, X = np.meshgrid(ax, ax)
    standard_gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) # constant factor omitted since we normalize later

    # DoG: dG/dZ = - (Z / sigma^2) * G(Z)
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

main()