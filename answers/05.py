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

main()