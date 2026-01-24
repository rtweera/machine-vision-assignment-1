import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

DIAMETER = 9
SIGMA_S = 5  # Spatial Sigma
SIGMA_R = 25  # Range Sigma

def manual_bilateral_filter(image, diameter, sigma_s, sigma_r):
    height, width = image.shape
    out = np.zeros((height, width), dtype=np.float32)
    radius = diameter // 2

    # Work in float to avoid uint8 wrap-around on differences
    padded_image = np.pad(image.astype(np.float32), ((radius, radius), (radius, radius)), mode='reflect')  # Pad image to handle borders
    
    # Pre-compute spatial gaussian weights
    ys, xs = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_w = np.exp(-(xs**2 + ys**2) / (2.0 * sigma_s**2)).astype(np.float32)

    for y in range(height):
        for x in range(width):
            patch = padded_image[y:y+diameter, x:x+diameter]
            center = padded_image[y + radius, x + radius]
            
            diff = patch - center   # Difference from center pixel
            range_w = np.exp(-(diff * diff) / (2.0 * sigma_r**2)).astype(np.float32)

            w = spatial_w * range_w     # Combine the two weights: spatial and range

            # Normalize & weighted sum
            w_sum = np.sum(w)
            if w_sum > 1e-8:
                out[y, x] = np.sum(w * patch) / w_sum
            else:
                out[y, x] = center
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    im = cv.imread("resources/highlights_and_shadows.jpg", cv.IMREAD_GRAYSCALE)
    assert im is not None, "Image not found!"

    blurred_image = cv.GaussianBlur(im, (DIAMETER, DIAMETER), sigmaX=SIGMA_S)
    bilateral_cv = cv.bilateralFilter(im, DIAMETER, SIGMA_R, SIGMA_S)
    bilateral_manual = manual_bilateral_filter(im, DIAMETER, SIGMA_S, SIGMA_R)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.imshow(im, cmap='gray')
    ax1.set_title('Original Grayscale Image')
    ax1.axis('off')
    
    ax2.imshow(blurred_image, cmap='gray')
    ax2.set_title('Gaussian Blurred Image')
    ax2.axis('off')
    
    ax3.imshow(bilateral_cv, cmap='gray')
    ax3.set_title('Bilateral Filtered Image (OpenCV)')
    ax3.axis('off')
    
    ax4.imshow(bilateral_manual, cmap='gray')
    ax4.set_title('Bilateral Filtered Image (Manual)')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

main()