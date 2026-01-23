import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

BIT_DEPTH = 255

def get_otsu_mask(image: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # cv.THRESH_BINARY | cv.THRESH_OTSU use to get optimal threshold; not the explicit 0 threshold passed in
    otsu_thresh_val, mask = cv.threshold(image, 0, BIT_DEPTH, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return otsu_thresh_val, mask, cv.bitwise_not(mask)

def main():
    im = cv.imread('resources/looking_out.jpg', cv.IMREAD_COLOR)
    assert im is not None, "Image not found!"

    otsu_val, mask, inv_mask = get_otsu_mask(cv.cvtColor(im, cv.COLOR_BGR2GRAY))
    print(f"Otsu's Optimal Threshold Value: {otsu_val}")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
    ax1.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(inv_mask, cmap='gray')
    ax2.set_title("Otsu's Thresholding Mask - for girl")
    ax2.axis('off')

    ax3.hist(cv.cvtColor(im, cv.COLOR_BGR2GRAY).flatten(), bins=256, range=(0, 256), color='black')
    ax3.axvline(otsu_val, color='red', linestyle='--', linewidth=2, label=f'Otsu Threshold: {otsu_val:.0f}')
    ax3.set_title('Histogram of Original Image')
    ax3.set_xlabel('Pixel Intensity')
    ax3.set_ylabel('Frequency')
    ax3.set_xlim([0, 255])
    ax3.legend()

    masked_region = cv.bitwise_and(im, im, mask=inv_mask)
    ax4.imshow(cv.cvtColor(masked_region, cv.COLOR_BGR2RGB), vmin=0, vmax=255)
    ax4.set_title("Segmented Image using Otsu's Mask (Masked Region)")
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

main()