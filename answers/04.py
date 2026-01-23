import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

BIT_DEPTH = 255

def get_otsu_mask(image: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # cv.THRESH_BINARY | cv.THRESH_OTSU use to get optimal threshold; not the explicit 0 threshold passed in
    otsu_thresh_val, mask = cv.threshold(image, 0, BIT_DEPTH, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return otsu_thresh_val, mask, cv.bitwise_not(mask)

def get_foreground_equalized_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Calc hist only for masked area
    hist = cv.calcHist([image], [0], mask, [256], [0, 256])
    cdf = hist.cumsum()

    # To normalize CDF, only use the pixels in mask (i.e., non zero pixels)
    total_foreground_pixels = cv.countNonZero(mask)
    cdf_normalized = cdf / total_foreground_pixels

    table = (cdf_normalized * BIT_DEPTH).astype(np.uint8)
    equalized_full = cv.LUT(image, table)

    foreground_equalized = cv.bitwise_and(equalized_full, equalized_full, mask=mask)
    return foreground_equalized

def main():
    im = cv.imread('resources/looking_out.jpg', cv.IMREAD_COLOR)
    assert im is not None, "Image not found!"

    otsu_val, mask, inv_mask = get_otsu_mask(cv.cvtColor(im, cv.COLOR_BGR2GRAY))
    print(f"Otsu's Optimal Threshold Value: {otsu_val}")

    # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 6))
    fig, ((ax1, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(12, 6))
    ax1.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # ax2.hist(cv.cvtColor(im, cv.COLOR_BGR2GRAY).flatten(), bins=256, range=(0, 256), color='black')
    # ax2.axvline(otsu_val, color='red', linestyle='--', linewidth=2, label=f'Otsu Threshold: {otsu_val:.0f}')
    # ax2.set_title('Histogram of Original Image')
    # ax2.set_xlabel('Pixel Intensity')
    # ax2.set_ylabel('Frequency')
    # ax2.set_xlim([0, 255])
    # ax2.legend()

    ax3.imshow(inv_mask, cmap='gray')
    ax3.set_title("Otsu's Thresholding Mask for foreground")
    ax3.axis('off')

    masked_region = cv.bitwise_and(im, im, mask=inv_mask)
    ax4.imshow(cv.cvtColor(masked_region, cv.COLOR_BGR2RGB), vmin=0, vmax=255)
    ax4.set_title("Segmented Image")
    ax4.axis('off')

    foreground_equalized = get_foreground_equalized_image(cv.cvtColor(im, cv.COLOR_BGR2GRAY), inv_mask)
    ax5.imshow(foreground_equalized, cmap='gray')
    ax5.set_title("Foreground Histogram Equalized")
    ax5.axis('off')

    # ax6.hist(foreground_equalized.flatten(), bins=256, range=(0, 256), color='black')
    # ax6.set_title("Histogram of Equalized Foreground")
    # ax6.set_xlabel("Pixel Intensity")
    # ax6.set_ylabel("Frequency")
    # ax6.set_xlim([0, 255])

    plt.tight_layout()
    plt.show()

main()