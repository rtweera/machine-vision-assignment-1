import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    im_bgr = cv.imread("resources/highlights_and_shadows.jpg")
    assert im_bgr is not None, "Image not found!"
    im_lab = cv.cvtColor(im_bgr, cv.COLOR_BGR2LAB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')

    # 0th channel => Lightness; 1st => Green-red; 2nd => Blue-yellow
    ax2.imshow(im_lab[:, :, 0], cmap="gray")    # single channel default cmap by matplotlib is virdis (green-yellow), but we want grayscale
    ax2.set_title("Lightness channel")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
main()