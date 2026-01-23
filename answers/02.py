import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.7 
# Checked ~0.5 which is too bright and histogram is right skewed; 0.9 is too dark and histogram is left skewed; 0.7 seems well balanced

def main():
    im_bgr = cv.imread("resources/highlights_and_shadows.jpg")
    assert im_bgr is not None, "Image not found!"
    im_lab = cv.cvtColor(im_bgr, cv.COLOR_BGR2LAB)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 8))

    # 0th channel => Lightness; 1st => Green-red; 2nd => Blue-yellow
    l_channel = im_lab[:, :, 0]    # single channel default cmap by matplotlib is virdis (green-yellow), but we want grayscale
    
    ax1.imshow(l_channel, cmap='gray')
    ax1.set_title("Original Image - L channel")
    ax1.axis('off')

    gamma = GAMMA
    table = np.array([(i/255)**gamma*255 for i in np.arange(0, 256)]).astype(np.uint8)
    l_channel_gamma_corrected = cv.LUT(l_channel, table)
    ax2.imshow(l_channel_gamma_corrected, cmap='gray')
    ax2.set_title(f"Gamma Corrected L channel (Gamma = {gamma})")
    ax2.axis('off')

    ax3.imshow(cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB))
    ax3.set_title("Original BGR Image")
    ax3.axis('off')

    im_lab_gamma_corrected = im_lab.copy()
    im_lab_gamma_corrected[:, :, 0] = l_channel_gamma_corrected
    im_bgr_gamma_corrected = cv.cvtColor(im_lab_gamma_corrected, cv.COLOR_LAB2BGR)
    ax4.imshow(cv.cvtColor(im_bgr_gamma_corrected, cv.COLOR_BGR2RGB))
    ax4.set_title(f"Gamma Corrected BGR Image (Gamma = {gamma})")
    ax4.axis('off')

    ax5.hist(l_channel.flatten(), bins=256, range=(0, 256), color='black')  # flatten 2D image to 1D array (hist needs 1D data)
    ax5.set_title("Histogram of Original L channel")
    ax5.set_xlabel("Pixel Intensity")
    ax5.set_ylabel("Frequency")
    ax5.set_xlim([0, 255])

    ax6.hist(l_channel_gamma_corrected.flatten(), bins=256, range=(0, 256), color='black')
    ax6.set_title(f"Histogram of Gamma Corrected L channel (Gamma = {gamma})")
    ax6.set_xlabel("Pixel Intensity")
    ax6.set_ylabel("Frequency")
    ax6.set_xlim([0, 255])
    plt.tight_layout()
    plt.show()
main()