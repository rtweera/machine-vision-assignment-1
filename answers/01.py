import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

BIT_DEPTH = 255

def get_gamma_corrected_image(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([(i/BIT_DEPTH)**gamma*BIT_DEPTH for i in np.arange(0, BIT_DEPTH+1)]).astype(np.uint8)
    return cv.LUT(image, table)

def get_contrast_stretched_image(image: np.ndarray, normalized_low_point: float, normalized_high_point: float) -> np.ndarray:
    # Assumed truncation is preferred over rounding
    low_point = int(BIT_DEPTH * normalized_low_point)   # input axis
    high_point = int(BIT_DEPTH * normalized_high_point) # input axis

    t1 = np.linspace(0, 0, low_point - 0)   # points < low_point; linspace is end inclusive
    t2 = np.linspace(0, BIT_DEPTH, high_point - low_point + 1)    # low_point <= points <= high_point
    t3 = np.linspace(BIT_DEPTH, BIT_DEPTH, BIT_DEPTH - high_point)    # points > high_point 

    table = np.concat((t1, t2, t3), axis=0).astype('uint8')
    return cv.LUT(image, table)

def main(): 
    im = cv.imread('resources/runway.png', cv.IMREAD_GRAYSCALE)
    assert im is not None, "Image not found!"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 6))
    ax1.imshow(im, cmap='gray')
    ax1.set_title('Original Grayscale Image')
    ax1.axis('off')

    im_gamma_0_5 = get_gamma_corrected_image(im, 0.5)
    ax2.imshow(im_gamma_0_5, cmap='gray')
    ax2.set_title('Gamma Corrected (Gamma = 0.5)')
    ax2.axis('off')

    im_gamma_2 = get_gamma_corrected_image(im, 2)
    ax3.imshow(im_gamma_2, cmap='gray')
    ax3.set_title('Gamma Corrected (Gamma = 2)')
    ax3.axis('off')

    im_contrast_stretched = get_contrast_stretched_image(im, 0.2, 0.8)
    ax4.imshow(im_contrast_stretched, cmap='gray')
    ax4.set_title('Constrast Stretched (r1=0.2, r2=0.8)')
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

main()