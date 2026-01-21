import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_gamma_corrected_image(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([(i/255)**gamma*255 for i in np.arange(0, 256)]).astype(np.uint8)
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
    ax2.set_title('Gamma = 0.5')
    ax2.axis('off')

    im_gamma_2 = get_gamma_corrected_image(im, 2)
    ax3.imshow(im_gamma_2, cmap='gray')
    ax3.set_title('Gamma = 2')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

main()