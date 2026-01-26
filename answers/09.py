import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def sharpen_opencv(image):
    # Standard sharpening kernel
    # The center is positive, neighbors are negative. Sum should be 1 (brightness preserved).
    kernel = np.array([[ 0, -2,  0],
                       [-2,  9, -2],
                       [ 0, -2,  0]])
    
    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened

def main():
    im = cv.imread('resources/cat.png', cv.IMREAD_COLOR)
    assert im is not None, "Image not found!"

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')

    im_sharpened = sharpen_opencv(im)
    ax2.imshow(cv.cvtColor(im_sharpened, cv.COLOR_BGR2RGB))
    ax2.set_title('Sharpened Image (Kernel via OpenCV)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

main()