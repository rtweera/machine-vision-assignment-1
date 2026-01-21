import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

DISPLAY_MODE: Literal["opencv", "matplotlib"] = "matplotlib"
BGR_TO_RBG_CONVERSION = True

im = cv.imread('resources/images_for_zoom/a1q5images/taylor.jpg')
assert im is not None, "Image not found!"

if DISPLAY_MODE == "opencv":
    cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
    cv.imshow('Image', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

elif DISPLAY_MODE == "matplotlib":
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB) if BGR_TO_RBG_CONVERSION else im
    im_display = ax.imshow(im_rgb)
    ax.xaxis.set_ticks_position('top')
    plt.show()

else:
    raise ValueError(f"Unsupported DISPLAY_MODE: {DISPLAY_MODE}")