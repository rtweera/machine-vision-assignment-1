
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

# Enums
class Interpolation(Enum):
    BILINEAR = auto()
    NEAREST = auto()

class ZoomMode(Enum):
    MANUAL = auto()
    OPENCV = auto()

def zoom_image(
    image: np.ndarray,
    zoom_factor: int,
    interpolation: Interpolation = Interpolation.BILINEAR,
    mode: ZoomMode = ZoomMode.OPENCV
) -> np.ndarray:
    height, width = image.shape[:2]
    new_height, new_width = height * zoom_factor, width * zoom_factor

    if not isinstance(interpolation, Interpolation):
        raise ValueError(f"Unsupported interpolation method: {interpolation}")
    if mode == ZoomMode.MANUAL:
        raise NotImplementedError("Manual zooming not implemented yet.")
    elif mode == ZoomMode.OPENCV:
        interp_method = cv.INTER_LINEAR if interpolation == Interpolation.BILINEAR else cv.INTER_NEAREST
        return cv.resize(image, (new_width, new_height), interpolation=interp_method)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    

def main():
    im = cv.imread('resources/images_for_zoom/a1q5images/taylor_small.jpg')
    assert im is not None, "Image not found!"

    zoom_factor = 4

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    im_bilinear = zoom_image(im, zoom_factor, interpolation=Interpolation.BILINEAR, mode=ZoomMode.OPENCV)
    axes[1].imshow(cv.cvtColor(im_bilinear, cv.COLOR_BGR2RGB))
    axes[1].set_title(f'Zoomed Image (Bilinear, x{zoom_factor})')
    axes[1].axis('off')

    im_nearest = zoom_image(im, zoom_factor, interpolation=Interpolation.NEAREST, mode=ZoomMode.OPENCV)
    axes[2].imshow(cv.cvtColor(im_nearest, cv.COLOR_BGR2RGB))
    axes[2].set_title(f'Zoomed Image (Nearest Neighbor, x{zoom_factor})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

main()