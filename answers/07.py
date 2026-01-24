
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

ZOOM_FACTOR = 2
PIXEL_GRID = False
SHOW_DETAIL_CROP = True

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
    
def draw_grid_lines(ax, image: np.ndarray, line_color: str, line_width: float, line_alpha: float, skip: int = 1):
    if not PIXEL_GRID:
        return
    h, w = image.shape[:2]
    for i in range(0, h, skip):
        ax.axhline(i - 0.5, color=line_color, linewidth=line_width, alpha=line_alpha)
    for i in range(0, w, skip):
        ax.axvline(i - 0.5, color=line_color, linewidth=line_width, alpha=line_alpha)

def main():
    im = cv.imread('resources/images_for_zoom/a1q5images/taylor_very_small.jpg')
    assert im is not None, "Image not found!"

    zoom_factor = ZOOM_FACTOR
    show_detail_crop = SHOW_DETAIL_CROP

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    print(f"Original image resolution: {im.shape[1]} x {im.shape[0]} pixels")
    axes[0, 0].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image (Small)')
    draw_grid_lines(axes[0, 0], im, 'cyan', 0.5, 0.5)
    axes[0, 0].axis('off')

    im_bilinear = zoom_image(im, zoom_factor, interpolation=Interpolation.BILINEAR, mode=ZoomMode.OPENCV)
    print(f"Bilinear zoomed resolution: {im_bilinear.shape[1]} x {im_bilinear.shape[0]} pixels")
    axes[0, 1].imshow(cv.cvtColor(im_bilinear, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Bilinear Zoomed (x{zoom_factor}) - Smooth')
    draw_grid_lines(axes[0, 1], im_bilinear, 'red', 0.5, 0.4)
    axes[0, 1].axis('off')

    im_nearest = zoom_image(im, zoom_factor, interpolation=Interpolation.NEAREST, mode=ZoomMode.OPENCV)
    print(f"Nearest Neighbor zoomed resolution: {im_nearest.shape[1]} x {im_nearest.shape[0]} pixels")
    axes[0, 2].imshow(cv.cvtColor(im_nearest, cv.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Nearest Neighbor Zoomed (x{zoom_factor}) - Sharp/Blocky')
    draw_grid_lines(axes[0, 2], im_nearest, 'lime', 0.8, 0.6)
    axes[0, 2].axis('off')

    if show_detail_crop:
        # Extract a small region from the center and zoom further for visibility
        h, w = im.shape[:2]
        crop_size = 20  # 20x20 pixel crop
        x_start = (w - crop_size) // 2
        y_start = (h - crop_size) // 2
        
        im_crop = im[y_start:y_start + crop_size, x_start:x_start + crop_size]
        im_bilinear_crop = im_bilinear[y_start*zoom_factor:(y_start+crop_size)*zoom_factor, x_start*zoom_factor:(x_start+crop_size)*zoom_factor]
        im_nearest_crop = im_nearest[y_start*zoom_factor:(y_start+crop_size)*zoom_factor, x_start*zoom_factor:(x_start+crop_size)*zoom_factor]
        
        axes[1, 1].imshow(cv.cvtColor(im_bilinear_crop, cv.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Bilinear Crop - Smooth/Blended')
        draw_grid_lines(axes[1, 1], im_bilinear_crop, 'red', 0.5, 0.4)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv.cvtColor(im_nearest_crop, cv.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'NN Crop - Blocky')
        draw_grid_lines(axes[1, 2], im_nearest_crop, 'lime', 1.0, 0.7)
        axes[1, 2].axis('off')
        
        axes[1, 0].imshow(cv.cvtColor(im_crop, cv.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Original Crop - {crop_size}x{crop_size}px')
        draw_grid_lines(axes[1, 0], im_crop, 'cyan', 0.5, 0.5)
        axes[1, 0].axis('off')
    else:
        # Hide the second row if detail crop is off
        for ax in axes[1]:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()

main()