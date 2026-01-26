
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
        return resize_image_manual(image, zoom_factor, interpolation)
    elif mode == ZoomMode.OPENCV:
        interp_method = cv.INTER_LINEAR if interpolation == Interpolation.BILINEAR else cv.INTER_NEAREST
        return cv.resize(image, (new_width, new_height), interpolation=interp_method)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def calculate_bilinear_interpolate(img, x, y):
    h, w = img.shape[:2]

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    fx = x - x0
    fy = y - y0

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return (
        Ia * (1 - fx) * (1 - fy) +
        Ib * fx * (1 - fy) +
        Ic * (1 - fx) * fy +
        Id * fx * fy
    )


def calculate_nearest_neighbor_interpolate(image: np.ndarray, x: float, y: float) -> np.ndarray:
    h, w = image.shape[:2]
    x_nn = min(max(int(round(x)), 0), w - 1)
    y_nn = min(max(int(round(y)), 0), h - 1)
    return image[y_nn, x_nn]

def resize_image_manual(
    image: np.ndarray,
    zoom_factor: int,
    interpolation: Interpolation = Interpolation.BILINEAR
) -> np.ndarray:
    height, width = image.shape[:2]
    new_height, new_width = height * zoom_factor, width * zoom_factor
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    for y in range(new_height):
        for x in range(new_width):
            # Pixel-center aligned mapping: output pixel center -> input pixel center
            src_x = (x + 0.5) / zoom_factor - 0.5
            src_y = (y + 0.5) / zoom_factor - 0.5
            if interpolation == Interpolation.BILINEAR:
                resized_image[y, x] = calculate_bilinear_interpolate(image, src_x, src_y)
            elif interpolation == Interpolation.NEAREST:
                resized_image[y, x] = calculate_nearest_neighbor_interpolate(image, src_x, src_y)
            else:
                raise ValueError(f"Unsupported interpolation method: {interpolation}")
    return resized_image

def draw_grid_lines(ax, image: np.ndarray, line_color: str, line_width: float, line_alpha: float, skip: int = 1):
    if not PIXEL_GRID:
        return
    h, w = image.shape[:2]
    for i in range(0, h, skip):
        ax.axhline(i - 0.5, color=line_color, linewidth=line_width, alpha=line_alpha)
    for i in range(0, w, skip):
        ax.axvline(i - 0.5, color=line_color, linewidth=line_width, alpha=line_alpha)

def get_zoom_factor(larger_image: np.ndarray, smaller_image: np.ndarray) -> int:
    h_large, w_large = larger_image.shape[:2]
    h_small, w_small = smaller_image.shape[:2]
    if h_large % h_small != 0 or w_large % w_small != 0:
        raise ValueError("Larger image dimensions are not integer multiples of smaller image dimensions.")
    zoom_factor_h = h_large // h_small
    zoom_factor_w = w_large // w_small
    if zoom_factor_h != zoom_factor_w:
        raise ValueError("Non-uniform zoom factors detected between height and width.")
    return zoom_factor_h

def calculate_normalized_ssd(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for SSD calculation.")
    diff = image1.astype(np.float32) - image2.astype(np.float32)    # from uint8 to float32 to avoid overflow during squaring
    ssd = np.sum(diff ** 2)
    normalized_ssd = ssd / (image1.shape[0] * image1.shape[1] * image1.shape[2])  # normalize by number of pixels * channels
    return normalized_ssd   # per pixel per channel error

def main():
    im = cv.imread('resources/images_for_zoom/a1q5images/im03small.png')
    im_large = cv.imread('resources/images_for_zoom/a1q5images/im03.png')
    assert im is not None, "Small image not found!"
    assert im_large is not None, "Large image not found!"
    print(f"Loaded small image of shape: {im.shape}")
    print(f"Loaded large image of shape: {im_large.shape}")

    zoom_factor = get_zoom_factor(im_large, im)
    print(f"Determined zoom factor: {zoom_factor}")
    show_detail_crop = SHOW_DETAIL_CROP

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    print(f"Original image resolution: {im.shape[1]} x {im.shape[0]} pixels")
    axes[0, 0].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image (Small)')
    draw_grid_lines(axes[0, 0], im, 'cyan', 0.5, 0.5)
    axes[0, 0].axis('off')

    im_bilinear = zoom_image(im, zoom_factor, interpolation=Interpolation.BILINEAR, mode=ZoomMode.MANUAL)
    print(f"Bilinear zoomed resolution: {im_bilinear.shape[1]} x {im_bilinear.shape[0]} pixels")
    axes[0, 1].imshow(cv.cvtColor(im_bilinear, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Bilinear Zoomed (x{zoom_factor}) - Smooth')
    draw_grid_lines(axes[0, 1], im_bilinear, 'red', 0.5, 0.4)
    axes[0, 1].axis('off')

    im_nearest = zoom_image(im, zoom_factor, interpolation=Interpolation.NEAREST, mode=ZoomMode.MANUAL)
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

    nssd_bilinear = calculate_normalized_ssd(im_large, im_bilinear)
    print(f"Normalized SSD - Bilinear: {nssd_bilinear:.2f}")
    nssd_nearest = calculate_normalized_ssd(im_large, im_nearest)
    print(f"Normalized SSD - Nearest Neighbor: {nssd_nearest:.2f}")

    plt.tight_layout()
    plt.show()

main()