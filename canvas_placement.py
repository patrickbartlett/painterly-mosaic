import numpy as np
from skimage import io, color, transform


def find_max_diff_region(img1: np.ndarray, img2: np.ndarray, brush_size: int) -> tuple[int, int, np.ndarray]:
    """Find region with maximum perceptual difference between two images.

    Args:
        img1: RGB image array (H, W, 3), uint8 or float
        img2: RGB image array (H, W, 3), uint8 or float
        brush_size: Size of brush for window calculation

    Returns:
        (center_x, center_y, diff_map)
    """
    arr1 = _to_rgb_float(img1)
    arr2 = _to_rgb_float(img2)

    lab1 = color.rgb2lab(arr1)
    lab2 = color.rgb2lab(arr2)

    diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    window_size = int(round(1.27 * brush_size))
    pad = max((window_size // 2) - 1, 0)

    diff_padded = np.pad(diff, pad, mode="constant", constant_values=0)

    integral = np.zeros((diff_padded.shape[0] + 1, diff_padded.shape[1] + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(diff_padded, axis=0), axis=1)

    h, w = diff_padded.shape
    ws = window_size
    y_max = h - ws + 1
    x_max = w - ws + 1

    sums = (
        integral[ws:ws+y_max, ws:ws+x_max]
        - integral[0:y_max, ws:ws+x_max]
        - integral[ws:ws+y_max, 0:x_max]
        + integral[0:y_max, 0:x_max]
    )

    best_idx = np.argmax(sums)
    best_y, best_x = np.unravel_index(best_idx, sums.shape)

    center_padded_y = best_y + ws // 2
    center_padded_x = best_x + ws // 2
    center_y = center_padded_y - pad
    center_x = center_padded_x - pad

    return (center_x, center_y, diff)


def get_placement_section(img: np.ndarray, x: int, y: int, region_size: int) -> np.ndarray:
    """Extract a square section centered at (x, y) from image.

    Args:
        img: RGB image array (H, W, 3)
        x: Center x coordinate
        y: Center y coordinate
        region_size: Size of square region to extract

    Returns:
        Square RGB array of shape (region_size, region_size, 3)
    """
    arr = _to_rgb_uint8(img)
    h, w = arr.shape[:2]

    half = region_size // 2
    result = np.zeros((region_size, region_size, 3), dtype=np.uint8)

    src_y_start = y - half
    src_y_end = src_y_start + region_size
    src_x_start = x - half
    src_x_end = src_x_start + region_size

    dst_y_start = 0
    dst_x_start = 0

    if src_y_start < 0:
        dst_y_start = -src_y_start
        src_y_start = 0
    if src_x_start < 0:
        dst_x_start = -src_x_start
        src_x_start = 0
    if src_y_end > h:
        src_y_end = h
    if src_x_end > w:
        src_x_end = w

    copy_h = src_y_end - src_y_start
    copy_w = src_x_end - src_x_start

    if copy_h > 0 and copy_w > 0:
        result[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = \
            arr[src_y_start:src_y_end, src_x_start:src_x_end]

    return result


def _to_rgb_float(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB float64 in [0, 1]."""
    if img.ndim == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB uint8."""
    if img.ndim == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img
