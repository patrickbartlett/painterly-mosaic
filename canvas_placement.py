import numpy as np
from PIL import Image
from skimage import color


def find_max_diff_region(img1: Image.Image, img2: Image.Image, brush_size: int) -> tuple[int, int]:
    arr1 = np.asarray(img1.convert("RGB")).astype(np.float64) / 255.0
    arr2 = np.asarray(img2.convert("RGB")).astype(np.float64) / 255.0

    lab1 = color.rgb2lab(arr1)
    lab2 = color.rgb2lab(arr2)

    diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    window_size = int(round(1.27 * brush_size))
    pad = (window_size // 2) - 1
    pad = max(pad, 0)

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

    # diff is output for testing
    return (center_x, center_y, diff)


def get_placement_section(img: Image.Image, x: int, y: int, region_size: int) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]

    half = region_size // 2

    # Output array initialized to 0
    result = np.zeros((region_size, region_size, 3), dtype=arr.dtype)

    # Source coordinates in original image
    src_y_start = y - half
    src_y_end = src_y_start + region_size
    src_x_start = x - half
    src_x_end = src_x_start + region_size

    # Destination coordinates in result array
    dst_y_start = 0
    dst_x_start = 0

    # Clip to image bounds and adjust destination accordingly
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

    # Compute how much we're actually copying
    copy_h = src_y_end - src_y_start
    copy_w = src_x_end - src_x_start

    if copy_h > 0 and copy_w > 0:
        result[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = \
            arr[src_y_start:src_y_end, src_x_start:src_x_end]

    return result

