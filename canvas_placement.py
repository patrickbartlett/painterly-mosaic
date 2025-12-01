import numpy as np
from skimage import io, color, transform
import cv2


def find_max_diff_region(img1: np.ndarray, img2: np.ndarray, brush_size: int) -> tuple[int, int, np.ndarray]:

    img1_lab = cv2.cvtColor(img1[:, :, :3], cv2.COLOR_RGB2LAB) # ~200ms conversion bottleneck
    img2_lab = cv2.cvtColor(img2[:, :, :3], cv2.COLOR_RGB2LAB)

    diff = np.sqrt(np.sum((img1_lab - img2_lab) ** 2, axis=2))
    diff_map = diff


    h, w = diff_map.shape
    s = int(brush_size)

    if s <= 0 or s > h or s > w:
        return 0, 0, diff_map

    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(diff_map, axis=0), axis=1)

    region_sums = (integral[s:, s:]
                 - integral[:-s, s:]
                 - integral[s:, :-s]
                 + integral[:-s, :-s])

    y, x = np.unravel_index(np.argmax(region_sums), region_sums.shape)

    return int(x), int(y), diff_map


def get_placement_section(img: np.ndarray, x: int, y: int, region_size: int) -> np.ndarray:
    section = img[y:y + region_size, x:x + region_size]
    if section.ndim == 3 and section.shape[2] == 4:
        section = section[:, :, :3]
    return section


def _to_rgb_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img
