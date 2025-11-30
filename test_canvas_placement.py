"""Test find_max_diff_region with synthetic images."""

import os
import numpy as np
from pathlib import Path
from skimage import io, transform
from canvas_placement import find_max_diff_region, get_placement_section


def test_directory(input_dir: str = 'testicons', output_path: str = 'testplacement/testicons.png',
                   brush_size: int = 10, cell_size: int = 64):
    """Test all images in input_dir against black, save results as grid."""
    Path(output_path).parent.mkdir(exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    n_cols = 3
    n_rows = len(image_files)
    padding = 4

    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for row, filename in enumerate(image_files):
        filepath = os.path.join(input_dir, filename)
        img = _load_rgb(filepath)
        h, w = img.shape[:2]

        black = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy, diff = find_max_diff_region(img, black, brush_size)

        diff_marked = _diff_to_marked_rgb(diff, cx, cy, w, h)

        y = padding + row * (cell_size + padding)
        _paste_resized(grid, black, padding, y, cell_size)
        _paste_resized(grid, img, padding + cell_size + padding, y, cell_size)
        _paste_resized(grid, diff_marked, padding + 2 * (cell_size + padding), y, cell_size)

        print(f"{filename}: center=({cx}, {cy})")

    io.imsave(output_path, grid)
    print(f"\nSaved grid with {len(image_files)} images to {output_path}")


def test_directory_2(input_dir: str = 'testicons', output_path: str = 'testplacement/testicons2.png',
                     brush_size: int = 10, cell_size: int = 64):
    """Test consecutive image pairs, save results as grid."""
    Path(output_path).parent.mkdir(exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if len(image_files) < 2:
        print(f"Need at least 2 images in {input_dir}")
        return

    n_cols = 3
    n_rows = len(image_files) - 1
    padding = 4

    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for row in range(n_rows):
        filepath1 = os.path.join(input_dir, image_files[row])
        filepath2 = os.path.join(input_dir, image_files[row + 1])

        img1 = _load_rgb(filepath1)
        img2 = _load_rgb(filepath2)

        if img1.shape[:2] != img2.shape[:2]:
            img2 = _resize_uint8(img2, (img1.shape[0], img1.shape[1]))

        h, w = img1.shape[:2]
        cx, cy, diff = find_max_diff_region(img1, img2, brush_size)

        diff_marked = _diff_to_marked_rgb(diff, cx, cy, w, h)

        y = padding + row * (cell_size + padding)
        _paste_resized(grid, img1, padding, y, cell_size)
        _paste_resized(grid, img2, padding + cell_size + padding, y, cell_size)
        _paste_resized(grid, diff_marked, padding + 2 * (cell_size + padding), y, cell_size)

        print(f"{image_files[row]} vs {image_files[row + 1]}: center=({cx}, {cy})")

    io.imsave(output_path, grid)
    print(f"\nSaved grid with {n_rows} comparisons to {output_path}")


def make_test_pair(case: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a pair of 64x64 test images with known difference region."""
    size = 64
    base = np.full((size, size, 3), 128, dtype=np.uint8)
    modified = base.copy()

    if case == 0:
        modified[28:36, 28:36] = [255, 0, 0]
        desc = "center red patch"
    elif case == 1:
        for x in range(48, 64):
            modified[:, x] = [100 + (x - 48) * 10, 128, 128]
        desc = "right gradient"
    elif case == 2:
        modified[0:8, 0:8] = [0, 255, 0]
        desc = "top-left corner green patch"
    elif case == 3:
        for i in range(size):
            if 20 <= i < 30:
                modified[i, i] = [255, 255, 0]
                modified[i, min(i + 1, 63)] = [255, 255, 0]
        desc = "diagonal stripe"
    else:
        modified[58:64, 20:44] = [0, 0, 255]
        desc = "bottom edge blue patch"

    return base, modified, desc


def test_generated(output_path: str = 'testplacement/testgenerated.png', brush_size: int = 3, cell_size: int = 64):
    """Test generated image pairs, save results as grid."""
    Path(output_path).parent.mkdir(exist_ok=True)

    num_cases = 5
    n_cols = 3
    n_rows = num_cases
    padding = 4

    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for row in range(num_cases):
        base, modified, desc = make_test_pair(row)
        h, w = base.shape[:2]

        cx, cy, diff = find_max_diff_region(base, modified, brush_size)

        diff_marked = _diff_to_marked_rgb(diff, cx, cy, w, h)

        y = padding + row * (cell_size + padding)
        _paste_resized(grid, base, padding, y, cell_size)
        _paste_resized(grid, modified, padding + cell_size + padding, y, cell_size)
        _paste_resized(grid, diff_marked, padding + 2 * (cell_size + padding), y, cell_size)

        print(f"Case {row} ({desc}): center=({cx}, {cy})")

    io.imsave(output_path, grid)
    print(f"\nSaved grid with {num_cases} test cases to {output_path}")


def test_get_placement_section(output_path: str = "testplacement/section.png"):
    Path(output_path).parent.mkdir(exist_ok=True)

    img = _load_rgb("assets/WoWlogo.png")
    h, w = img.shape[:2]

    region_size = int(64 * 1.27)
    x, y = w // 2, h // 2

    section = get_placement_section(img, x, y, region_size)
    io.imsave(output_path, section)


def _load_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def _resize_uint8(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize image to (height, width) using nearest neighbor."""
    resized = transform.resize(img, shape, order=0, preserve_range=True, anti_aliasing=False)
    return resized.astype(np.uint8)


def _paste_resized(grid: np.ndarray, img: np.ndarray, x: int, y: int, cell_size: int):
    """Resize img to cell_size and paste into grid at (x, y)."""
    resized = _resize_uint8(img, (cell_size, cell_size))
    grid[y:y + cell_size, x:x + cell_size] = resized


def _diff_to_marked_rgb(diff: np.ndarray, cx: int, cy: int, w: int, h: int) -> np.ndarray:
    """Convert diff map to grayscale RGB with red marker at center."""
    normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255
    gray = normalized.astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            px = np.clip(cx + dx, 0, w - 1)
            py = np.clip(cy + dy, 0, h - 1)
            rgb[py, px] = [255, 0, 0]

    return rgb


def main():
    test_directory()
    test_directory_2()
    test_generated()
    test_get_placement_section()


if __name__ == "__main__":
    main()
