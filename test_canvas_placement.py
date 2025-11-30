"""Test find_max_diff_region with synthetic images."""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from canvas_placement import find_max_diff_region


def test_directory(input_dir: str = 'testicons', output_path: str = 'testplacement/testicons.png', brush_size: int = 10, cell_size: int = 64):
    """Test all images in input_dir against black, save results as grid."""
    Path(output_path).parent.mkdir(exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    # Collect valid image files
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # Grid layout: 3 columns (black, original, diff), N rows
    n_cols = 3
    n_rows = len(image_files)
    padding = 4

    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for row, filename in enumerate(image_files):
        filepath = os.path.join(input_dir, filename)
        img = Image.open(filepath).convert("RGB")
        w, h = img.size

        # Create black image of same size
        black = Image.new("RGB", (w, h), (0, 0, 0))

        cx, cy, diff = find_max_diff_region(img, black, brush_size)

        # Convert diff to grayscale RGB
        diff_gray = diff_to_grayscale(diff).convert("RGB")
        diff_arr = np.array(diff_gray)

        # Draw red marker at center point (3x3 for visibility)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px = np.clip(cx + dx, 0, w - 1)
                py = np.clip(cy + dy, 0, h - 1)
                diff_arr[py, px] = [255, 0, 0]
        diff_marked = Image.fromarray(diff_arr)

        # Resize all to cell_size
        black_resized = black.resize((cell_size, cell_size), Image.NEAREST)
        img_resized = img.resize((cell_size, cell_size), Image.NEAREST)
        diff_resized = diff_marked.resize((cell_size, cell_size), Image.NEAREST)

        # Place in grid
        y = padding + row * (cell_size + padding)
        grid.paste(black_resized, (padding, y))
        grid.paste(img_resized, (padding + cell_size + padding, y))
        grid.paste(diff_resized, (padding + 2 * (cell_size + padding), y))

        print(f"{filename}: center=({cx}, {cy})")

    grid.save(output_path)
    print(f"\nSaved grid with {len(image_files)} images to {output_path}")


def test_directory_2(input_dir: str = 'testicons', output_path: str = 'testplacement/testicons2.png', brush_size: int = 10, cell_size: int = 64):
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

    # Grid layout: 3 columns (img1, img2, diff), N-1 rows
    n_cols = 3
    n_rows = len(image_files) - 1
    padding = 4

    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for row in range(n_rows):
        filepath1 = os.path.join(input_dir, image_files[row])
        filepath2 = os.path.join(input_dir, image_files[row + 1])

        img1 = Image.open(filepath1).convert("RGB")
        img2 = Image.open(filepath2).convert("RGB")

        # Resize img2 to match img1 if needed
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.NEAREST)

        w, h = img1.size
        cx, cy, diff = find_max_diff_region(img1, img2, brush_size)

        # Convert diff to grayscale RGB
        diff_gray = diff_to_grayscale(diff).convert("RGB")
        diff_arr = np.array(diff_gray)

        # Draw red marker at center point (3x3 for visibility)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px = np.clip(cx + dx, 0, w - 1)
                py = np.clip(cy + dy, 0, h - 1)
                diff_arr[py, px] = [255, 0, 0]
        diff_marked = Image.fromarray(diff_arr)

        # Resize all to cell_size
        img1_resized = img1.resize((cell_size, cell_size), Image.NEAREST)
        img2_resized = img2.resize((cell_size, cell_size), Image.NEAREST)
        diff_resized = diff_marked.resize((cell_size, cell_size), Image.NEAREST)

        # Place in grid
        y = padding + row * (cell_size + padding)
        grid.paste(img1_resized, (padding, y))
        grid.paste(img2_resized, (padding + cell_size + padding, y))
        grid.paste(diff_resized, (padding + 2 * (cell_size + padding), y))

        print(f"{image_files[row]} vs {image_files[row + 1]}: center=({cx}, {cy})")

    grid.save(output_path)
    print(f"\nSaved grid with {n_rows} comparisons to {output_path}")


def make_test_pair(case: int) -> tuple[Image.Image, Image.Image, str]:
    """Generate a pair of 64x64 test images with known difference region."""
    size = 64
    base = np.full((size, size, 3), 128, dtype=np.uint8)
    modified = base.copy()

    if case == 0:
        # Bright patch in center
        modified[28:36, 28:36] = [255, 0, 0]
        desc = "center red patch"
    elif case == 1:
        # Gradient difference on right side
        for x in range(48, 64):
            modified[:, x] = [100 + (x - 48) * 10, 128, 128]
        desc = "right gradient"
    elif case == 2:
        # Corner difference (top-left)
        modified[0:8, 0:8] = [0, 255, 0]
        desc = "top-left corner green patch"
    elif case == 3:
        # Diagonal stripe
        for i in range(size):
            if 20 <= i < 30:
                modified[i, i] = [255, 255, 0]
                modified[i, min(i + 1, 63)] = [255, 255, 0]
        desc = "diagonal stripe"
    else:
        # Edge difference (bottom edge)
        modified[58:64, 20:44] = [0, 0, 255]
        desc = "bottom edge blue patch"

    return Image.fromarray(base), Image.fromarray(modified), desc


def test_generated(output_path: str = 'testplacement/testgenerated.png', brush_size: int = 3, cell_size: int = 64):
    """Test generated image pairs, save results as grid."""
    Path(output_path).parent.mkdir(exist_ok=True)

    num_cases = 5
    n_cols = 3
    n_rows = num_cases
    padding = 4

    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for row in range(num_cases):
        base, modified, desc = make_test_pair(row)
        w, h = base.size

        cx, cy, diff = find_max_diff_region(base, modified, brush_size)

        # Convert diff to grayscale RGB
        diff_gray = diff_to_grayscale(diff).convert("RGB")
        diff_arr = np.array(diff_gray)

        # Draw red marker at center point (3x3 for visibility)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px = np.clip(cx + dx, 0, w - 1)
                py = np.clip(cy + dy, 0, h - 1)
                diff_arr[py, px] = [255, 0, 0]
        diff_marked = Image.fromarray(diff_arr)

        # Resize all to cell_size
        base_resized = base.resize((cell_size, cell_size), Image.NEAREST)
        modified_resized = modified.resize((cell_size, cell_size), Image.NEAREST)
        diff_resized = diff_marked.resize((cell_size, cell_size), Image.NEAREST)

        # Place in grid
        y = padding + row * (cell_size + padding)
        grid.paste(base_resized, (padding, y))
        grid.paste(modified_resized, (padding + cell_size + padding, y))
        grid.paste(diff_resized, (padding + 2 * (cell_size + padding), y))

        print(f"Case {row} ({desc}): center=({cx}, {cy})")

    grid.save(output_path)
    print(f"\nSaved grid with {num_cases} test cases to {output_path}")

def diff_to_grayscale(diff: np.ndarray) -> Image.Image:
    """Convert diff map to grayscale image, normalized."""
    normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255
    return Image.fromarray(normalized.astype(np.uint8), mode="L")


def main():
    test_directory()
    test_directory_2()
    test_generated()



if __name__ == "__main__":
    main()
