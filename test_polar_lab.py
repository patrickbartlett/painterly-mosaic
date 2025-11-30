"""
PolarLAB Testing Suite

Generates test images, converts to PolarLAB, rotates, and outputs comparison grid.
"""

import numpy as np
from pathlib import Path
from skimage import color, io
from skimage.transform import resize
from polar_lab import PolarLAB


def polar_to_image(polar: PolarLAB, size: int = 128) -> np.ndarray:
    """Reconstruct an image from PolarLAB representation."""
    center = size / 2
    y, x = np.mgrid[0:size, 0:size]
    dx, dy = x - center, y - center

    distance = np.sqrt(dx**2 + dy**2)
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    ring_bounds = np.array([
        center * np.sqrt((i + 1) / polar.rings)
        for i in range(polar.rings)
    ])

    ring_idx = np.clip(np.searchsorted(ring_bounds, distance, side='left'), 0, polar.rings - 1)
    sector_idx = np.clip((angle / (2 * np.pi) * polar.sectors).astype(int), 0, polar.sectors - 1)

    lab_img = polar.data[ring_idx, sector_idx]
    rgb = color.lab2rgb(lab_img)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


# Test image generators
def generate_quadrant(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    h = size // 2
    img[:h, :h], img[:h, h:], img[h:, :h], img[h:, h:] = [255,0,0], [0,255,0], [0,0,255], [255,255,0]
    return img

def generate_gradient(size: int = 64) -> np.ndarray:
    t = (np.arange(size)[:, None] + np.arange(size)[None, :]) / (2 * size)
    img = np.stack([t * 255, (1 - t) * 255, np.full((size, size), 128)], axis=-1)
    return img.astype(np.uint8)

def generate_circle(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size / 2
    y, x = np.mgrid[0:size, 0:size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    angle = np.arctan2(y - center, x - center)
    mask = dist < size * 0.4
    img[mask, 0] = ((np.sin(angle[mask]) + 1) * 127).astype(np.uint8)
    img[mask, 1] = ((np.cos(angle[mask]) + 1) * 127).astype(np.uint8)
    img[mask, 2] = ((dist[mask] / (size * 0.4)) * 255).astype(np.uint8)
    return img

def generate_stripe(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    colors = [[255,0,0], [255,128,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [128,0,255], [255,0,255]]
    w = size // 8
    for i, c in enumerate(colors):
        img[:, i*w:(i+1)*w] = c
    return img

def generate_asymmetric(size: int = 64) -> np.ndarray:
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    img[:size//4, :size//4] = [255, 0, 0]
    img[:size//8, size//4:3*size//4] = [0, 255, 0]
    y, x = np.mgrid[0:size, 0:size]
    mask = (x - 3*size//4)**2 + (y - 3*size//4)**2 < (size//8)**2
    img[mask] = [0, 0, 255]
    return img


GENERATORS = [
    ("quadrant", generate_quadrant),
    ("gradient", generate_gradient),
    ("circle", generate_circle),
    ("stripe", generate_stripe),
    ("asymmetric", generate_asymmetric),
]


def generate_comparison_grid(
    output_path: str = "testpolar/comparison_grid.png",
    rings: int = 10,
    sectors: int = 72,
    cell_size: int = 128,
    num_rotations: int = 5
):
    """Generate comparison grid showing original, polar, and rotations."""
    rotation_steps = [i * sectors // num_rotations for i in range(num_rotations)]

    padding = 4
    n_rows = len(GENERATORS)
    n_cols = 1 + num_rotations

    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for row, (name, gen_func) in enumerate(GENERATORS):
        y = padding + row * (cell_size + padding)
        original = gen_func(64)
        polar = PolarLAB.from_image(original, rings=rings, sectors=sectors)

        # Original (upscaled)
        original_big = np.repeat(np.repeat(original, 2, axis=0), 2, axis=1)
        grid[y:y+cell_size, padding:padding+cell_size] = original_big

        # Rotations
        for col, steps in enumerate(rotation_steps):
            x = padding + (1 + col) * (cell_size + padding)
            grid[y:y+cell_size, x:x+cell_size] = polar_to_image(polar.rotate(steps), cell_size)

    Path(output_path).parent.mkdir(exist_ok=True)
    io.imsave(output_path, grid)
    print(f"Saved: {output_path}")

def test_comparison_grid(
    input_dir: str = "testicons",
    output_path: str = "testpolar/test_comparison_grid.png",
    rings: int = 10,
    sectors: int = 72,
    cell_size: int = 128,
    num_rotations: int = 5
):

    # Find all image files in the input directory
    input_path = Path(input_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.tga'}
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    rotation_steps = [i * sectors // num_rotations for i in range(num_rotations)]

    padding = 4
    n_rows = len(image_files)
    n_cols = 1 + num_rotations

    grid_h = n_rows * cell_size + (n_rows + 1) * padding
    grid_w = n_cols * cell_size + (n_cols + 1) * padding
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for row, image_file in enumerate(image_files):
        y = padding + row * (cell_size + padding)

        # Load and prepare the image
        original = io.imread(image_file)

        # Handle RGBA images by discarding alpha channel
        if original.ndim == 3 and original.shape[2] == 4:
            original = original[:, :, :3]

        # Handle grayscale images by converting to RGB
        if original.ndim == 2:
            original = np.stack([original] * 3, axis=-1)

        polar = PolarLAB.from_image(original, rings=rings, sectors=sectors)

        # Original (resized to cell_size)
        original_resized = (resize(original, (cell_size, cell_size), anti_aliasing=True) * 255).astype(np.uint8)
        grid[y:y+cell_size, padding:padding+cell_size] = original_resized

        # Rotations
        for col, steps in enumerate(rotation_steps):
            x = padding + (1 + col) * (cell_size + padding)
            grid[y:y+cell_size, x:x+cell_size] = polar_to_image(polar.rotate(steps), cell_size)

    Path(output_path).parent.mkdir(exist_ok=True)
    io.imsave(output_path, grid)
    print(f"Saved: {output_path} ({len(image_files)} images)")


if __name__ == "__main__":

    generate_comparison_grid()
    test_comparison_grid()
