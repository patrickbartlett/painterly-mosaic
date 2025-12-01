import numpy as np
import os
import math
from skimage import io, color
from scipy.ndimage import rotate

from canvas_placement import find_max_diff_region, get_placement_section
from polar_lab import PolarLAB


def paint(target: np.ndarray, brushstroke_count: int) -> None:

    if target.ndim == 3 and target.shape[2] == 4:
        target = target[:, :, :3]

    brush_vectors, brush_filenames = load_brushes("icons")

    brush_size = 64
    paint_size = int(math.ceil(brush_size * math.sqrt(2)))

    h, w = target.shape[:2]
    canvas = np.full_like(target, 0)

    x, y, diff = find_max_diff_region(target, canvas, paint_size)
    io.imsave("testpaint/diffbefore.png", diff_to_rgb(diff))

    for i in range(100):

        x, y, diff = find_max_diff_region(target, canvas, paint_size)

        section = get_placement_section(target, x, y, paint_size)
        target_polar = PolarLAB.from_query_section(section, brush_size)
        filename, rotation, dist = find_best_match(target_polar, brush_vectors, brush_filenames)
        brushstroke = io.imread(filename)[:, :, :3]

        place_rotated(canvas, brushstroke, x, y, -rotation)

        section_after = canvas[y:y+paint_size, x:x+paint_size]
        target_section = target[y:y+paint_size, x:x+paint_size]
        print(f"  section diff after placement: {np.abs(section_after.astype(float) - target_section.astype(float)).mean():.1f}")


        # reconstructed = polar.mask(0).to_image(size=paint_size)
        # canvas[y:y+paint_size, x:x+paint_size] = reconstructed

    os.makedirs("testpaint", exist_ok=True)
    # io.imsave("testpaint/first.png", (diff_to_rgb(diff)))
    io.imsave("testpaint/first.png", canvas)
    io.imsave("testpaint/diffafter.png", diff_to_rgb(diff))


def place_rotated(canvas: np.ndarray, image: np.ndarray, x: int, y: int,
                  rotation_steps: int, sectors: int = 72) -> None:
    angle = rotation_steps * (360 / sectors)

    # Create mask before rotation (non-black pixels)
    mask = np.any(image > 10, axis=2)  # threshold to avoid noise

    rotated = rotate(image, angle, reshape=True, order=1, mode='constant', cval=0)
    rotated_mask = rotate(mask.astype(float), angle, reshape=True, order=0, mode='constant', cval=0) > 0.5

    rotated = np.clip(rotated, 0, 255).astype(np.uint8)

    square_size = image.shape[0] * np.sqrt(2) # magic number 1.42
    cx = x + square_size / 2
    cy = y + square_size / 2

    rh, rw = rotated.shape[:2]
    px = int(round(cx - rw / 2))
    py = int(round(cy - rh / 2))

    src_x0 = max(0, -px)
    src_y0 = max(0, -py)
    src_x1 = min(rw, canvas.shape[1] - px)
    src_y1 = min(rh, canvas.shape[0] - py)

    dst_x0 = max(0, px)
    dst_y0 = max(0, py)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        src_slice = rotated[src_y0:src_y1, src_x0:src_x1]
        mask_slice = rotated_mask[src_y0:src_y1, src_x0:src_x1]
        canvas[dst_y0:dst_y1, dst_x0:dst_x1][mask_slice] = src_slice[mask_slice]


def load_brushes(icons_dir: str = "icons", rings: int = 10, sectors: int = 72) -> tuple[np.ndarray, list[str]]:
    """Load brushes, precompute all rotations.

    Returns:
        vectors: (n_brushes, sectors, dim) - vectors[b, r] = brush b rotated r steps
        filenames: (n_brushes,)
    """
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.tga'}
    filenames = []
    polar_objects = []

    for root, _, files in os.walk(icons_dir):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() not in extensions:
                continue
            filepath = os.path.join(root, filename)
            try:
                image = io.imread(filepath)

                if image.ndim == 3 and image.shape[2] == 4:
                    image = image[:, :, :3]
                elif image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)

                h, w = image.shape[:2]
                if h != w:
                    size = max(h, w)
                    padded = np.full((size, size, 3), 255, dtype=image.dtype)
                    y0, x0 = (size - h) // 2, (size - w) // 2
                    padded[y0:y0+h, x0:x0+w] = image
                    image = padded

                polar_objects.append(PolarLAB.from_image_padded(image, rings, sectors))
                filenames.append(filepath)

                if len(polar_objects) % 50 == 0:
                    print(f"Loaded {len(polar_objects)} brushes")
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    n = len(polar_objects)
    dim = rings * sectors * 3
    vectors = np.empty((n, sectors, dim), dtype=np.float32)

    for i, p in enumerate(polar_objects):
        for r in range(sectors):
            vectors[i, r] = np.roll(p.data, -r, axis=1).ravel()

    print(f"Loaded {n} brushes total")
    return vectors, filenames


def find_best_match(
    query: PolarLAB,
    brush_vectors: np.ndarray,  # (B, sectors, dim)
    filenames: list[str],
    sectors: int = 72
) -> tuple[str, int, float]:
    """Find closest (brush, rotation) to query.

    Compares brush.rotate(n) vs query.mask(n) for all n.

    Returns: (filename, rotation, distance)
    """
    # Build query vectors: (sectors, dim)
    queries = np.stack([query.mask(r).flatten() for r in range(sectors)])

    # brush_vectors[:, r, :] vs queries[r, :]
    # (B, S, dim) -> transpose -> (S, B, dim)
    # (S, dim) -> expand -> (S, 1, dim)
    diff = brush_vectors.transpose(1, 0, 2) - queries[:, np.newaxis, :]
    distances = np.einsum('sbd,sbd->sb', diff, diff)  # (S, B)

    best_brush_per_rot = np.argmin(distances, axis=1)  # (S,)
    best_dist_per_rot = distances[np.arange(sectors), best_brush_per_rot]  # (S,)

    best_rot = int(np.argmin(best_dist_per_rot))
    best_brush = int(best_brush_per_rot[best_rot])

    return filenames[best_brush], best_rot, float(best_dist_per_rot[best_rot])


def diff_to_rgb(diff: np.ndarray, max_diff: float = None) -> np.ndarray:
    if max_diff is None:
        max_diff = diff.max() if diff.max() > 0 else 1.0

    normalized = np.clip(diff / max_diff, 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)

if __name__ == "__main__":
    target = io.imread("assets/WoWlogo.png")
    paint(target, brushstroke_count=10)

