import numpy as np
from skimage import color


def polar_geometry(size, rings=10, sectors=72):
    """Compute geometry for circumscribed polar grid."""
    diag = int(np.ceil(size * np.sqrt(2)))
    r_max = diag / 2
    ring_bounds = r_max * np.sqrt(np.arange(1, rings + 1) / rings)
    return diag, r_max, ring_bounds


def get_mask_for_rotation(size, rings=10, sectors=72):
    """
    Returns boolean mask (rings, sectors) indicating which bins
    contain real pixels for a size×size source rotated by rotation_steps.
    """
    diag, r_max, ring_bounds = polar_geometry(size, rings, sectors)
    pad = (diag - size) // 2

    center = diag / 2
    y, x = np.mgrid[0:diag, 0:diag]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    angle = np.mod(np.arctan2(y - center, x - center), 2 * np.pi)

    ring_idx = np.clip(np.searchsorted(ring_bounds, distance, side='left'), 0, rings - 1)
    sector_idx = np.clip((angle / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)

    # Valid pixels: inside original square
    valid_pixels = np.zeros((diag, diag), dtype=bool)
    valid_pixels[pad:pad+size, pad:pad+size] = True

    flat_idx = (ring_idx * sectors + sector_idx).ravel()
    valid_counts = np.zeros(rings * sectors, dtype=np.int32)
    np.add.at(valid_counts, flat_idx, valid_pixels.ravel().astype(np.int32))

    mask = (valid_counts > 0).reshape(rings, sectors)
    return mask

class PolarLAB:
    """Polar grid representation of an image in CIELAB color space."""

    _base_masks = {}  # Cache per (size, rings, sectors)

    def __init__(self, data: np.ndarray, rings: int, sectors: int, original_size: int = None):
        self.data = data
        self.rings = rings
        self.sectors = sectors
        self.original_size = original_size

    @classmethod
    def _get_base_mask(cls, size, rings, sectors):
        key = (size, rings, sectors)
        if key not in cls._base_masks:
            cls._base_masks[key] = get_mask_for_rotation(size, rings, sectors)
        return cls._base_masks[key]

    @classmethod
    def from_mask(cls, mask: np.ndarray, original_size: int = None) -> "PolarLAB":
        """Create PolarLAB from boolean mask (rings, sectors). True=white, False=black."""
        rings, sectors = mask.shape
        data = np.zeros((rings, sectors, 3), dtype=np.float32)
        data[mask, 0] = 100.0  # L=100 for white, L=0 for black; a=b=0
        return cls(data, rings, sectors, original_size)

    @classmethod
    def from_image(cls, image: np.ndarray, rings: int = 10, sectors: int = 72) -> "PolarLAB":
        """Create PolarLAB from an RGB image (H, W, 3)."""
        # Normalize to 0-1
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        lab = color.rgb2lab(image)
        h, w = image.shape[:2]
        center_y, center_x = h / 2, w / 2
        r_max = min(center_x, center_y)

        # Equal-area ring boundaries
        ring_bounds = r_max * np.sqrt(np.arange(1, rings + 1) / rings)

        # Pixel coordinates
        y, x = np.mgrid[0:h, 0:w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = np.mod(np.arctan2(y - center_y, x - center_x), 2 * np.pi)

        # Assign pixels to bins
        ring_idx = np.clip(np.searchsorted(ring_bounds, distance, side='left'), 0, rings - 1)
        sector_idx = np.clip((angle / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)

        # Flatten for accumulation
        flat_idx = (ring_idx * sectors + sector_idx).ravel()
        flat_lab = lab.reshape(-1, 3)

        # Accumulate with np.add.at (single pass)
        bins_flat = np.zeros((rings * sectors, 3), dtype=np.float64)
        counts = np.zeros(rings * sectors, dtype=np.int32)
        np.add.at(bins_flat, flat_idx, flat_lab)
        np.add.at(counts, flat_idx, 1)

        # Average non-empty bins
        nonempty = counts > 0
        bins_flat[nonempty] /= counts[nonempty, np.newaxis]
        bins = bins_flat.reshape(rings, sectors, 3)

        # Fill empty bins from neighbors
        empty_indices = np.where(counts.reshape(rings, sectors) == 0)
        for r, s in zip(*empty_indices):
            neighbors = []
            for dr, ds in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, ns = r + dr, (s + ds) % sectors
                if 0 <= nr < rings and counts[nr * sectors + ns] > 0:
                    neighbors.append(bins[nr, ns])
            if neighbors:
                bins[r, s] = np.mean(neighbors, axis=0)

        return cls(bins.astype(np.float32), rings, sectors, h)


    @classmethod
    def from_image_padded(cls, image: np.ndarray, rings: int = 10, sectors: int = 72) -> "PolarLAB":
        """Create PolarLAB from an RGB image with circumscribed padding."""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        assert h == w, "Image must be square"
        size = h

        diag, r_max, ring_bounds = polar_geometry(size, rings, sectors)
        pad = (diag - size) // 2

        # Pad and convert
        padded = np.zeros((diag, diag, 3), dtype=np.float32)
        padded[pad:pad+size, pad:pad+size] = image
        lab = color.rgb2lab(padded)

        # Validity mask
        valid_pixels = np.zeros((diag, diag), dtype=bool)
        valid_pixels[pad:pad+size, pad:pad+size] = True

        # Coordinates
        center = diag / 2
        y, x = np.mgrid[0:diag, 0:diag]
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        angle = np.mod(np.arctan2(y - center, x - center), 2 * np.pi)

        ring_idx = np.clip(np.searchsorted(ring_bounds, distance, side='left'), 0, rings - 1)
        sector_idx = np.clip((angle / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)

        flat_idx = (ring_idx * sectors + sector_idx).ravel()
        flat_lab = lab.reshape(-1, 3)
        flat_valid = valid_pixels.ravel()

        # Accumulate only valid pixels
        bins_flat = np.zeros((rings * sectors, 3), dtype=np.float64)
        counts = np.zeros(rings * sectors, dtype=np.int32)

        np.add.at(bins_flat, flat_idx[flat_valid], flat_lab[flat_valid])
        np.add.at(counts, flat_idx[flat_valid], 1)

        nonempty = counts > 0
        bins_flat[nonempty] /= counts[nonempty, np.newaxis]
        bins_flat[~nonempty, 0] = 100.0 #white bg
        bins = bins_flat.reshape(rings, sectors, 3)

        return cls(bins.astype(np.float32), rings, sectors, original_size=size)


    def to_image(self, size: int = 128) -> np.ndarray:
        """Reconstruct an RGB image from PolarLAB representation.

        Returns uint8 array of shape (size, size, 3).
        """
        center = size / 2
        y, x = np.mgrid[0:size, 0:size]
        dx, dy = x - center, y - center

        distance = np.sqrt(dx**2 + dy**2)
        angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)

        ring_bounds = center * np.sqrt(np.arange(1, self.rings + 1) / self.rings)

        ring_idx = np.clip(np.searchsorted(ring_bounds, distance, side='left'), 0, self.rings - 1)
        sector_idx = np.clip((angle / (2 * np.pi) * self.sectors).astype(int), 0, self.sectors - 1)

        lab_img = self.data[ring_idx, sector_idx]
        rgb = color.lab2rgb(lab_img)
        return (rgb * 255).clip(0, 255).astype(np.uint8)


    def mask(self, rotation_steps: int, fill_value: float = 0.0) -> "PolarLAB":
        if self.original_size is None:
            raise ValueError("original_size required for masking")

        base_mask = self._get_base_mask(self.original_size, self.rings, self.sectors)
        validity = np.roll(base_mask, -rotation_steps, axis=1)

        result = self.data.copy()
        result[~validity, 0] = fill_value
        return PolarLAB(result, self.rings, self.sectors, self.original_size)

    def rotate(self, steps: int) -> "PolarLAB":
        """Return rotated copy. O(1) via index shift."""
        return PolarLAB(np.roll(self.data, -steps, axis=1), self.rings, self.sectors, self.original_size)

    def flatten(self) -> np.ndarray:
        """Flatten to 1D vector for FAISS. Shape: (rings * sectors * 3,)"""
        return self.data.ravel().astype(np.float32)

    def distance(self, other: "PolarLAB") -> float:
        """Euclidean distance (ΔE76 summed across bins)."""
        return float(np.linalg.norm(self.data - other.data))
