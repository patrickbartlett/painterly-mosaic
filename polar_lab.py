import numpy as np
from skimage import color


class PolarLAB:
    """Polar grid representation of an image in CIELAB color space."""

    def __init__(self, data: np.ndarray, rings: int, sectors: int):
        self.data = data
        self.rings = rings
        self.sectors = sectors

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

        return cls(bins.astype(np.float32), rings, sectors)

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

    def rotate(self, steps: int) -> "PolarLAB":
        """Return rotated copy. O(1) via index shift."""
        return PolarLAB(np.roll(self.data, -steps, axis=1), self.rings, self.sectors)

    def flatten(self) -> np.ndarray:
        """Flatten to 1D vector for FAISS. Shape: (rings * sectors * 3,)"""
        return self.data.ravel().astype(np.float32)

    def distance(self, other: "PolarLAB") -> float:
        """Euclidean distance (ΔE76 summed across bins)."""
        return float(np.linalg.norm(self.data - other.data))
