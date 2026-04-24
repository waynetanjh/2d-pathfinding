"""
2D Pathfinding on Black-and-White Images

Given a 2D universe (black-and-white image), finds paths between points
that only cross black pixels. Supports visualization and non-intersecting
dual-path finding.
"""

from __future__ import annotations

import collections

import imageio.v3 as imageio
import numpy as np


def load_universe(path: str) -> np.ndarray:
    """Load an image and convert to binary (0=traversable, 255=obstacle)."""
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return np.where(img <= 127, 0, 255).astype(np.uint8)


def bfs(
    universe: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked_mask: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    """
    BFS on a 4-connected grid. Traverses black (0) pixels only.

    Args:
        universe: 2D array where 0=traversable, 255=obstacle.
        start: (row, col) start position.
        end: (row, col) end position.
        blocked_mask: Optional boolean array; True = additionally blocked.

    Returns:
        List of (row, col) from start to end, or None if no path exists.
    """
    rows, cols = universe.shape

    for label, (row, col) in [("Start", start), ("End", end)]:
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"{label} ({row}, {col}) is out of bounds for {rows}x{cols} image")

    if universe[start[0], start[1]] != 0 or universe[end[0], end[1]] != 0:
        return None

    if blocked_mask is not None:
        if blocked_mask[start[0], start[1]] or blocked_mask[end[0], end[1]]:
            return None

    if start == end:
        return [start]

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start[0], start[1]] = True
    parent = {}
    queue = collections.deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col = queue.popleft()
        for row_offset, col_offset in directions:
            neighbor_row = row + row_offset
            neighbor_col = col + col_offset

            if not (0 <= neighbor_row < rows and 0 <= neighbor_col < cols):
                continue
            if visited[neighbor_row, neighbor_col]:
                continue
            if universe[neighbor_row, neighbor_col] != 0:
                continue
            if blocked_mask is not None and blocked_mask[neighbor_row, neighbor_col]:
                continue

            visited[neighbor_row, neighbor_col] = True
            parent[(neighbor_row, neighbor_col)] = (row, col)

            if (neighbor_row, neighbor_col) == end:
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path

            queue.append((neighbor_row, neighbor_col))

    return None


def find_path(
    universe: np.ndarray,
    start_x: int, start_y: int,
    end_x: int, end_y: int,
) -> list[tuple[int, int]] | None:
    """
    Find a path between two points, returning it in (x, y) coordinates.

    Returns:
        List of (x, y) tuples from start to end, or None.
    """
    result = bfs(universe, (start_y, start_x), (end_y, end_x))
    if result is None:
        return None
    return [(col, row) for row, col in result]


def path_exists(
    universe: np.ndarray,
    start_x: int, start_y: int,
    end_x: int, end_y: int,
) -> bool:
    """
    Check if a path exists between two points on black pixels only.

    Args:
        universe: 2D binary array (0=traversable, 255=obstacle).
        start_x, start_y: Start point in (x, y) image coordinates.
        end_x, end_y: End point in (x, y) image coordinates.

    Returns:
        True if a path exists, False otherwise.
    """
    return find_path(universe, start_x, start_y, end_x, end_y) is not None


def find_non_intersecting_paths(
    universe: np.ndarray,
    start1_x: int, start1_y: int, end1_x: int, end1_y: int,
    start2_x: int, start2_y: int, end2_x: int, end2_y: int,
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """
    Find two non-intersecting paths for two point pairs.

    Tries finding path1 first then path2 with path1 blocked.
    If that fails, tries the reverse order.

    Returns:
        (path1, path2) in (x, y) coordinates, or None for unfound paths.
    """
    pairs = [
        ((start1_y, start1_x), (end1_y, end1_x), (start2_y, start2_x), (end2_y, end2_x)),
        ((start2_y, start2_x), (end2_y, end2_x), (start1_y, start1_x), (end1_y, end1_x)),
    ]

    for first_start, first_end, second_start, second_end in pairs:
        first_path = bfs(universe, first_start, first_end)
        if first_path is None:
            continue

        blocked = np.zeros(universe.shape, dtype=bool)
        for row, col in first_path:
            blocked[row, col] = True

        second_path = bfs(universe, second_start, second_end, blocked_mask=blocked)
        if second_path is None:
            continue

        path1 = [(col, row) for row, col in first_path]
        path2 = [(col, row) for row, col in second_path]
        # Return in the correct order (path1 for pair1, path2 for pair2)
        if first_start == (start1_y, start1_x):
            return path1, path2
        return path2, path1

    return None, None


def save_path_image(
    universe: np.ndarray,
    paths: list[list[tuple[int, int]]],
    output_path: str,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    """
    Save the universe with paths drawn on it.

    Args:
        universe: 2D binary array.
        paths: List of paths, each a list of (x, y) tuples.
        output_path: Output image file path.
        colors: RGB colors for each path. Defaults to blue then red.
    """
    if colors is None:
        colors = [(0, 100, 255), (255, 50, 50), (50, 255, 50), (255, 255, 0)]

    rgb = np.stack([universe, universe, universe], axis=-1)

    for path_index, path in enumerate(paths):
        color = colors[path_index % len(colors)]
        for x, y in path:
            rgb[y, x] = color

    imageio.imwrite(output_path, rgb)


def find_black_pixel(universe: np.ndarray, region: str) -> tuple[int, int]:
    """Find a black pixel in a given region of the image. Returns (x, y)."""
    rows, cols = universe.shape
    region_ranges = {
        "top_left":     (range(rows // 2),                range(cols // 2)),
        "top_right":    (range(rows // 2),                range(cols - 1, cols // 2, -1)),
        "bottom_left":  (range(rows - 1, rows // 2, -1), range(cols // 2)),
        "bottom_right": (range(rows - 1, rows // 2, -1), range(cols - 1, cols // 2, -1)),
    }

    row_range, col_range = region_ranges[region]
    for row in row_range:
        for col in col_range:
            if universe[row, col] == 0:
                return (col, row)

    raise ValueError(f"No black pixel found in {region}")
