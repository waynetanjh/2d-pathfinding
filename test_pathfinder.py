"""Tests for 2D pathfinding."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pathfinder import (
    find_black_pixel,
    find_non_intersecting_paths,
    find_path,
    load_universe,
    path_exists,
    save_path_image,
)
from utils import download_test_images


# 1. Helpers

def assert_path_is_valid(universe, path):
    """Assert that every pixel in the path is black and each step is 4-adjacent."""
    assert path is not None, "Expected a path but got None"
    assert len(path) >= 1

    for x, y in path:
        assert universe[y, x] == 0, f"Path pixel ({x}, {y}) is not black"

    for step in range(len(path) - 1):
        x_diff = abs(path[step + 1][0] - path[step][0])
        y_diff = abs(path[step + 1][1] - path[step][1])
        assert (x_diff + y_diff) == 1, f"Non-adjacent step at index {step}"


def make_ring_universe(size, center, outer_radius, inner_radius):
    """Create a ring-shaped white obstacle on a black background."""
    universe = np.zeros((size, size), dtype=np.uint8)
    for row in range(size):
        for col in range(size):
            distance = ((row - center) ** 2 + (col - center) ** 2) ** 0.5
            if inner_radius <= distance <= outer_radius:
                universe[row, col] = 255
    return universe


# 2. path_exists: basic connectivity

class TestPathExists:
    def test_open_grid(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        assert path_exists(grid, 0, 0, 4, 4)

    def test_full_wall_blocks_path(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, :] = 255
        assert not path_exists(grid, 0, 0, 4, 4)

    def test_wall_with_gap(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 0:4] = 255
        assert path_exists(grid, 0, 0, 4, 0)

    def test_start_on_white_pixel(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[0, 0] = 255
        assert not path_exists(grid, 0, 0, 4, 4)

    def test_end_on_white_pixel(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[4, 4] = 255
        assert not path_exists(grid, 0, 0, 4, 4)

    def test_same_start_and_end(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        assert path_exists(grid, 2, 2, 2, 2)

    def test_all_white_image(self):
        grid = np.full((5, 5), 255, dtype=np.uint8)
        assert not path_exists(grid, 0, 0, 4, 4)

    def test_single_pixel_image(self):
        grid = np.zeros((1, 1), dtype=np.uint8)
        assert path_exists(grid, 0, 0, 0, 0)

    def test_out_of_bounds_raises(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            path_exists(grid, 10, 10, 0, 0)

    def test_disconnected_islands(self):
        grid = np.full((5, 5), 255, dtype=np.uint8)
        grid[0, 0] = 0
        grid[4, 4] = 0
        assert not path_exists(grid, 0, 0, 4, 4)


# 3. path_exists: corridors and bottlenecks

class TestCorridorsAndBottlenecks:
    def test_narrow_one_pixel_corridor(self):
        grid = np.full((5, 5), 255, dtype=np.uint8)
        grid[0, :] = 0
        assert path_exists(grid, 0, 0, 4, 0)

    def test_single_pixel_bottleneck(self):
        grid = np.full((5, 5), 255, dtype=np.uint8)
        grid[0, 0] = 0
        grid[0, 1] = 0
        grid[1, 1] = 0
        grid[2, 1] = 0
        grid[2, 2] = 0
        assert path_exists(grid, 0, 0, 2, 2)

    def test_maze_pattern(self):
        grid = np.zeros((7, 7), dtype=np.uint8)
        grid[1, 0:6] = 255
        grid[3, 1:7] = 255
        grid[5, 0:6] = 255
        assert path_exists(grid, 0, 0, 6, 6)


# 4. path_exists: ring obstacle (from problem description screenshot)

class TestRingObstacle:
    @pytest.fixture
    def ring(self):
        return make_ring_universe(size=50, center=25, outer_radius=15, inner_radius=12)

    def test_inside_to_outside_blocked(self, ring):
        assert not path_exists(ring, 25, 25, 0, 0)

    def test_outside_to_outside_connected(self, ring):
        assert path_exists(ring, 2, 2, 48, 48)

    def test_path_around_ring_is_valid(self, ring):
        path = find_path(ring, 2, 2, 48, 48)
        assert_path_is_valid(ring, path)


# 5. find_path: path correctness

class TestFindPath:
    def test_returns_correct_endpoints(self):
        grid = np.zeros((3, 3), dtype=np.uint8)
        path = find_path(grid, 0, 0, 2, 2)
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_all_steps_are_adjacent(self):
        grid = np.zeros((3, 3), dtype=np.uint8)
        path = find_path(grid, 0, 0, 2, 2)
        assert_path_is_valid(grid, path)

    def test_path_stays_on_black_pixels(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[5, 2:8] = 255
        path = find_path(grid, 0, 0, 9, 9)
        assert_path_is_valid(grid, path)

    def test_maze_path_is_valid(self):
        grid = np.zeros((7, 7), dtype=np.uint8)
        grid[1, 0:6] = 255
        grid[3, 1:7] = 255
        grid[5, 0:6] = 255
        path = find_path(grid, 0, 0, 6, 6)
        assert_path_is_valid(grid, path)

    def test_returns_none_when_no_path(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, :] = 255
        assert find_path(grid, 0, 0, 4, 4) is None


# 6. find_non_intersecting_paths

class TestNonIntersectingPaths:
    def test_open_grid_finds_two_paths(self):
        grid = np.zeros((5, 10), dtype=np.uint8)
        path1, path2 = find_non_intersecting_paths(grid, 0, 0, 9, 0, 0, 4, 9, 4)
        assert path1 is not None
        assert path2 is not None

    def test_paths_share_no_pixels(self):
        grid = np.zeros((5, 10), dtype=np.uint8)
        path1, path2 = find_non_intersecting_paths(grid, 0, 0, 9, 0, 0, 4, 9, 4)
        assert len(set(path1) & set(path2)) == 0

    def test_both_paths_are_valid(self):
        grid = np.zeros((5, 10), dtype=np.uint8)
        path1, path2 = find_non_intersecting_paths(grid, 0, 0, 9, 0, 0, 4, 9, 4)
        assert_path_is_valid(grid, path1)
        assert_path_is_valid(grid, path2)

    def test_impossible_in_single_pixel_corridor(self):
        grid = np.full((3, 5), 255, dtype=np.uint8)
        grid[1, :] = 0
        path1, path2 = find_non_intersecting_paths(grid, 0, 1, 4, 1, 0, 1, 4, 1)
        assert path1 is None or path2 is None

    def test_around_ring(self):
        ring = make_ring_universe(size=50, center=25, outer_radius=15, inner_radius=12)
        path1, path2 = find_non_intersecting_paths(ring, 2, 2, 48, 2, 2, 48, 48, 48)
        assert path1 is not None and path2 is not None
        assert len(set(path1) & set(path2)) == 0
        assert_path_is_valid(ring, path1)
        assert_path_is_valid(ring, path2)


# 7. save_path_image: visualization

class TestVisualization:
    def test_saves_file(self, tmp_path):
        grid = np.zeros((5, 10), dtype=np.uint8)
        path = find_path(grid, 0, 0, 9, 0)
        output = str(tmp_path / "viz.png")
        save_path_image(grid, [path], output)
        assert os.path.exists(output)

    def test_ring_visualization(self, tmp_path):
        ring = make_ring_universe(size=50, center=25, outer_radius=15, inner_radius=12)
        path = find_path(ring, 2, 2, 48, 48)
        output = str(tmp_path / "ring.png")
        save_path_image(ring, [path], output)
        assert os.path.exists(output)


# 8. Real images from github.com/mcollinswisc/2D_paths

class TestRealImages:
    @pytest.fixture(scope="class")
    def image_dir(self, tmp_path_factory):
        return str(tmp_path_factory.mktemp("test_images"))

    @pytest.fixture(scope="class")
    def images(self, image_dir):
        try:
            return download_test_images(image_dir)
        except Exception:
            pytest.skip("Could not download test images")

    def test_bars_path_exists(self, images):
        universe = load_universe(images[0])
        top_left = find_black_pixel(universe, "top_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        assert path_exists(universe, *top_left, *bottom_right)

    def test_bars_path_is_valid(self, images):
        universe = load_universe(images[0])
        top_left = find_black_pixel(universe, "top_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        path = find_path(universe, *top_left, *bottom_right)
        assert_path_is_valid(universe, path)

    def test_polygons_path_exists(self, images):
        universe = load_universe(images[1])
        top_left = find_black_pixel(universe, "top_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        assert path_exists(universe, *top_left, *bottom_right)

    def test_polygons_path_is_valid(self, images):
        universe = load_universe(images[1])
        top_left = find_black_pixel(universe, "top_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        path = find_path(universe, *top_left, *bottom_right)
        assert_path_is_valid(universe, path)

    def test_polygons_non_intersecting_paths(self, images):
        universe = load_universe(images[1])
        top_left = find_black_pixel(universe, "top_left")
        top_right = find_black_pixel(universe, "top_right")
        bottom_left = find_black_pixel(universe, "bottom_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        path1, path2 = find_non_intersecting_paths(
            universe, *top_left, *top_right, *bottom_left, *bottom_right)
        assert path1 is not None and path2 is not None
        assert len(set(path1) & set(path2)) == 0
        assert_path_is_valid(universe, path1)
        assert_path_is_valid(universe, path2)

    def test_small_ring_path_exists(self, images):
        universe = load_universe(images[2])
        top_left = find_black_pixel(universe, "top_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        assert path_exists(universe, *top_left, *bottom_right)

    def test_small_ring_path_is_valid(self, images):
        universe = load_universe(images[2])
        top_left = find_black_pixel(universe, "top_left")
        bottom_right = find_black_pixel(universe, "bottom_right")
        path = find_path(universe, *top_left, *bottom_right)
        assert_path_is_valid(universe, path)
