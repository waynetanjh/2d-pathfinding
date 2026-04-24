# 2D Pathfinding

Given a black-and-white image representing a 2D universe, this project finds paths between points that only cross black pixels. White pixels are obstacles, black pixels are traversable.

## Project structure

```
├── pathfinder.py       # Core pathfinding functions (BFS, visualization, non-intersecting paths)
├── test_pathfinder.py             # 35 pytest test cases
├── utils.py            # Test image downloader
├── requirements.txt    # Dependencies (numpy, imageio, Pillow, pytest)
├── test_images/        # Test images (bars, polygons, small-ring)
├── Makefile            # Shortcuts for setup, test, clean
├── README.md
└── .gitignore
```

## How it works

1. **`load_universe`** reads the image and converts it to a binary grid (`0` = black/walkable, `255` = white/wall)
2. **`find_path`** converts the `(x, y)` coordinates to `(row, col)` and calls **`bfs`**, which runs breadth-first search on a 4-connected pixel grid (up, down, left, right). BFS explores neighbors level by level, guaranteeing the shortest path. When the end pixel is reached, it backtracks through a parent map to reconstruct the path
3. **`path_exists`** calls `find_path` and returns `True` if a path was found
4. **`find_non_intersecting_paths`** calls `bfs` to find the first path, marks those pixels as blocked, then calls `bfs` again for the second path. If the second path can't get through, it tries the reverse order (second path first, then first path)
5. **`save_path_image`** converts the grayscale grid to RGB and paints each path's pixels in a different color, then writes the image to disk

## Time complexity

All functions run in **O(H x W)** time and space, where H x W is the number of pixels. BFS visits each pixel at most once, and each pixel has at most 4 neighbors.

This is optimal — you can't do better than O(H x W) worst case because you may need to visit every pixel. BFS also guarantees the shortest path on an unweighted grid. A* with a Manhattan distance heuristic would explore fewer pixels on average, but the worst case is the same.

For non-intersecting paths, the greedy approach (find one path, block it, find the other) is a heuristic. The general vertex-disjoint paths problem is NP-hard on arbitrary graphs, so the greedy strategy with both-orders fallback is a practical tradeoff.

## Setup

```bash
make setup
source venv/bin/activate
```

`make setup` creates the virtual environment and installs dependencies. You then need to activate it in your terminal with `source venv/bin/activate`.

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Functions

### 1. `load_universe(path) -> np.ndarray`

Loads a black-and-white image and converts it to a binary 2D array where `0` = traversable (black) and `255` = obstacle (white). This is the first step — everything else operates on this array.

```python
from pathfinder import load_universe

universe = load_universe("polygons.png")
```

### 2. `bfs(universe, start, end, blocked_mask=None) -> list | None`

The core BFS algorithm. Takes `(row, col)` coordinates and searches the 4-connected grid for the shortest path on black pixels. Returns a list of `(row, col)` tuples or `None`. This is called internally by `find_path`, `path_exists`, and `find_non_intersecting_paths`.

```python
from pathfinder import bfs, load_universe

universe = load_universe("polygons.png")
path = bfs(universe, start=(0, 0), end=(99, 99))
```

### 3. `find_black_pixel(universe, region) -> (x, y)`

Finds the first black pixel in a region of the image. Region can be `"top_left"`, `"top_right"`, `"bottom_left"`, or `"bottom_right"`. Useful for picking start/end points automatically.

```python
from pathfinder import load_universe, find_black_pixel

universe = load_universe("polygons.png")
start = find_black_pixel(universe, "top_left")      # (0, 0)
end = find_black_pixel(universe, "bottom_right")     # (99, 99)
```

### 4. `find_path(universe, start_x, start_y, end_x, end_y) -> list | None`

The core function. Runs BFS to find the shortest path between two points, returning a list of `(x, y)` coordinates. Returns `None` if no path exists.

```python
from pathfinder import load_universe, find_path

universe = load_universe("polygons.png")
path = find_path(universe, 0, 0, 99, 99)
# [(0, 0), (0, 1), (0, 2), ..., (99, 99)]
```

### 5. `path_exists(universe, start_x, start_y, end_x, end_y) -> bool`

Convenience wrapper around `find_path`. Returns `True` if a path exists, `False` otherwise.

```python
from pathfinder import load_universe, path_exists

universe = load_universe("polygons.png")
print(path_exists(universe, 0, 0, 99, 99))  # True
```

### 6. `find_non_intersecting_paths(universe, ...) -> (list | None, list | None)`

Given two pairs of points, finds two paths that don't share any pixels. Calls `find_path` for the first pair, blocks those pixels, then finds the second. If that fails, tries the reverse order. Returns `(None, None)` if not possible.

```python
from pathfinder import load_universe, find_non_intersecting_paths

universe = load_universe("polygons.png")
path1, path2 = find_non_intersecting_paths(universe, 0, 0, 99, 0, 0, 99, 99, 99)
```

### 7. `save_path_image(universe, paths, output_path)`

Takes the results from `find_path` or `find_non_intersecting_paths` and draws them onto the universe image. Paths are colored blue, red, green, yellow in order. Saves the result to disk.

```python
from pathfinder import load_universe, find_path, save_path_image

universe = load_universe("polygons.png")
path = find_path(universe, 0, 0, 99, 99)
save_path_image(universe, [path], "output.png")
```

## Tests

```bash
make test
```

Or manually:

```bash
pytest test_pathfinder.py -v
```

To delete the virtual environment, caches, and output images:

```bash
make clean
```

Test images are automatically downloaded from [mcollinswisc/2D_paths](https://github.com/mcollinswisc/2D_paths).
