"""Utility functions for downloading test images."""

from __future__ import annotations

import os
import urllib.request


def download_test_images(destination_dir: str) -> list[str]:
    """Download test images from the GitHub repo."""
    base_url = "https://raw.githubusercontent.com/mcollinswisc/2D_paths/main"
    image_names = ["bars.png", "polygons.png", "small-ring.png"]
    downloaded_paths = []
    for name in image_names:
        destination = os.path.join(destination_dir, name)
        if not os.path.exists(destination):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(f"{base_url}/{name}", destination)
        downloaded_paths.append(destination)
    return downloaded_paths
