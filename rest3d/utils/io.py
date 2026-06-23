import time
from typing import Any

import cv2
import numpy as np

from PIL import Image


def _pil_load(path: str, image_format: str) -> Image.Image:
    with Image.open(path) as img:
        if img is not None and image_format.lower() == "rgb":
            img = img.convert("RGB")
    return img


def _cv2_load(path: str, image_format: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None and image_format.lower() == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image(
    path: str,
    backend: str = "pil",
    image_format: str = "rgb",
    retry: int = 10,
) -> Any:
    for i_try in range(retry):
        if backend == "pil":
            img = _pil_load(path, image_format)
        elif backend == "cv2":
            img = _cv2_load(path, image_format)
        else:
            raise ValueError("Invalid backend {} for loading image.".format(backend))

        if img is not None:
            return img
        else:
            print("Reading {} failed. Will retry.".format(path))
            time.sleep(1.0)
        if i_try == retry - 1:
            raise Exception("Failed to load image {}".format(path))
