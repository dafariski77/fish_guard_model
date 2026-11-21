"""
Unit Test: CIELab Preprocessor and Quality Checks
Validates:
- Laplacian variance blur detection
- Exposure luminance checks
- CIELab normalization ranges
- Aspect ratio crop behavior
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir.parent))

from fish_guard_model.models.cielab_preprocessor import CIELabPreprocessor


@pytest.fixture
def preprocessor():
    return CIELabPreprocessor()


def test_blur_detection(preprocessor):
    # Sharp image with high-frequency edges
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp_img[::4, ::4] = 255
    is_sharp, var_val = preprocessor.check_blur(sharp_img)
    assert is_sharp is True
    assert var_val > 100.0

    # Blurred image (Gaussian blur)
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)
    is_sharp_blur, blur_var = preprocessor.check_blur(blurred_img)
    assert is_sharp_blur is False
    assert blur_var < 100.0


def test_exposure_detection(preprocessor):
    # Normal exposure
    normal_img = np.full((100, 100, 3), 120, dtype=np.uint8)
    ok, lum, _ = preprocessor.check_exposure(normal_img)
    assert ok is True
    assert 40.0 <= lum <= 225.0

    # Too dark
    dark_img = np.full((100, 100, 3), 20, dtype=np.uint8)
    ok_dark, lum_dark, msg_dark = preprocessor.check_exposure(dark_img)
    assert ok_dark is False
    assert "TOO_DARK" in msg_dark

    # Too bright / glare
    glare_img = np.full((100, 100, 3), 245, dtype=np.uint8)
    ok_glare, lum_glare, msg_glare = preprocessor.check_exposure(glare_img)
    assert ok_glare is False
    assert "TOO_BRIGHT" in msg_glare


def test_cielab_normalization_range(preprocessor):
    # Test across color extremes: black, white, red, green, blue
    rgb_test = np.array([
        [[0, 0, 0], [255, 255, 255]],
        [[255, 0, 0], [0, 255, 0]]
    ], dtype=np.uint8)

    lab_norm = preprocessor.rgb_to_cielab(rgb_test)

    # All normalized channels should be bounded within [-1.5, 1.5]
    assert np.all(lab_norm >= -1.5)
    assert np.all(lab_norm <= 1.5)
    assert lab_norm.shape == (2, 2, 3)
    assert lab_norm.dtype == np.float32


def test_crop_and_resize(preprocessor):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 100
    bbox = [0.2, 0.3, 0.4, 0.3]  # [x, y, w, h]

    cropped = preprocessor.crop_and_resize(img, bbox, padding=0.1, target_size=(224, 224))
    assert cropped.shape == (224, 224, 3)
