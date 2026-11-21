"""
CIELab Preprocessor and Quality Assurance Module for FishGuard.
Performs pre-capture quality checks (blur, exposure) and lighting-invariant
CIELab color space transformations to decouple illumination (L*) from
biological freshness markers (a* gill redness, b* eye cloudiness).
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional
import torch


class CIELabPreprocessor:
    """
    Handles image quality validation and CIELab color transformations
    tailored for wet-market lighting environments.
    """

    def __init__(
        self,
        min_blur_threshold: float = 100.0,
        min_luminance: float = 40.0,
        max_luminance: float = 225.0,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.min_blur_threshold = min_blur_threshold
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.target_size = target_size

    def check_blur(self, image_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Calculates Laplacian variance to detect out-of-focus or motion-blurred captures.
        Returns: (is_sharp, variance)
        """
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return laplacian_var >= self.min_blur_threshold, laplacian_var

    def check_exposure(self, image_bgr: np.ndarray) -> Tuple[bool, float, str]:
        """
        Assesses exposure using average luminance.
        Returns: (is_good_exposure, avg_luminance, status_message)
        """
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        avg_lum = float(np.mean(gray))

        if avg_lum < self.min_luminance:
            return False, avg_lum, "TOO_DARK: Market environment too dim. Turn on torch/flash."
        elif avg_lum > self.max_luminance:
            return False, avg_lum, "TOO_BRIGHT: Direct glare or overexposure detected."
        else:
            return True, avg_lum, "EXPOSURE_OK"

    def validate_image_quality(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs comprehensive pre-capture quality checks.
        """
        sharp, blur_val = self.check_blur(image_bgr)
        exp_ok, lum_val, exp_msg = self.check_exposure(image_bgr)

        passed = sharp and exp_ok
        return {
            "passed": passed,
            "blur_metric": {
                "passed": sharp,
                "laplacian_variance": round(blur_val, 2),
                "threshold": self.min_blur_threshold
            },
            "exposure_metric": {
                "passed": exp_ok,
                "average_luminance": round(lum_val, 2),
                "status": exp_msg
            }
        }

    def rgb_to_cielab(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Converts RGB float32/uint8 image to normalized CIELab float32 tensor.
        Output range:
        L* normalized: (L - 50.0) / 50.0 -> [-1.0, 1.0]
        a* normalized: a / 128.0         -> [-1.0, 1.0]
        b* normalized: b / 128.0         -> [-1.0, 1.0]
        """
        if rgb_img.dtype != np.uint8:
            rgb_uint8 = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_img

        lab_raw = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        # OpenCV maps: L in [0, 255] representing [0, 100], a in [0, 255] representing [-127, 127], b similarly
        # Standard OpenCV scaling: L = L * 100 / 255; a = a - 128; b = b - 128
        l_std = lab_raw[:, :, 0] * (100.0 / 255.0)
        a_std = lab_raw[:, :, 1] - 128.0
        b_std = lab_raw[:, :, 2] - 128.0

        # Normalize to [-1.0, 1.0]
        l_norm = (l_std - 50.0) / 50.0
        a_norm = a_std / 128.0
        b_norm = b_std / 128.0

        normalized_lab = np.stack([l_norm, a_norm, b_norm], axis=-1)
        return normalized_lab.astype(np.float32)

    def crop_and_resize(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        padding: float = 0.15,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Crops a region given normalized bbox [x_min, y_min, width, height] in 0..1,
        applies padding, clamps to image boundaries, and resizes to target_size.
        """
        size = target_size or self.target_size
        h, w = image.shape[:2]
        x_min, y_min, box_w, box_h = bbox

        # Denormalize
        x1 = x_min * w
        y1 = y_min * h
        bw = box_w * w
        bh = box_h * h

        # Add contextual padding
        pad_x = bw * padding
        pad_y = bh * padding

        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(w, int(round(x1 + bw + pad_x)))
        cy2 = min(h, int(round(y1 + bh + pad_y)))

        if cx2 <= cx1 or cy2 <= cy1:
            # Fallback to center crop if bbox is degenerate
            crop = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            return crop

        crop = image[cy1:cy2, cx1:cx2]
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
        return resized

    def preprocess_for_detector(
        self,
        image_rgb: np.ndarray,
        target_size: Tuple[int, int] = (320, 320)
    ) -> Tuple[torch.Tensor, Tuple[float, float, int, int]]:
        """
        Prepares whole fish RGB image for Stage 1 YOLO-Nano detector:
        Letterboxes image with aspect-ratio preservation, normalizes to ImageNet stats.
        Returns: (tensor [1, 3, 320, 320], (gain, pad_x, pad_y, orig_w, orig_h))
        """
        h0, w0 = image_rgb.shape[:2]
        tw, th = target_size
        r = min(tw / w0, th / h0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw = (tw - new_unpad[0]) / 2
        dh = (th - new_unpad[1]) / 2

        resized = cv2.resize(image_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize to [0, 1] then ImageNet mean/std
        img = letterboxed.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Transpose to [C, H, W] and add batch dim
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        meta = (r, left, top, w0, h0)
        return tensor, meta

    def preprocess_for_classifier(
        self,
        whole_fish_rgb: np.ndarray,
        eye_crop_rgb: np.ndarray,
        gill_crop_rgb: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares multi-task classifier inputs:
        - Whole fish: RGB normalized with ImageNet stats
        - Eye crop: CIELab normalized [-1, 1]
        - Gill crop: CIELab normalized [-1, 1]
        """
        # 1. Whole fish RGB
        wf_resized = cv2.resize(whole_fish_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        wf_norm = (wf_resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        wf_tensor = torch.from_numpy(wf_norm.transpose(2, 0, 1)).unsqueeze(0).float()

        # 2. Eye crop CIELab
        eye_resized = cv2.resize(eye_crop_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        eye_lab = self.rgb_to_cielab(eye_resized)
        eye_tensor = torch.from_numpy(eye_lab.transpose(2, 0, 1)).unsqueeze(0).float()

        # 3. Gill crop CIELab
        gill_resized = cv2.resize(gill_crop_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        gill_lab = self.rgb_to_cielab(gill_resized)
        gill_tensor = torch.from_numpy(gill_lab.transpose(2, 0, 1)).unsqueeze(0).float()

        return {
            "whole_fish": wf_tensor,
            "eye_lab": eye_tensor,
            "gill_lab": gill_tensor
        }
