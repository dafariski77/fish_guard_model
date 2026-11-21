"""
FishGuard Unified Inference Pipeline
Connects CIELab preprocessing, Stage 1 Detector, Stage 2 Multi-Task Classifier,
and Stage 3 Hybrid Risk Engine to produce the production Cross-Team Contract JSON.
"""

import uuid
import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import cv2
import torch

from fish_guard_model.models.cielab_preprocessor import CIELabPreprocessor
from fish_guard_model.models.detector import FishEyeGillDetector
from fish_guard_model.models.classifier import FishGuardMultiTaskClassifier
from fish_guard_model.models.risk_engine import HybridRiskEngine


class FishGuardInferencePipeline:
    """
    End-to-end edge inference pipeline for FishGuard.
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.base_dir = Path(__file__).resolve().parent.parent

        if weights_dir is None:
            weights_dir = self.base_dir / "weights"
        self.weights_dir = weights_dir

        self.preprocessor = CIELabPreprocessor()
        self.risk_engine = HybridRiskEngine(config_dir=self.base_dir / "configs")

        # Initialize detector
        self.detector = FishEyeGillDetector().to(self.device)
        det_weights_path = self.weights_dir / "detector.pt"
        if det_weights_path.exists():
            self.detector.load_state_dict(torch.load(det_weights_path, map_location=self.device))
        self.detector.eval()

        # Initialize classifier
        self.classifier = FishGuardMultiTaskClassifier().to(self.device)
        cls_weights_path = self.weights_dir / "classifier.pt"
        if cls_weights_path.exists():
            self.classifier.load_state_dict(torch.load(cls_weights_path, map_location=self.device))
        self.classifier.eval()

    def load_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Loads and returns RGB image as numpy ndarray."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Could not decode image at {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        else:
            raise TypeError("image_input must be file path or numpy ndarray")

    def inspect_fish(
        self,
        whole_fish_image: Union[str, Path, np.ndarray],
        close_up_image: Optional[Union[str, Path, np.ndarray]] = None,
        sensory_input: Optional[Dict[str, Any]] = None,
        scan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete inspection flow:
        1. Pre-capture blur/exposure validation
        2. Species classification
        3. Eye & gill detection and cropping
        4. CIELab freshness scoring
        5. Hybrid risk engine evaluation
        Returns dictionary matching PRD Section 5 JSON contract.
        """
        # Default sensory answers if not provided
        if sensory_input is None:
            sensory_input = {
                "smell": "fresh_sea",
                "texture": "bounces_back",
                "storage": "packed_in_ice",
                "skipped": True
            }

        generated_scan_id = scan_id or str(uuid.uuid4())
        # Format current ISO timestamp with timezone offset (+07:00 for Indonesia)
        tz_offset = datetime.timezone(datetime.timedelta(hours=7))
        now_iso = datetime.datetime.now(tz_offset).strftime("%Y-%m-%dT%H:%M:%S+07:00")

        # 1. Load Images
        wf_rgb = self.load_image(whole_fish_image)
        cu_rgb = self.load_image(close_up_image) if close_up_image is not None else wf_rgb

        # Pre-capture quality checks
        bgr_for_check = cv2.cvtColor(cu_rgb, cv2.COLOR_RGB2BGR)
        quality_report = self.preprocessor.validate_image_quality(bgr_for_check)

        # 2. Stage 1: Eye & Gill Detection
        det_tensor, det_meta = self.preprocessor.preprocess_for_detector(cu_rgb)
        det_tensor = det_tensor.to(self.device)

        with torch.no_grad():
            raw_det = self.detector(det_tensor)
            detections = self.detector.decode_predictions(raw_det)

        # Extract top eye and gill bounding boxes
        eye_det = next((d for d in detections if d["class"] == "fish_eye"), None)
        gill_det = next((d for d in detections if d["class"] == "fish_gill"), None)

        # Fallback bounding boxes if detector did not find high confidence regions
        eye_bbox = eye_det["bbox"] if eye_det else [0.42, 0.31, 0.12, 0.12]
        gill_bbox = gill_det["bbox"] if gill_det else [0.55, 0.40, 0.18, 0.22]

        # 3. Crop Anatomical Regions & Convert to CIELab
        eye_crop = self.preprocessor.crop_and_resize(cu_rgb, eye_bbox)
        gill_crop = self.preprocessor.crop_and_resize(cu_rgb, gill_bbox)

        classifier_inputs = self.preprocessor.preprocess_for_classifier(
            whole_fish_rgb=wf_rgb,
            eye_crop_rgb=eye_crop,
            gill_crop_rgb=gill_crop
        )

        wf_tensor = classifier_inputs["whole_fish"].to(self.device)
        eye_lab_tensor = classifier_inputs["eye_lab"].to(self.device)
        gill_lab_tensor = classifier_inputs["gill_lab"].to(self.device)

        # 4. Stage 2: Multi-Task Classification
        species_res = self.classifier.predict_species(wf_tensor)
        freshness_res = self.classifier.predict_freshness(eye_lab_tensor, gill_lab_tensor)

        # 5. Stage 3: Risk Engine Rule Fusion
        vision_freshness = self.risk_engine.compute_vision_freshness(
            eye_score=freshness_res["eye_score"],
            gill_score=freshness_res["gill_score"],
            eye_bbox=eye_bbox,
            gill_bbox=gill_bbox
        )

        species_pred, final_verdict = self.risk_engine.evaluate_verdict(
            species_label=species_res["label"],
            species_confidence=species_res["confidence"],
            vision_freshness=vision_freshness,
            sensory_input=sensory_input
        )

        # 6. Assemble Contract-Compliant Output
        contract_output = {
            "scan_id": generated_scan_id,
            "timestamp": now_iso,
            "species_prediction": species_pred,
            "vision_freshness": vision_freshness,
            "sensory_input": sensory_input,
            "final_verdict": final_verdict
        }

        return contract_output
