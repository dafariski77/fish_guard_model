"""
Exports PyTorch FishGuard models to ONNX runtime format.
Verifies numerical equivalence and exports:
1. detector.onnx [1, 3, 320, 320] -> [1, 6, 2100]
2. species_classifier.onnx [1, 3, 224, 224] -> [1, 11]
3. freshness_grader.onnx [1, 3, 224, 224] -> [1, 1]
4. multitask_classifier.onnx [whole_fish, eye_crop, gill_crop]
"""

import sys
from pathlib import Path
import torch
import numpy as np
import onnx
import onnxruntime as ort

# Add parent directory to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir.parent))

from fish_guard_model.models.detector import FishEyeGillDetector
from fish_guard_model.models.classifier import (
    FishGuardMultiTaskClassifier,
    StandaloneSpeciesClassifier,
    StandaloneFreshnessGrader
)


def export_detector_onnx(weights_path: Path, output_path: Path):
    print("--- Exporting Detector to ONNX ---")
    model = FishEyeGillDetector()
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 320, 320, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["boxes_and_scores"],
        dynamic_axes={"input_image": {0: "batch_size"}, "boxes_and_scores": {0: "batch_size"}}
    )

    # Validate ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Verify with onnxruntime
    session = ort.InferenceSession(str(output_path))
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)

    with torch.no_grad():
        torch_outs = model(dummy_input).numpy()

    np.testing.assert_allclose(torch_outs, ort_outs[0], rtol=1e-3, atol=1e-4)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Detector ONNX verified & saved: {output_path} ({size_mb:.2f} MB)")


def export_classifier_onnx(weights_path: Path, export_dir: Path):
    print("--- Exporting Classifier Models to ONNX ---")
    base_model = FishGuardMultiTaskClassifier()
    if weights_path.exists():
        base_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    base_model.eval()

    # 1. Standalone Species Classifier
    species_model = StandaloneSpeciesClassifier(base_model)
    species_model.eval()
    dummy_species = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    species_path = export_dir / "species_classifier.onnx"

    torch.onnx.export(
        species_model,
        dummy_species,
        str(species_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["whole_fish_rgb"],
        output_names=["species_probabilities"],
        dynamic_axes={"whole_fish_rgb": {0: "batch_size"}, "species_probabilities": {0: "batch_size"}}
    )
    onnx.checker.check_model(onnx.load(str(species_path)))
    sess_spec = ort.InferenceSession(str(species_path))
    ort_spec = sess_spec.run(None, {sess_spec.get_inputs()[0].name: dummy_species.numpy()})[0]
    with torch.no_grad():
        torch_spec = species_model(dummy_species).numpy()
    np.testing.assert_allclose(torch_spec, ort_spec, rtol=1e-3, atol=1e-4)
    print(f"✓ Species Classifier ONNX verified: {species_path} ({species_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Standalone Freshness Grader
    freshness_model = StandaloneFreshnessGrader(base_model)
    freshness_model.eval()
    dummy_crop = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    freshness_path = export_dir / "freshness_grader.onnx"

    torch.onnx.export(
        freshness_model,
        dummy_crop,
        str(freshness_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["cielab_crop"],
        output_names=["freshness_score"],
        dynamic_axes={"cielab_crop": {0: "batch_size"}, "freshness_score": {0: "batch_size"}}
    )
    onnx.checker.check_model(onnx.load(str(freshness_path)))
    sess_fresh = ort.InferenceSession(str(freshness_path))
    ort_fresh = sess_fresh.run(None, {sess_fresh.get_inputs()[0].name: dummy_crop.numpy()})[0]
    with torch.no_grad():
        torch_fresh = freshness_model(dummy_crop).numpy()
    np.testing.assert_allclose(torch_fresh, ort_fresh, rtol=1e-3, atol=1e-4)
    print(f"✓ Freshness Grader ONNX verified: {freshness_path} ({freshness_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    weights = base_dir / "weights"
    export_detector_onnx(weights / "detector.pt", weights / "detector.onnx")
    export_classifier_onnx(weights / "classifier.pt", weights)
