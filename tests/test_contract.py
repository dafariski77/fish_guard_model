"""
Unit Test: Cross-Team Integration Contract Compliance (PRD Section 5)
Verifies that the inference pipeline output matches the exact contract schema
expected by the Flutter mobile application.
"""

import sys
import re
from pathlib import Path
import pytest
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir.parent))

from fish_guard_model.pipeline.inference_pipeline import FishGuardInferencePipeline


@pytest.fixture
def sample_images(tmp_path):
    # Create simple RGB test image
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    return img


def test_contract_schema_compliance(sample_images):
    pipeline = FishGuardInferencePipeline()

    sensory_sample = {
        "smell": "fresh_sea",
        "texture": "bounces_back",
        "storage": "no_ice_room_temp",
        "skipped": False
    }

    result = pipeline.inspect_fish(
        whole_fish_image=sample_images,
        close_up_image=sample_images,
        sensory_input=sensory_sample
    )

    # 1. Root Keys Check
    required_root_keys = [
        "scan_id", "timestamp", "species_prediction",
        "vision_freshness", "sensory_input", "final_verdict"
    ]
    for key in required_root_keys:
        assert key in result, f"Missing required contract key: {key}"

    # 2. scan_id format (UUID string)
    assert isinstance(result["scan_id"], str)
    assert len(result["scan_id"]) >= 32

    # 3. timestamp format (ISO 8601 with timezone offset)
    iso_regex = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+|-)\d{2}:\d{2}$"
    assert re.match(iso_regex, result["timestamp"]), f"Invalid ISO timestamp format: {result['timestamp']}"

    # 4. species_prediction object
    sp = result["species_prediction"]
    assert "label" in sp and isinstance(sp["label"], str)
    assert "display_name" in sp and isinstance(sp["display_name"], str)
    assert "confidence" in sp and isinstance(sp["confidence"], (float, int))
    assert 0.0 <= sp["confidence"] <= 1.0
    assert "intrinsic_risk" in sp
    assert sp["intrinsic_risk"] in ["HIGH_HISTAMINE", "CIGUATERA_WATCH", "LOW_INTRINSIC_RISK", "UNKNOWN_RISK"]

    # 5. vision_freshness object
    vf = result["vision_freshness"]
    assert "eye_bbox" in vf and len(vf["eye_bbox"]) == 4
    assert "gill_bbox" in vf and len(vf["gill_bbox"]) == 4
    for coord in vf["eye_bbox"] + vf["gill_bbox"]:
        assert isinstance(coord, (float, int))
        assert 0.0 <= coord <= 1.0, f"Bounding box coordinate {coord} out of range [0, 1]"

    assert "eye_score" in vf and 0.0 <= vf["eye_score"] <= 1.0
    assert "gill_score" in vf and 0.0 <= vf["gill_score"] <= 1.0
    assert "combined_freshness_score" in vf and 0.0 <= vf["combined_freshness_score"] <= 100.0
    assert "grade" in vf and vf["grade"] in ["SAFE", "CAUTION", "AVOID"]

    # Check mathematical formula: (0.45 * eye + 0.55 * gill) * 100
    expected_combined = round((0.45 * vf["eye_score"] + 0.55 * vf["gill_score"]) * 100.0, 1)
    assert abs(vf["combined_freshness_score"] - expected_combined) < 0.2

    # 6. sensory_input object
    si = result["sensory_input"]
    assert si["smell"] in ["fresh_sea", "sour_ammonia", "odorless"]
    assert si["texture"] in ["bounces_back", "mushy", "unnaturally_stiff"]
    assert si["storage"] in ["packed_in_ice", "no_ice_room_temp"]
    assert isinstance(si["skipped"], bool)

    # 7. final_verdict object
    fv = result["final_verdict"]
    assert fv["status"] in ["GREEN", "YELLOW", "RED"]
    assert "reasons" in fv and isinstance(fv["reasons"], list)
    assert len(fv["reasons"]) > 0
    for r in fv["reasons"]:
        assert isinstance(r, str) and len(r) > 0
    assert "recommendation" in fv and isinstance(fv["recommendation"], str)
    assert len(fv["recommendation"]) > 0
