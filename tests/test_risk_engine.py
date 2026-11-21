"""
Unit Test: Hybrid Risk Scoring Engine Deterministic Logic
Validates 100% of business rules, safety thresholds, and edge-case overrides:
- Green / Yellow / Red thresholds
- Scombroid/Histamine room-temperature penalties
- Formalin adulteration flags
- Bacterial decay flags
- Ciguatera advisory rules
- Sensory skipped disclaimer
"""

import sys
from pathlib import Path
import pytest

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir.parent))

from fish_guard_model.models.risk_engine import HybridRiskEngine


@pytest.fixture
def risk_engine():
    return HybridRiskEngine()


def test_green_verdict_fresh_iced_tongkol(risk_engine):
    # Tongkol with high eye and gill score, packed on ice
    vision = risk_engine.compute_vision_freshness(eye_score=0.90, gill_score=0.85)
    assert vision["combined_freshness_score"] >= 75.0
    assert vision["grade"] == "SAFE"

    sensory = {
        "smell": "fresh_sea",
        "texture": "bounces_back",
        "storage": "packed_in_ice",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("tongkol", 0.95, vision, sensory)
    assert verdict["status"] == "GREEN"
    assert "Safe to buy" in verdict["recommendation"]


def test_yellow_verdict_high_histamine_room_temp(risk_engine):
    # Tongkol that looks visually fresh, BUT kept without ice at room temperature
    vision = risk_engine.compute_vision_freshness(eye_score=0.88, gill_score=0.80)
    assert vision["combined_freshness_score"] >= 75.0

    sensory = {
        "smell": "fresh_sea",
        "texture": "bounces_back",
        "storage": "no_ice_room_temp",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("tongkol", 0.94, vision, sensory)
    # Must cap at YELLOW due to invisible histamine accumulation risk
    assert verdict["status"] == "YELLOW"
    assert any("high-histamine" in r for r in verdict["reasons"])
    assert any("without ice" in r for r in verdict["reasons"])
    assert "Cook immediately" in verdict["recommendation"]


def test_red_verdict_high_histamine_room_temp_degraded(risk_engine):
    # Tongkol with moderate visual decline + room temp -> DROPS TO RED
    vision = risk_engine.compute_vision_freshness(eye_score=0.65, gill_score=0.60)
    assert vision["combined_freshness_score"] < 70.0

    sensory = {
        "smell": "fresh_sea",
        "texture": "bounces_back",
        "storage": "no_ice_room_temp",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("tongkol", 0.92, vision, sensory)
    assert verdict["status"] == "RED"
    assert any("histamine" in r.lower() for r in verdict["reasons"])


def test_red_verdict_formalin_flag(risk_engine):
    # Fish with high visual freshness (chemically preserved) BUT odorless + unnaturally stiff
    vision = risk_engine.compute_vision_freshness(eye_score=0.92, gill_score=0.90)
    assert vision["combined_freshness_score"] > 80.0

    sensory = {
        "smell": "odorless",
        "texture": "unnaturally_stiff",
        "storage": "no_ice_room_temp",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("bandeng", 0.88, vision, sensory)
    assert verdict["status"] == "RED"
    assert any("formalin" in r.lower() for r in verdict["reasons"])
    assert "chemical adulteration" in verdict["recommendation"].lower()


def test_red_verdict_bacterial_ammonia_smell(risk_engine):
    vision = risk_engine.compute_vision_freshness(eye_score=0.75, gill_score=0.75)
    sensory = {
        "smell": "sour_ammonia",
        "texture": "bounces_back",
        "storage": "packed_in_ice",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("kakap", 0.85, vision, sensory)
    assert verdict["status"] == "RED"
    assert any("ammonia" in r.lower() for r in verdict["reasons"])


def test_red_verdict_mushy_flesh(risk_engine):
    vision = risk_engine.compute_vision_freshness(eye_score=0.70, gill_score=0.70)
    sensory = {
        "smell": "fresh_sea",
        "texture": "mushy",
        "storage": "packed_in_ice",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("gurame", 0.90, vision, sensory)
    assert verdict["status"] == "RED"
    assert any("mushy" in r.lower() for r in verdict["reasons"])


def test_ciguatera_advisory_on_reef_fish(risk_engine):
    vision = risk_engine.compute_vision_freshness(eye_score=0.90, gill_score=0.88)
    sensory = {
        "smell": "fresh_sea",
        "texture": "bounces_back",
        "storage": "packed_in_ice",
        "skipped": False
    }

    species, verdict = risk_engine.evaluate_verdict("kerapu", 0.93, vision, sensory)
    assert verdict["status"] == "GREEN"
    assert any("reef predator" in r.lower() or "ciguatera" in r.lower() for r in verdict["reasons"])


def test_sensory_skipped_advisory(risk_engine):
    vision = risk_engine.compute_vision_freshness(eye_score=0.88, gill_score=0.85)
    sensory = {
        "smell": "fresh_sea",
        "texture": "bounces_back",
        "storage": "packed_in_ice",
        "skipped": True
    }

    species, verdict = risk_engine.evaluate_verdict("lele", 0.95, vision, sensory)
    assert verdict["status"] == "GREEN"
    assert any("skipped" in r.lower() for r in verdict["reasons"])
