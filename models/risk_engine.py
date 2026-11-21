"""
Stage 3: Hybrid Risk Scoring Engine (Deterministic Rule-Based Fusion)
Fuses vision freshness scores, species intrinsic risk tier, and 10-second sensory checklist
into a legally sound, explainable GREEN / YELLOW / RED safety verdict.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class HybridRiskEngine:
    """
    Deterministic rule and scoring engine that guarantees explainable,
    reproducible safety verdicts compliant with the Cross-Team Contract.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).resolve().parent.parent / "configs"

        self.config_dir = config_dir
        with open(config_dir / "labels.json", "r", encoding="utf-8") as f:
            self.labels_cfg = json.load(f)
        with open(config_dir / "species_risk_config.json", "r", encoding="utf-8") as f:
            self.species_cfg = json.load(f)
        with open(config_dir / "risk_engine_rules.json", "r", encoding="utf-8") as f:
            self.rules_cfg = json.load(f)

        self.eye_weight = self.rules_cfg["vision_weights"]["eye_weight"]
        self.gill_weight = self.rules_cfg["vision_weights"]["gill_weight"]
        self.grade_thresholds = self.rules_cfg["vision_weights"]["grade_thresholds"]

    def compute_vision_freshness(
        self,
        eye_score: float,
        gill_score: float,
        eye_bbox: Optional[List[float]] = None,
        gill_bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Computes weighted freshness score and vision grade.
        Formula: (0.45 * eye_score + 0.55 * gill_score) * 100
        """
        eye_score = max(0.0, min(1.0, float(eye_score)))
        gill_score = max(0.0, min(1.0, float(gill_score)))

        combined = round((self.eye_weight * eye_score + self.gill_weight * gill_score) * 100.0, 1)

        if combined >= self.grade_thresholds["SAFE"]:
            grade = "SAFE"
        elif combined >= self.grade_thresholds["CAUTION"]:
            grade = "CAUTION"
        else:
            grade = "AVOID"

        return {
            "eye_bbox": eye_bbox if eye_bbox is not None else [0.0, 0.0, 0.0, 0.0],
            "gill_bbox": gill_bbox if gill_bbox is not None else [0.0, 0.0, 0.0, 0.0],
            "eye_score": round(eye_score, 2),
            "gill_score": round(gill_score, 2),
            "combined_freshness_score": combined,
            "grade": grade
        }

    def evaluate_verdict(
        self,
        species_label: str,
        species_confidence: float,
        vision_freshness: Dict[str, Any],
        sensory_input: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluates species prediction, vision freshness, and sensory input.
        Returns: (species_prediction_dict, final_verdict_dict)
        """
        # 1. Resolve Species Metadata
        meta = self.labels_cfg["species_metadata"].get(
            species_label,
            self.labels_cfg["species_metadata"]["unknown"]
        )
        display_name = meta["display_name"]
        intrinsic_risk = meta["intrinsic_risk"]

        species_pred = {
            "label": species_label,
            "display_name": display_name,
            "confidence": round(float(species_confidence), 2),
            "intrinsic_risk": intrinsic_risk
        }

        # 2. Extract Key Signals
        combined_score = vision_freshness["combined_freshness_score"]
        eye_score = vision_freshness["eye_score"]
        gill_score = vision_freshness["gill_score"]

        smell = sensory_input.get("smell", "fresh_sea")
        texture = sensory_input.get("texture", "bounces_back")
        storage = sensory_input.get("storage", "packed_in_ice")
        is_skipped = sensory_input.get("skipped", False)

        reasons: List[str] = []
        status: str = "GREEN"
        recommendation: str = ""

        # Priority 1: Check Illegal Chemical Adulteration (Formalin Flag)
        if not is_skipped and smell == "odorless" and texture == "unnaturally_stiff":
            status = "RED"
            reasons.append("Suspected formalin adulteration: unnaturally stiff flesh paired with lack of natural fish scent.")
            reasons.append("Chemical preservatives mask visual decay but do not make fish safe.")
            recommendation = "Do not purchase or consume. Report suspected chemical adulteration."
            return species_pred, {
                "status": status,
                "reasons": reasons,
                "recommendation": recommendation
            }

        # Priority 2: Critical Spoilage Signs in Sensory Check
        if not is_skipped and smell == "sour_ammonia":
            status = "RED"
            reasons.append("Strong sour or ammonia odor indicates advanced microbial decomposition.")
            recommendation = "Do not buy. High risk of acute bacterial food poisoning."
            return species_pred, {
                "status": status,
                "reasons": reasons,
                "recommendation": recommendation
            }

        if not is_skipped and texture == "mushy":
            status = "RED"
            reasons.append("Mushy flesh: severe muscle proteolysis and bacterial degradation.")
            recommendation = "Do not buy. Fish tissue is broken down and unsafe."
            return species_pred, {
                "status": status,
                "reasons": reasons,
                "recommendation": recommendation
            }

        # Priority 3: Histamine Temperature Abuse Rules
        if intrinsic_risk == "HIGH_HISTAMINE" and not is_skipped and storage == "no_ice_room_temp":
            if combined_score >= 70.0:
                status = "YELLOW"
                reasons.append(f"{display_name} is a high-histamine species.")
                reasons.append("Kept without ice: bacteria rapidly produce heat-stable histamine.")
                if gill_score < 0.75:
                    reasons.append("Gills show early color dulling.")
                recommendation = "Cook immediately if purchased, or choose a fish packed in ice."
            else:
                status = "RED"
                reasons.append(f"{display_name} is a high-histamine species stored without ice.")
                reasons.append("Visual freshness is degraded; high probability of toxic histamine build-up.")
                reasons.append("Cooking or frying does NOT destroy histamine.")
                recommendation = "Do not buy. Scombroid poisoning risk requires strict prevention."

            return species_pred, {
                "status": status,
                "reasons": reasons,
                "recommendation": recommendation
            }

        # Priority 4: Vision Freshness Score Thresholds
        if combined_score >= 75.0:
            status = "GREEN"
            reasons.append("Clear cornea and healthy eye convexity.")
            reasons.append("Gills show vibrant, natural reddish biological coloration.")
            recommendation = "Safe to buy. Keep chilled on ice, refrigerate within 2 hours, and cook within 24 hours."

            # Subtle ciguatera awareness note if applicable
            if intrinsic_risk == "CIGUATERA_WATCH":
                reasons.append(f"{display_name} is a reef predator. For maximum safety, avoid consuming head or viscera.")

        elif combined_score >= 50.0:
            status = "YELLOW"
            if eye_score < 0.65:
                reasons.append("Eye exhibits noticeable cloudiness or sunken pupil.")
            if gill_score < 0.65:
                reasons.append("Gills show early brownish discoloration or fading.")
            if not reasons:
                reasons.append("Moderate biological freshness decline detected.")
            recommendation = "Exercise caution. Check gills closely and cook thoroughly today if purchased."

        else:
            status = "RED"
            reasons.append("Sunken or opaque eye indicates advanced post-mortem decay.")
            reasons.append("Gills show severe discoloration or graying.")
            recommendation = "Do not purchase or consume. Fish is spoiled."

        # Add sensory skipped disclaimer if user bypassed questions
        if is_skipped and status == "GREEN":
            reasons.append("Sensory check skipped: invisible room-temperature or chemical hazards could not be assessed.")

        return species_pred, {
            "status": status,
            "reasons": reasons,
            "recommendation": recommendation
        }
