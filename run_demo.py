"""
FishGuard End-to-End Pipeline Demo
Runs sample scans on traditional market fish images and displays the contract JSON.
"""

import sys
import json
from pathlib import Path

# Add project root to path
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir.parent))

from fish_guard_model.pipeline.inference_pipeline import FishGuardInferencePipeline


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_section("🐟 FishGuard Offline Edge ML Inspection Pipeline Demo 🐟")
    pipeline = FishGuardInferencePipeline()

    samples_dir = base_dir / "test_samples"

    # DEMO 1: Fresh Tongkol on Ice
    print_section("DEMO 1: Fresh Tongkol on Crushed Ice (Expected: GREEN / SAFE)")
    res_fresh = pipeline.inspect_fish(
        whole_fish_image=samples_dir / "tongkol_fresh_iced.jpg",
        close_up_image=samples_dir / "tongkol_fresh_iced.jpg",
        sensory_input={
            "smell": "fresh_sea",
            "texture": "bounces_back",
            "storage": "packed_in_ice",
            "skipped": False
        }
    )
    print(json.dumps(res_fresh, indent=2))

    # DEMO 2: High-Histamine Tongkol Stored Without Ice at Room Temp
    print_section("DEMO 2: Tongkol Stored Off-Ice at Room Temperature (Expected: YELLOW / CAUTION)")
    res_off_ice = pipeline.inspect_fish(
        whole_fish_image=samples_dir / "tongkol_fresh_iced.jpg",
        close_up_image=samples_dir / "tongkol_fresh_iced.jpg",
        sensory_input={
            "smell": "fresh_sea",
            "texture": "bounces_back",
            "storage": "no_ice_room_temp",
            "skipped": False
        }
    )
    print(json.dumps(res_off_ice, indent=2))

    # DEMO 3: Suspected Formalin Adulteration
    print_section("DEMO 3: Formalin Adulteration Flag (Expected: RED / AVOID)")
    res_formalin = pipeline.inspect_fish(
        whole_fish_image=samples_dir / "kembung_formalin_flag.jpg",
        close_up_image=samples_dir / "kembung_formalin_flag.jpg",
        sensory_input={
            "smell": "odorless",
            "texture": "unnaturally_stiff",
            "storage": "no_ice_room_temp",
            "skipped": False
        }
    )
    print(json.dumps(res_formalin, indent=2))


if __name__ == "__main__":
    main()
