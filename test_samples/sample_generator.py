"""
Synthetic Test Image Generator for FishGuard
Generates realistic market fish test images with calibrated eye and gill biological markers
to test and validate the end-to-end CV and risk engine pipeline.
"""

from pathlib import Path
import numpy as np
import cv2


def create_gradient_ellipse(img, center, axes, angle, color_start, color_end):
    """Draws a smooth gradient ellipse representing fish body or organs."""
    overlay = img.copy()
    cv2.ellipse(overlay, center, axes, angle, 0, 360, color_start, -1)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def generate_market_background(w: int, h: int, with_ice: bool = True, tarp_color: str = "blue") -> np.ndarray:
    """Generates market stall backdrop with optional crushed ice and tarp color cast."""
    if tarp_color == "blue":
        base_color = np.array([120, 90, 60], dtype=np.uint8)  # BGR blue tarp
    else:
        base_color = np.array([40, 80, 110], dtype=np.uint8)   # BGR orange tarp

    img = np.full((h, w, 3), base_color, dtype=np.uint8)

    # Add realistic tarp texture and lighting gradient
    gradient = np.tile(np.linspace(0.8, 1.2, w, dtype=np.float32), (h, 1))
    img = np.clip(img * gradient[:, :, None], 0, 255).astype(np.uint8)

    if with_ice:
        # Add crushed ice crystals (white/cyan speckled noise)
        ice_mask = np.random.rand(h, w) > 0.4
        ice_tint = np.random.randint(220, 255, (h, w, 3), dtype=np.uint8)
        img[ice_mask] = (img[ice_mask] * 0.3 + ice_tint[ice_mask] * 0.7).astype(np.uint8)

    return img


def render_tongkol_fresh(w: int = 640, h: int = 480) -> np.ndarray:
    """
    Renders a fresh Tongkol (Bullet Tuna) packed in crushed ice.
    - Metallic dark blue dorsal area with tuna finlets
    - Bright silvery belly
    - Eye: bulging, crystal clear cornea, black pupil
    - Gill: vibrant bright crimson red (high a* in Lab)
    """
    img = generate_market_background(w, h, with_ice=True, tarp_color="blue")

    # 1. Fish Body (Torpedo Tuna Shape)
    center = (int(w * 0.52), int(h * 0.52))
    axes = (int(w * 0.38), int(h * 0.18))
    cv2.ellipse(img, center, axes, 0, 0, 360, (190, 185, 175), -1)  # Silver belly

    # Dorsal dark blue tuna back
    dorsal_pts = np.array([
        [int(w * 0.16), int(h * 0.52)],
        [int(w * 0.35), int(h * 0.35)],
        [int(w * 0.70), int(h * 0.38)],
        [int(w * 0.90), int(h * 0.52)]
    ], np.int32)
    cv2.fillPoly(img, [dorsal_pts], (110, 45, 15))  # Deep oceanic navy blue

    # Forked tail
    tail_pts = np.array([
        [int(w * 0.88), int(h * 0.52)],
        [int(w * 0.96), int(h * 0.38)],
        [int(w * 0.93), int(h * 0.52)],
        [int(w * 0.96), int(h * 0.66)]
    ], np.int32)
    cv2.fillPoly(img, [tail_pts], (120, 110, 100))

    # 2. Eye Region (Upper front head: ~[0.42, 0.31, 0.12, 0.12])
    # Clear cornea, jet black pupil, sharp reflective glint
    eye_c = (int(w * 0.28), int(h * 0.46))
    eye_r = int(w * 0.035)
    cv2.circle(img, eye_c, eye_r + 2, (150, 150, 150), -1)      # Eye ring
    cv2.circle(img, eye_c, eye_r, (235, 240, 245), -1)          # Clear cornea
    cv2.circle(img, eye_c, int(eye_r * 0.65), (20, 20, 20), -1) # Dark pupil
    cv2.circle(img, (eye_c[0] - 3, eye_c[1] - 3), 3, (255, 255, 255), -1) # Fresh glint

    # 3. Gill Region (Operculum slit: ~[0.55, 0.40, 0.18, 0.22] in close-up)
    # Vibrant crimson red fresh filaments
    gill_c = (int(w * 0.38), int(h * 0.53))
    cv2.ellipse(img, gill_c, (int(w * 0.04), int(h * 0.09)), 25, 0, 360, (25, 30, 200), -1) # Crimson Red (BGR)
    cv2.ellipse(img, gill_c, (int(w * 0.04), int(h * 0.09)), 25, 0, 360, (15, 20, 150), 3)

    return img


def render_tongkol_spoiled(w: int = 640, h: int = 480) -> np.ndarray:
    """
    Renders spoiled Tongkol stored at room temperature without ice.
    - Dull, yellowish-brown discolored body
    - Eye: sunken, cloudy, milky opaque cornea (high b* yellowing)
    - Gill: dull brownish-gray discolored (low a* redness)
    """
    img = generate_market_background(w, h, with_ice=False, tarp_color="blue")

    # Fish Body (Dull, desiccated tuna shape)
    center = (int(w * 0.52), int(h * 0.52))
    axes = (int(w * 0.38), int(h * 0.18))
    cv2.ellipse(img, center, axes, 0, 0, 360, (140, 145, 140), -1)  # Dull faded belly

    dorsal_pts = np.array([
        [int(w * 0.16), int(h * 0.52)],
        [int(w * 0.35), int(h * 0.35)],
        [int(w * 0.70), int(h * 0.38)],
        [int(w * 0.90), int(h * 0.52)]
    ], np.int32)
    cv2.fillPoly(img, [dorsal_pts], (70, 60, 50))  # Murky brownish dorsal

    # 2. Spoiled Eye: Sunken, milky cloudy, no glint
    eye_c = (int(w * 0.28), int(h * 0.46))
    eye_r = int(w * 0.030)  # Shrunken/sunken
    cv2.circle(img, eye_c, eye_r + 2, (90, 95, 95), -1)
    cv2.circle(img, eye_c, eye_r, (160, 180, 195), -1)         # Milky yellowed cornea
    cv2.circle(img, eye_c, int(eye_r * 0.65), (70, 75, 80), -1) # Faded pupil

    # 3. Spoiled Gill: Muddy brownish-gray
    gill_c = (int(w * 0.38), int(h * 0.53))
    cv2.ellipse(img, gill_c, (int(w * 0.04), int(h * 0.09)), 25, 0, 360, (70, 85, 105), -1) # Muddy brown/gray
    cv2.ellipse(img, gill_c, (int(w * 0.04), int(h * 0.09)), 25, 0, 360, (50, 60, 70), 2)

    return img


def render_kembung_formalin(w: int = 640, h: int = 480) -> np.ndarray:
    """
    Renders Kembung (Mackerel) with suspected formalin adulteration.
    - Unnaturally stiff, waxy, dry skin
    - Gills: pale, bleached whitish-pink
    - Eye: glassy but dry, shrunken margins
    """
    img = generate_market_background(w, h, with_ice=False, tarp_color="orange")

    # Mackerel slender body
    center = (int(w * 0.50), int(h * 0.50))
    axes = (int(w * 0.35), int(h * 0.14))
    cv2.ellipse(img, center, axes, 0, 0, 360, (200, 205, 205), -1)  # Unnaturally bleached

    # Waxy mackerel stripes
    for offset_x in range(int(w * 0.35), int(w * 0.65), 25):
        cv2.line(img, (offset_x, int(h * 0.40)), (offset_x + 10, int(h * 0.48)), (60, 100, 70), 3)

    # Formalin Eye: Glassy, hard, dry
    eye_c = (int(w * 0.26), int(h * 0.47))
    eye_r = int(w * 0.030)
    cv2.circle(img, eye_c, eye_r, (210, 215, 215), -1)
    cv2.circle(img, eye_c, int(eye_r * 0.5), (40, 40, 40), -1)

    # Formalin Gill: Bleached, pale whitish-pink (chemically fixed)
    gill_c = (int(w * 0.36), int(h * 0.52))
    cv2.ellipse(img, gill_c, (int(w * 0.035), int(h * 0.08)), 20, 0, 360, (150, 155, 185), -1)

    return img


def generate_all_samples(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fresh_img = render_tongkol_fresh()
    cv2.imwrite(str(output_dir / "tongkol_fresh_iced.jpg"), fresh_img)
    print(f"Generated {output_dir / 'tongkol_fresh_iced.jpg'}")

    spoiled_img = render_tongkol_spoiled()
    cv2.imwrite(str(output_dir / "tongkol_spoiled_no_ice.jpg"), spoiled_img)
    print(f"Generated {output_dir / 'tongkol_spoiled_no_ice.jpg'}")

    formalin_img = render_kembung_formalin()
    cv2.imwrite(str(output_dir / "kembung_formalin_flag.jpg"), formalin_img)
    print(f"Generated {output_dir / 'kembung_formalin_flag.jpg'}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    generate_all_samples(out)
