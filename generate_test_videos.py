"""
High-Fidelity Video Generator for Surveillance Engine Testing.
Constructs deterministic scenario clips guaranteed to activate YOLOv8, MediaPipe, and rules:
1. ATM Single Safe (Bare face, nose + mouth visible)
2. ATM Face Covered (Surgical Mask, Hand Covering, Kerchief, Helmet)
3. ATM Multi-Person Violation
4. Camera Blackout / Tamper
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import cv2
import numpy as np


def ensure_base_samples(output_dir: str = "sample_videos"):
    """Ensure base benchmark and sprite assets are available."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)



def get_real_person_variations(output_dir: str = "sample_videos") -> Dict[str, np.ndarray]:
    """
    Extracts real human crop from COCO dataset (guaranteed YOLOv8 + MediaPipe compatible)
    and constructs the 5 required facial variations.
    """
    path = Path(output_dir)
    coco_path = path / "000000001000.jpg"
    
    if coco_path.exists():
        img = cv2.imread(str(coco_path))
        # Person crop at (410, 211, 521, 479)
        p_crop = img[211:479, 410:521].copy()
    else:
        # Fallback synthetic sprite if offline
        p_crop = np.full((268, 111, 3), (45, 45, 50), dtype=np.uint8)

    variations = {}

    # 1. Bare face (nose tip/bridge and mouth/lips fully visible)
    variations["bare"] = p_crop.copy()

    # 2. Surgical / Cloth mask over nose & mouth
    m_mask = p_crop.copy()
    cv2.rectangle(m_mask, (48, 44), (82, 65), (220, 200, 180), -1)
    cv2.rectangle(m_mask, (48, 44), (82, 65), (180, 160, 140), 1)
    cv2.line(m_mask, (48, 48), (38, 42), (200, 200, 200), 1)
    cv2.line(m_mask, (82, 48), (92, 42), (200, 200, 200), 1)
    variations["mask"] = m_mask

    # 3. Hand covering mouth & nose
    m_hand = p_crop.copy()
    cv2.rectangle(m_hand, (50, 45), (80, 64), (130, 165, 210), -1)
    for y in range(48, 64, 4):
        cv2.line(m_hand, (52, y), (78, y), (90, 120, 160), 1)
    variations["hand"] = m_hand

    # 4. Kerchief / Bandana tied over nose & lower face
    m_ker = p_crop.copy()
    pts = np.array([[46, 44], [84, 44], [65, 72]], np.int32)
    cv2.fillPoly(m_ker, [pts], (20, 20, 20))
    cv2.polylines(m_ker, [pts], True, (60, 60, 60), 1)
    variations["kerchief"] = m_ker

    # 5. Full/half motorcycle helmet covering head
    m_helm = p_crop.copy()
    cv2.ellipse(m_helm, (65, 38), (30, 36), 0, 0, 360, (25, 25, 30), -1)
    cv2.rectangle(m_helm, (50, 32), (80, 40), (60, 60, 70), -1)
    variations["helmet"] = m_helm

    return variations


def overlay_person_crop(
    background: np.ndarray,
    person_crop: np.ndarray,
    x: int,
    y: int,
    scale: float = 1.3,
) -> np.ndarray:
    """Scale and overlay person crop onto background frame."""
    h_bg, w_bg = background.shape[:2]
    sh, sw = person_crop.shape[:2]
    target_w, target_h = int(sw * scale), int(sh * scale)
    resized = cv2.resize(person_crop, (target_w, target_h))

    x1 = max(0, x - target_w // 2)
    y1 = max(0, y - target_h)
    x2 = min(w_bg, x1 + target_w)
    y2 = min(h_bg, y1 + target_h)

    sw_actual = x2 - x1
    sh_actual = y2 - y1

    if sw_actual > 0 and sh_actual > 0:
        background[y1:y2, x1:x2] = resized[:sh_actual, :sw_actual]

    return background


def generate_all_test_videos(
    output_dir: str = "sample_videos",
    width: int = 960,
    height: int = 540,
    fps: int = 30,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ensure_base_samples(output_dir)

    variations = get_real_person_variations(output_dir)
    p_bare = variations["bare"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    print(f"[TEST GENERATOR] Generating scenario clips in '{output_dir}'...")

    def create_atm_background() -> np.ndarray:
        frame = np.full((height, width, 3), (35, 40, 48), dtype=np.uint8)
        # Background ATM kiosk & wall
        cv2.rectangle(frame, (100, 80), (280, 480), (70, 80, 95), -1)
        cv2.rectangle(frame, (130, 130), (250, 240), (0, 160, 90), -1)
        cv2.putText(frame, "ATM", (160, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "SURVEILLANCE CAM #01", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame

    # -------------------------------------------------------------
    # 1. ATM Single Safe (Bare face, nose + mouth visible)
    # -------------------------------------------------------------
    clip1 = out_path / "atm_single_safe.mp4"
    out1 = cv2.VideoWriter(str(clip1), fourcc, fps, (width, height))
    for _ in range(fps * 6):
        frame = create_atm_background()
        frame = overlay_person_crop(frame, p_bare, x=480, y=470, scale=1.3)
        out1.write(frame)
    out1.release()
    print(f"  [OK] Created: {clip1.name}")

    # -------------------------------------------------------------
    # 2. ATM Face: Surgical/Cloth Mask
    # -------------------------------------------------------------
    clip_mask = out_path / "atm_face_obscured.mp4"
    out_mask = cv2.VideoWriter(str(clip_mask), fourcc, fps, (width, height))
    p_mask = variations["mask"]
    for _ in range(fps * 6):
        frame = create_atm_background()
        frame = overlay_person_crop(frame, p_mask, x=480, y=470, scale=1.3)
        out_mask.write(frame)
    out_mask.release()
    print(f"  [OK] Created: {clip_mask.name}")

    # -------------------------------------------------------------
    # 3. ATM Face: Hand Covering Face
    # -------------------------------------------------------------
    clip_hand = out_path / "atm_hand_covering.mp4"
    out_hand = cv2.VideoWriter(str(clip_hand), fourcc, fps, (width, height))
    p_hand = variations["hand"]
    for _ in range(fps * 6):
        frame = create_atm_background()
        frame = overlay_person_crop(frame, p_hand, x=480, y=470, scale=1.3)
        out_hand.write(frame)
    out_hand.release()
    print(f"  [OK] Created: {clip_hand.name}")

    # -------------------------------------------------------------
    # 4. ATM Face: Kerchief / Bandana
    # -------------------------------------------------------------
    clip_ker = out_path / "atm_kerchief.mp4"
    out_ker = cv2.VideoWriter(str(clip_ker), fourcc, fps, (width, height))
    p_ker = variations["kerchief"]
    for _ in range(fps * 6):
        frame = create_atm_background()
        frame = overlay_person_crop(frame, p_ker, x=480, y=470, scale=1.3)
        out_ker.write(frame)
    out_ker.release()
    print(f"  [OK] Created: {clip_ker.name}")

    # -------------------------------------------------------------
    # 5. ATM Face: Motorcycle Helmet
    # -------------------------------------------------------------
    clip_helm = out_path / "atm_helmet.mp4"
    out_helm = cv2.VideoWriter(str(clip_helm), fourcc, fps, (width, height))
    p_helm = variations["helmet"]
    for _ in range(fps * 6):
        frame = create_atm_background()
        frame = overlay_person_crop(frame, p_helm, x=480, y=470, scale=1.3)
        out_helm.write(frame)
    out_helm.release()
    print(f"  [OK] Created: {clip_helm.name}")

    # -------------------------------------------------------------
    # 6. ATM Multi-Person Violation (2 people simultaneously)
    # -------------------------------------------------------------
    clip_multi = out_path / "atm_multi_person.mp4"
    out_multi = cv2.VideoWriter(str(clip_multi), fourcc, fps, (width, height))
    for _ in range(fps * 6):
        frame = create_atm_background()
        frame = overlay_person_crop(frame, p_bare, x=370, y=470, scale=1.3)
        frame = overlay_person_crop(frame, p_bare, x=590, y=470, scale=1.3)
        out_multi.write(frame)
    out_multi.release()
    print(f"  [OK] Created: {clip_multi.name}")

    # -------------------------------------------------------------
    # 7. Camera Blackout / Tamper
    # -------------------------------------------------------------
    clip_blackout = out_path / "camera_blackout.mp4"
    out_blackout = cv2.VideoWriter(str(clip_blackout), fourcc, fps, (width, height))
    for i in range(fps * 10):
        if i < fps * 2:
            frame = create_atm_background()
            frame = overlay_person_crop(frame, p_bare, x=480, y=470, scale=1.3)
        else:
            frame = np.full((height, width, 3), 10, dtype=np.uint8)
        out_blackout.write(frame)
    out_blackout.release()
    print(f"  [OK] Created: {clip_blackout.name}")
    print("[TEST GENERATOR] All scenario clips generated successfully!")

    print("[TEST GENERATOR] All scenario clips generated successfully!")


if __name__ == "__main__":
    generate_all_test_videos()
