"""
ATM Security Rule Module (Isolated Plug-in).
Enforces ATM physical security guidelines:
1. Robust Face Detection & Mouth/Nose Landmark Covering Verification (Mask, Helmet, Kerchief, Hand).
2. Multi-Person Transaction Violation Detection.
3. Camera Blackout / Tamper / Occlusion Detection with Emergency Escalation.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import cv2
import numpy as np

from alert_system import AlertCondition, AlertSeverity
from config import ATMModuleConfig
from detection_model import TrackedObject

logger = logging.getLogger(__name__)

# Key facial landmark indices for mouth and nose verification:
# Nose tip: 1, Nose bridge: 4, Nose base / subnasale: 2
# Upper lip: 0 (outer top), 13 (inner top)
# Lower lip: 17 (outer bottom), 14 (inner bottom)
# Mouth corners: 61 (left), 291 (right)
TARGET_LANDMARK_INDICES: List[int] = [1, 4, 2, 0, 13, 14, 17, 61, 291]


def get_utc_now() -> datetime:
    """Return timezone-naive UTC timestamp for consistent SQLite compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class ATMStatus:
    """Consolidated state evaluation for ATM security overlay."""

    is_safe: bool = True
    operational: bool = True
    status_label: str = "ATM Idle — Ready"
    face_detected: bool = False
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in frame coordinates
    face_landmarks_visible: bool = False
    mouth_nose_confidence: float = 1.0       # Current frame confidence (0.0 to 1.0)
    visibility_score: float = 1.0            # Backward-compatible alias for confidence (0.0 to 1.0)
    rolling_visibility_score: float = 1.0    # Rolling average confidence (0.0 to 1.0)
    person_count: int = 0
    blackout_active: bool = False
    active_alerts: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class ATMRuleModule:
    """
    ATM Security Intelligence Module.
    Optimized for high-FPS, real-time performance on live webcam streams.
    """

    def __init__(self, config: Optional[ATMModuleConfig] = None):
        self.config = config or ATMModuleConfig()
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.blackout_start_time: Optional[datetime] = None
        self.simulated_dispatch_logged = False

        # Rolling Buffer Configuration
        self.rolling_window_size: int = int(getattr(self.config, "visibility_window_size", 30))
        self.covered_threshold: float = float(getattr(self.config, "covered_threshold", 0.35))
        self.safe_recovery_threshold: float = float(getattr(self.config, "safe_recovery_threshold", 0.55))

        # Buffers and state tracking per person track_id
        self.face_visibility_history: Dict[int, deque] = {}  # track_id -> deque of float confidences
        self.face_confidence_history = self.face_visibility_history  # Alias
        self.face_covered_state: Dict[int, bool] = {}        # track_id -> bool (True=covered)
        self.face_state_start_time: Dict[int, datetime] = {}

        # Initialize MediaPipe FaceLandmarker with multi-face support (num_faces=4)
        self.mediapipe_landmarker = None
        task_path = Path("models/face_landmarker.task")
        if task_path.exists():
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(task_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    min_face_detection_confidence=0.20,
                    min_face_presence_confidence=0.20,
                    min_tracking_confidence=0.20,
                    num_faces=4,
                )
                self.mediapipe_landmarker = vision.FaceLandmarker.create_from_options(options)
                logger.info("[ATM RULES] MediaPipe FaceLandmarker initialized successfully (multi-face).")
            except Exception as e:
                logger.warning(f"[ATM RULES] Could not initialize MediaPipe FaceLandmarker: {e}")

        # Initialize fast Haar Face Cascade for supplementary face/crowd verification
        self.face_cascade = None
        cascade_path = Path("models/haarcascade_frontalface_default.xml")
        if cascade_path.exists():
            try:
                self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
                logger.info("[ATM RULES] Haar Frontal Face Cascade loaded.")
            except Exception as e:
                logger.warning(f"[ATM RULES] Could not load face cascade: {e}")

        import threading
        self._lock = threading.Lock()
        self._next_face_person_id = 500

    def detect_all_faces(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect all faces and landmarks in full frame.
        Combines MediaPipe multi-face landmarker and Haar cascade.
        
        Returns:
            List of dicts: [{"bbox": (fx1, fy1, fx2, fy2), "vis_score": float, "landmarks": list, "lms_raw": list}, ...]
        """
        h, w = frame_bgr.shape[:2]
        detected_faces: List[Dict[str, Any]] = []
        found_mp = False

        if self.mediapipe_landmarker is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                with self._lock:
                    res = self.mediapipe_landmarker.detect(mp_img)

                if res.face_landmarks and len(res.face_landmarks) > 0:
                    found_mp = True
                    for lms in res.face_landmarks:
                        xs = [lm.x * w for lm in lms]
                        ys = [lm.y * h for lm in lms]
                        fx1 = int(np.clip(min(xs) - (max(xs) - min(xs)) * 0.10, 0, w))
                        fy1 = int(np.clip(min(ys) - (max(ys) - min(ys)) * 0.10, 0, h))
                        fx2 = int(np.clip(max(xs) + (max(xs) - min(xs)) * 0.10, 0, w))
                        fy2 = int(np.clip(max(ys) + (max(ys) - min(ys)) * 0.10, 0, h))

                        # Landmark extraction
                        landmark_pts = []
                        for idx in TARGET_LANDMARK_INDICES:
                            px = int(np.clip(lms[idx].x * w, 0, w - 1))
                            py = int(np.clip(lms[idx].y * h, 0, h - 1))
                            landmark_pts.append((px, py))

                        # Evaluate skin chrominance
                        vis_count = 0
                        for px, py in landmark_pts:
                            patch = frame_bgr[max(0, py - 3):min(h, py + 4), max(0, px - 3):min(w, px + 4)]
                            if patch.size > 0:
                                ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
                                cr = float(np.mean(ycrcb[:, :, 1]))
                                cb = float(np.mean(ycrcb[:, :, 2]))
                                y_val = float(np.mean(ycrcb[:, :, 0]))
                                if (105 <= cr <= 210) and (60 <= cb <= 165) and (18 <= y_val <= 252) and ((cr - cb) >= 2):
                                    vis_count += 1
                        vis_score = vis_count / float(len(TARGET_LANDMARK_INDICES)) if TARGET_LANDMARK_INDICES else 1.0

                        detected_faces.append({
                            "bbox": (fx1, fy1, fx2, fy2),
                            "vis_score": vis_score,
                            "landmarks": landmark_pts,
                            "source": "mediapipe",
                        })
            except Exception as e:
                logger.debug(f"[ATM RULES] MediaPipe detect_all_faces error: {e}")

        # Supplementary Haar cascade face check for additional background / bystander faces
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                haar_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(35, 35)
                )
                for (hx, hy, hw, hh) in haar_faces:
                    h_bbox = (int(hx), int(hy), int(hx + hw), int(hy + hh))
                    # Check if already covered by an existing MediaPipe detection
                    already_found = False
                    for df in detected_faces:
                        db = df["bbox"]
                        # Overlap check
                        ix1 = max(h_bbox[0], db[0])
                        iy1 = max(h_bbox[1], db[1])
                        ix2 = min(h_bbox[2], db[2])
                        iy2 = min(h_bbox[3], db[3])
                        if ix1 < ix2 and iy1 < iy2:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            h_area = hw * hh
                            if inter_area / float(h_area) > 0.35:
                                already_found = True
                                break
                    if not already_found:
                        detected_faces.append({
                            "bbox": h_bbox,
                            "vis_score": 1.0,
                            "landmarks": [],
                            "source": "haar",
                        })
            except Exception as e:
                logger.debug(f"[ATM RULES] Haar detect error: {e}")

        return detected_faces

    def fuse_persons_and_faces(
        self,
        frame_bgr: np.ndarray,
        yolo_persons: List[TrackedObject],
    ) -> Tuple[List[TrackedObject], List[Dict[str, Any]]]:
        """
        Fuses YOLO body detections with Multi-Face detections so all visible people
        (including seated persons, bystanders leaning into frame, or upper-body webcam angles)
        are reliably detected and tracked as distinct TrackedObjects.
        """
        h, w = frame_bgr.shape[:2]
        faces = self.detect_all_faces(frame_bgr)
        fused_list: List[TrackedObject] = list(yolo_persons)

        # For each face, see if an existing YOLO person encompasses or overlaps it
        for i, face in enumerate(faces):
            fb = face["bbox"]
            fcx = (fb[0] + fb[2]) / 2
            fcy = (fb[1] + fb[3]) / 2

            matched_yolo = False
            for yp in yolo_persons:
                yb = yp.bbox
                # Containment or heavy overlap
                if yb[0] <= fcx <= yb[2] and yb[1] <= fcy <= yb[3]:
                    matched_yolo = True
                    break
                # Bounding box IoU overlap
                ix1 = max(fb[0], yb[0])
                iy1 = max(fb[1], yb[1])
                ix2 = min(fb[2], yb[2])
                iy2 = min(fb[3], yb[3])
                if ix1 < ix2 and iy1 < iy2:
                    inter_a = (ix2 - ix1) * (iy2 - iy1)
                    face_a = (fb[2] - fb[0]) * (fb[3] - fb[1])
                    if inter_a / float(max(1, face_a)) > 0.40:
                        matched_yolo = True
                        break

            if not matched_yolo:
                # Synthesize a person bounding box from the face region (expanding downwards for shoulders/torso)
                fw = fb[2] - fb[0]
                fh = fb[3] - fb[1]
                px1 = max(0, int(fb[0] - fw * 0.35))
                py1 = max(0, int(fb[1] - fh * 0.15))
                px2 = min(w, int(fb[2] + fw * 0.35))
                py2 = min(h, int(fb[3] + fh * 2.4))

                new_id = self._next_face_person_id
                self._next_face_person_id += 1
                if self._next_face_person_id > 999:
                    self._next_face_person_id = 500

                fused_person = TrackedObject(
                    track_id=new_id,
                    class_id=0,
                    class_name="person",
                    confidence=0.88,
                    bbox=(px1, py1, px2, py2),
                )
                fused_list.append(fused_person)

        # Deduplicate heavily overlapping fused objects (IoU > 0.65)
        unique_persons: List[TrackedObject] = []
        for p in fused_list:
            is_dup = False
            for up in unique_persons:
                ix1 = max(p.bbox[0], up.bbox[0])
                iy1 = max(p.bbox[1], up.bbox[1])
                ix2 = min(p.bbox[2], up.bbox[2])
                iy2 = min(p.bbox[3], up.bbox[3])
                if ix1 < ix2 and iy1 < iy2:
                    inter_a = (ix2 - ix1) * (iy2 - iy1)
                    p_a = max(1, p.width * p.height)
                    if inter_a / float(p_a) > 0.65:
                        is_dup = True
                        break
            if not is_dup:
                unique_persons.append(p)

        return unique_persons, faces

    def _check_face_visibility(
        self,
        frame_bgr: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, float, List[Tuple[int, int]]]:
        """Backward-compatible helper method for tests and external callers."""
        face_det, _, score, lms, _ = self._detect_face_and_landmarks(frame_bgr, person_bbox)
        return face_det, score, lms

    def _check_blackout_tamper(
        self,
        frame: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Fast evaluation of frame variance, luminance, and Laplacian edge variance.
        Optimized by running on downscaled 160x90 frame for sub-millisecond execution.
        """
        now = get_utc_now()
        small_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian_var = float(cv2.Laplacian(blurred, cv2.CV_64F).var())
        mean_val = float(np.mean(gray))

        is_currently_blacked_out = (
            laplacian_var < self.config.blackout_variance_threshold
            or mean_val < 8.0
            or mean_val > 248.0
        )

        if not is_currently_blacked_out and self.prev_frame_gray is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame_gray)
            diff_score = float(np.mean(frame_diff))
            if diff_score < 0.2 and laplacian_var < (self.config.blackout_variance_threshold * 1.5):
                is_currently_blacked_out = True

        self.prev_frame_gray = gray.copy()

        if is_currently_blacked_out:
            if self.blackout_start_time is None:
                self.blackout_start_time = now
            duration = (now - self.blackout_start_time).total_seconds()
            if duration >= self.config.blackout_duration_sec:
                return True, laplacian_var
        else:
            self.blackout_start_time = None
            self.simulated_dispatch_logged = False

        return False, laplacian_var

    def evaluate_frame(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedObject],
        zone_name: str = "ATM Area",
        precomputed_face: Optional[Tuple[bool, float, List[Tuple[int, int]]]] = None,
    ) -> Tuple[ATMStatus, List[AlertCondition], List[Tuple[int, int]]]:
        """
        Evaluate complete ATM safety rules on current frame and tracked persons.

        Returns:
            (ATMStatus, List[AlertCondition], List[LandmarkPoints])
        """
        now = get_utc_now()
        conditions: List[AlertCondition] = []
        all_landmarks: List[Tuple[int, int]] = []
        person_count = len(tracked_persons)

        status = ATMStatus(
            is_safe=True,
            operational=True,
            status_label="ATM Idle — Ready",
            person_count=person_count,
            face_detected=False,
            face_bbox=None,
            face_landmarks_visible=False,
            mouth_nose_confidence=1.0,
            visibility_score=1.0,
            rolling_visibility_score=1.0,
            active_alerts=[],
            debug_info={
                "person_found": person_count > 0,
                "person_count": person_count,
                "face_found": False,
                "vis_score_pct": 0,
                "rolling_vis_pct": 0,
            }
        )

        # 1. Camera Blackout / Tamper Rule
        is_tampered, var_val = self._check_blackout_tamper(frame)
        if is_tampered:
            status.blackout_active = True
            status.is_safe = False
            status.operational = False
            status.status_label = "CRITICAL: CAMERA BLACKOUT / TAMPER DETECTED"

            if not self.simulated_dispatch_logged and self.config.simulated_dispatch_enabled:
                logger.critical(
                    f"[SIMULATION] *** HIGH PRIORITY ALARM *** "
                    f"Camera tamper confirmed at ATM terminal ({var_val:.1f} var for >{self.config.blackout_duration_sec}s). "
                    f"Dispatching patrol unit to {self.config.authority_contact}."
                )
                self.simulated_dispatch_logged = True

            conditions.append(
                AlertCondition(
                    condition_key="atm_camera_blackout",
                    alert_type="CAMERA_BLACKOUT",
                    severity=AlertSeverity.CRITICAL,
                    module_name="atm_rules",
                    message=f"Camera blackout / physical tamper detected (variance: {var_val:.1f}). Emergency escalation initiated.",
                    is_active=True,
                    zone_name=zone_name,
                    sound_siren=self.config.sound_siren_on_tamper,
                    voice_announcement="Warning. Camera tamper detected. Security dispatched.",
                    trigger_phone_call=True,
                    details={"variance": var_val, "dispatch_contact": self.config.authority_contact},
                )
            )
            return status, conditions, all_landmarks
        else:
            conditions.append(
                AlertCondition(
                    condition_key="atm_camera_blackout",
                    alert_type="CAMERA_BLACKOUT",
                    severity=AlertSeverity.CRITICAL,
                    module_name="atm_rules",
                    message="Camera clear",
                    is_active=False,
                    zone_name=zone_name,
                )
            )

        # 2. Multi-Person Rule (> 1 person present in ATM area)
        multi_person_active = person_count > 1
        conditions.append(
            AlertCondition(
                condition_key="atm_multi_person",
                alert_type="MULTI_PERSON",
                severity=AlertSeverity.MEDIUM,
                module_name="atm_rules",
                message="Multiple people detected — please use the ATM one person at a time",
                is_active=multi_person_active,
                tracked_ids=[p.track_id for p in tracked_persons],
                zone_name=zone_name,
                voice_announcement="Multiple people detected. Please use the ATM one person at a time.",
                details={"person_count": person_count},
            )
        )

        if multi_person_active:
            status.is_safe = False
            status.operational = False
            status.status_label = "Multiple people detected — please use the ATM one person at a time"
            status.debug_info["status"] = "MULTIPLE_PEOPLE"
            return status, conditions, all_landmarks

        # Clean up stale track IDs not currently present
        current_tids = {p.track_id for p in tracked_persons}
        for tid in list(self.face_visibility_history.keys()):
            if tid not in current_tids:
                self.face_visibility_history.pop(tid, None)
                self.face_covered_state.pop(tid, None)
                self.face_state_start_time.pop(tid, None)

        # 3. Single Person Face Landmark Visibility Rule
        if person_count == 1:
            person = tracked_persons[0]
            pid = person.track_id

            if precomputed_face is not None:
                face_detected, frame_vis_score, landmarks = precomputed_face
                face_bbox = (int(person.bbox[0]), int(person.bbox[1]), int(person.bbox[2]), int(person.bbox[3])) if face_detected else None
                metrics = {}
            elif self._check_face_visibility.__code__ != ATMRuleModule._check_face_visibility.__code__:
                # Monkeypatched in unit tests
                face_detected, frame_vis_score, landmarks = self._check_face_visibility(frame, person.bbox)
                face_bbox = (int(person.bbox[0]), int(person.bbox[1]), int(person.bbox[2]), int(person.bbox[3])) if face_detected else None
                metrics = {}
            else:
                face_detected, face_bbox, frame_vis_score, landmarks, metrics = self._detect_face_and_landmarks(
                    frame, person.bbox
                )

            all_landmarks.extend(landmarks)
            status.face_detected = face_detected
            status.face_bbox = face_bbox

            # Handle backward compatibility if mock returned boolean
            if isinstance(frame_vis_score, bool):
                frame_vis_score = 1.0 if frame_vis_score else 0.0

            frame_vis_score = float(frame_vis_score)
            status.visibility_score = frame_vis_score
            status.mouth_nose_confidence = frame_vis_score

            # Initialize history buffer and state if new track
            if pid not in self.face_visibility_history:
                self.face_visibility_history[pid] = deque(maxlen=self.rolling_window_size)
                self.face_covered_state[pid] = False
                self.face_state_start_time[pid] = now

            buf = self.face_visibility_history[pid]
            buf.append(frame_vis_score)

            # Calculate rolling window visibility score
            n = len(buf)
            rolling_avg_vis = float(sum(buf) / float(n)) if n > 0 else 0.0
            status.rolling_visibility_score = rolling_avg_vis

            current_covered_state = self.face_covered_state.get(pid, False)

            # Two-Signal Hysteresis State Machine:
            # 1. Transition SAFE -> FACE_COVERED:
            min_eval_frames = min(5, self.rolling_window_size)
            if not current_covered_state:
                if n >= min_eval_frames and rolling_avg_vis < self.covered_threshold:
                    self.face_covered_state[pid] = True
                    self.face_state_start_time[pid] = now
                    logger.warning(
                        f"[ATM RULES] Person #{pid} STATE TRANSITION: SAFE -> FACE_COVERED "
                        f"(rolling_vis={rolling_avg_vis:.1%}, frame_vis={frame_vis_score:.1%}, window={n})"
                    )

            # 2. Transition FACE_COVERED -> SAFE:
            else:
                if rolling_avg_vis > self.safe_recovery_threshold:
                    self.face_covered_state[pid] = False
                    self.face_state_start_time[pid] = now
                    logger.info(
                        f"[ATM RULES] Person #{pid} STATE TRANSITION: FACE_COVERED -> SAFE "
                        f"(rolling_vis={rolling_avg_vis:.1%}, frame_vis={frame_vis_score:.1%}, window={n})"
                    )

            face_covered = self.face_covered_state.get(pid, False)
            status.face_landmarks_visible = not face_covered

            conditions.append(
                AlertCondition(
                    condition_key="atm_face_covered",
                    alert_type="FACE_COVERED",
                    severity=AlertSeverity.MEDIUM,
                    module_name="atm_rules",
                    message="Face covering detected — please remove it to proceed",
                    is_active=face_covered,
                    tracked_ids=[pid],
                    zone_name=zone_name,
                    voice_announcement="Face covering detected. Please remove it to proceed.",
                    details={
                        "track_id": pid,
                        "face_detected": face_detected,
                        "visibility_score": round(frame_vis_score, 2),
                        "rolling_visibility_score": round(rolling_avg_vis, 2),
                        "covered_threshold": self.covered_threshold,
                    },
                )
            )

            if face_covered:
                status.is_safe = False
                status.operational = False
                status.status_label = "Face covering detected — please remove it to proceed"
            else:
                status.is_safe = True
                status.operational = True
                status.status_label = "Safe to use ATM — Face Verified"

            status.debug_info = {
                "person_found": True,
                "person_count": 1,
                "face_found": face_detected,
                "vis_score_pct": round(frame_vis_score * 100.0, 1),
                "rolling_vis_pct": round(rolling_avg_vis * 100.0, 1),
                "state": "COVERED" if face_covered else "SAFE",
            }

        elif person_count == 0:
            status.is_safe = True
            status.operational = True
            status.status_label = "ATM Idle — Ready for Next User"
            conditions.append(
                AlertCondition(
                    condition_key="atm_face_covered",
                    alert_type="FACE_COVERED",
                    severity=AlertSeverity.MEDIUM,
                    module_name="atm_rules",
                    message="Face clear",
                    is_active=False,
                    zone_name=zone_name,
                )
            )

        return status, conditions, all_landmarks

    def close(self):
        """Release MediaPipe resources."""
        if self.mediapipe_landmarker:
            try:
                self.mediapipe_landmarker.close()
            except Exception:
                pass
            self.mediapipe_landmarker = None
