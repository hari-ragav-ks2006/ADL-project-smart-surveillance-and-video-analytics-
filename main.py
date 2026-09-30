"""
Main Application Pipeline Orchestrator and FastAPI Backend.
Integrates YOLOv8 ByteTrack Core Engine, Pluggable Rules, GUI Display Window,
Persistence, WebSocket streaming, and REST API with two isolated operational modes:
- ATM Mode (Live Webcam Only, Face Covering, Multi-Person, Camera Blackout)
- Restricted Zone Mode (Outdoor Pedestrian Video, Zone intrusion rules)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import asyncio
import logging
import os
import sys
import threading
import time

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn

from alert_system import AlertManager
from atm_rules import ATMRuleModule
from config import AppConfig, load_config
from database import DatabaseManager
from detection_model import ObjectDetectorTracker, TrackedObject
from display import DisplayManager
from logger import SurveillanceLogger
from restricted_rules import RestrictedZoneRuleModule
from zone_engine import PolygonZone, ZoneEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SurveillanceMain")


class SurveillancePipeline:
    """
    Coordinates real-time video capture, inference, rule evaluation,
    display rendering, and network broadcasting for a selected surveillance mode.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        mode: str = "atm",
        gui_enabled: bool = True,
        config_path: str = "config.yaml",
        draw_zone: bool = False,
    ):
        self.config: AppConfig = load_config(config_path)
        self.mode = mode.lower()
        if self.mode not in ("atm", "restricted"):
            logger.warning(f"[CONFIG] Unknown mode '{self.mode}', defaulting to 'atm'.")
            self.mode = "atm"

        self.gui_enabled = gui_enabled
        self.draw_zone = draw_zone

        # Auto-resolve video source from config if not explicitly provided
        if source is not None and str(source).strip() != "":
            self.source = str(source)
        else:
            if self.mode == "atm":
                self.source = self.config.video.sources.atm  # Strictly live webcam ("0")
            elif self.mode == "restricted":
                self.source = self.config.video.sources.restricted
            else:
                self.source = self.config.video.default_source

        # Database and logging
        self.db_manager = DatabaseManager(self.config.persistence.db_path)
        self.surveillance_logger = SurveillanceLogger(
            db_manager=self.db_manager,
            screenshots_dir=self.config.persistence.screenshots_dir,
            excel_path=self.config.persistence.excel_export_path,
        )

        # Alert Manager
        tts_enabled = self.config.atm_module.tts_enabled
        if self.mode == "restricted":
            tts_enabled = self.config.restricted_module.tts_enabled
        self.alert_manager = AlertManager(
            logger_instance=self.surveillance_logger,
            config=self.config.alert_system,
            twilio_config=self.config.twilio,
            tts_enabled=tts_enabled,
        )

        # Detection target classes: strictly person (class 0) for both ATM and Restricted modes
        target_classes = [0]

        # Core Engine (Shared across modes, optimized for fast inference)
        self.detector = ObjectDetectorTracker(
            model_name=self.config.detection.model_name,
            confidence_threshold=self.config.detection.confidence_threshold,
            iou_threshold=self.config.detection.iou_threshold,
            tracker_type=self.config.detection.tracker_type,
            target_classes=target_classes,
            device=self.config.system.device,
        )

        # Zone Engine - Registered only for restricted mode (ATM mode has NO zone)
        self.zone_engine = ZoneEngine()
        if self.mode == "restricted":
            if "restricted_zone" in self.config.zones:
                zdef = self.config.zones["restricted_zone"]
                self.zone_engine.add_zone(
                    "restricted_zone",
                    PolygonZone(name=zdef.name, polygon=zdef.polygon, color=tuple(zdef.color)),
                )

        # Pluggable Rule Modules - Strictly instantiate only the active mode module
        self.atm_module: Optional[ATMRuleModule] = None
        self.restricted_module: Optional[RestrictedZoneRuleModule] = None

        if self.mode == "atm":
            self.atm_module = ATMRuleModule(self.config.atm_module)
        elif self.mode == "restricted":
            self.restricted_module = RestrictedZoneRuleModule(self.config.restricted_module)

        # GUI Display Manager with Mode-Specific Title & Window
        mode_titles = {
            "atm": "Smart Surveillance - ATM Security Mode (Live Webcam)",
            "restricted": "Smart Surveillance - Restricted Zone Intrusion Mode",
        }
        self.config.video.display_window.enabled = self.gui_enabled
        self.config.video.display_window.title = mode_titles.get(
            self.mode, f"Smart Surveillance - {self.mode.upper()} Mode"
        )
        self.display = DisplayManager(self.config.video.display_window)

        # Runtime states
        self.running = False
        self.paused = False
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {}
        self.connected_websockets: List[WebSocket] = []
        self.fps_tracker = 0.0
        self.total_frames_processed = 0

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open camera index or video file with strict validation."""
        source_val: Any = self.source
        if str(self.source).isdigit():
            source_val = int(self.source)
            logger.info(f"[SOURCE] Connecting to live local camera index {source_val}...")
        else:
            logger.info(f"[SOURCE] Opening video file: '{source_val}'...")

        cap = cv2.VideoCapture(source_val)

        if not cap.isOpened():
            if self.mode == "atm":
                logger.error(
                    f"[ERROR] Could not open live webcam index '{self.source}'. "
                    f"Please connect a webcam or verify device permissions."
                )
                return None
            else:
                logger.error(
                    f"[ERROR] Could not open video file '{self.source}'. "
                    f"Attempting fallback to default restricted video..."
                )
                fallback_sample = "sample_videos/pedestrians_surveillance.avi"
                if Path(fallback_sample).exists():
                    logger.warning(f"[SOURCE] Falling back to: '{fallback_sample}'")
                    cap = cv2.VideoCapture(fallback_sample)
                    if cap.isOpened():
                        self.source = fallback_sample
                        return cap
                return None

        # Warm up & read test frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error(f"[ERROR] Video source '{self.source}' opened but returned empty frame.")
            cap.release()
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.video.frame_width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.video.frame_height
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not str(self.source).isdigit():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info(
            f"[SOURCE] Active stream confirmed: resolution {w}x{h}, total_frames: {total_frames}, FPS: {fps_val:.1f}"
        )
        print(f"[INFO] Video stream loaded: {self.source} ({w}x{h}, {total_frames} frames @ {fps_val:.1f} FPS)")
        return cap

    def _interactive_draw_zone(self, sample_frame: np.ndarray, zone_id: str = "restricted_zone"):
        """Interactive polygon drawing tool at startup for restricted mode."""
        points: List[List[int]] = []
        window_name = "Define Zone - Click 3+ Points | Press 'C' to Save | Press 'R' to Reset"
        frame_copy = sample_frame.copy()

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.config.video.frame_width, self.config.video.frame_height)
        cv2.setMouseCallback(window_name, mouse_cb)

        logger.info("[ZONE DRAW] Click points on the window to define restricted polygon zone. Press 'C' when done.")

        while True:
            display_img = frame_copy.copy()
            if len(points) > 1:
                cv2.polylines(display_img, [np.array(points, np.int32)], False, (0, 0, 255), 2)
            for p in points:
                cv2.circle(display_img, (p[0], p[1]), 5, (0, 255, 0), -1)

            cv2.putText(
                display_img,
                f"Points: {len(points)} | Left-Click: Add Point | 'C': Confirm | 'R': Reset | 'ESC': Cancel",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("c"), ord("C"), 13):  # Enter or 'c'
                if len(points) >= 3:
                    break
            elif key in (ord("r"), ord("R")):
                points.clear()
            elif key in (27, ord("q"), ord("Q")):
                break

        cv2.destroyWindow(window_name)

        if len(points) >= 3:
            logger.info(f"[ZONE SETUP] Custom polygon zone defined: {points}")
            self.zone_engine.add_zone(
                zone_id,
                PolygonZone(
                    name="Custom Restricted Zone",
                    polygon=points,
                    color=(0, 0, 255),
                ),
            )

    def run_pipeline(self):
        """Main real-time surveillance processing loop."""
        cap = self._open_capture()
        if cap is None:
            print(f"[FATAL] Video source '{self.source}' unavailable. Pipeline exiting.")
            return

        self.running = True

        # Optional interactive zone drawing on the first frame if requested
        if self.mode == "restricted" and self.draw_zone and self.gui_enabled:
            ret, first_frame = cap.read()
            if ret and first_frame is not None:
                first_resized = cv2.resize(
                    first_frame,
                    (self.config.video.frame_width, self.config.video.frame_height),
                )
                self._interactive_draw_zone(first_resized, zone_id=self.config.restricted_module.zone_id)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if self.gui_enabled:
            self.display.init_window()

        self.surveillance_logger.log_system_event(
            event_type="SYSTEM_STARTUP",
            message=f"Surveillance pipeline started on source '{self.source}' in mode '{self.mode}'.",
        )

        frame_count = 0
        last_time = time.time()
        tracked_objects: List[TrackedObject] = []
        atm_status = None
        restricted_status = None
        frame_skip = max(1, int(getattr(self.config.video, "frame_skip", 1)))

        try:
            while self.running:
                if getattr(self, "_reopen_capture_flag", False):
                    self._reopen_capture_flag = False
                    if cap is not None:
                        cap.release()
                        cap = None
                    if str(self.source).lower() not in ("client_stream", "client", "none"):
                        cap = self._open_capture()
                        if cap is None:
                            logger.warning("[PIPELINE] Capture source re-open returned None. Entering client-feed standby mode.")

                if cap is None:
                    time.sleep(0.1)
                    continue

                if self.paused:
                    time.sleep(0.05)
                    key = self.display.render_and_wait(frame, wait_ms=30)
                    if key in (ord("p"), ord("P")):
                        self.paused = False
                    elif key in (ord("q"), ord("Q"), 27):
                        break
                    continue

                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    # If video file ended, loop back to beginning
                    if not str(self.source).isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("[SOURCE] Frame grab failed from camera.")
                        time.sleep(0.05)
                        continue

                frame = cv2.resize(
                    raw_frame,
                    (self.config.video.frame_width, self.config.video.frame_height),
                )
                frame_count += 1
                self.total_frames_processed += 1

                # Frame pacing to maintain real-time FPS (25 FPS) and yield CPU
                time.sleep(0.04)

                # 1. Core Object Detection & Tracking (ByteTrack)
                should_infer = (frame_count % frame_skip == 0) or len(tracked_objects) == 0

                if should_infer:
                    tracked_objects = self.detector.track_frame(frame)
                    zone_memberships = self.zone_engine.process_objects(tracked_objects)

                    # 2. Mode-Specific Rule Evaluation (Strictly Isolated)
                    all_conditions = []

                    if self.mode == "atm" and self.atm_module is not None:
                        # ATM mode evaluates persons across full frame (strictly person class 0, no zone restriction)
                        atm_persons = [obj for obj in tracked_objects if obj.class_id == 0]
                        atm_status, atm_conds, _ = self.atm_module.evaluate_frame(
                            frame, atm_persons, zone_name="ATM Area"
                        )
                        all_conditions.extend(atm_conds)

                    elif self.mode == "restricted" and self.restricted_module is not None:
                        res_zone_id = self.config.restricted_module.zone_id
                        res_zone_objs = zone_memberships.get(res_zone_id, [])
                        res_zone_name = (
                            self.config.zones[res_zone_id].name
                            if res_zone_id in self.config.zones
                            else "Restricted Security Zone"
                        )
                        restricted_status, res_conds = self.restricted_module.evaluate_frame(
                            frame, res_zone_objs, zone_name=res_zone_name
                        )
                        all_conditions.extend(res_conds)

                    # 3. Feed conditions into Alert Manager (Debouncing & Escalation)
                    for cond in all_conditions:
                        debounce_val = 1.5
                        if cond.alert_type == "FACE_COVERED":
                            debounce_val = 0.0  # Managed by rolling window hysteresis in atm_rules
                        elif cond.alert_type == "MULTI_PERSON":
                            debounce_val = self.config.atm_module.multi_person_debounce_sec
                        elif cond.alert_type == "CAMERA_BLACKOUT":
                            debounce_val = 0.0  # Managed by blackout duration in atm_rules
                        elif cond.alert_type == "ZONE_INTRUSION":
                            debounce_val = self.config.restricted_module.intrusion_debounce_sec

                        self.alert_manager.evaluate_condition(
                            condition=cond,
                            frame=frame.copy(),
                            debounce_seconds=debounce_val,
                        )

                # 4. Render Visual Elements
                annotated_frame = frame.copy()

                # Draw polygon zones ONLY for restricted mode (NEVER in ATM mode)
                if self.mode == "restricted" and self.config.video.display_window.show_zones:
                    annotated_frame = self.zone_engine.draw_zones(annotated_frame, alpha=0.18)

                # Determine mode-specific bounding box tags and highlight colors
                custom_labels: Dict[int, str] = {}
                custom_colors: Dict[int, Tuple[int, int, int]] = {}

                if self.mode == "atm" and atm_status is not None:
                    if atm_status.person_count > 1:
                        for p in tracked_objects:
                            custom_labels[p.track_id] = f"#{p.track_id} PERSON: Multiple people"
                            custom_colors[p.track_id] = (0, 140, 255)  # Orange
                    elif atm_status.person_count == 1:
                        p = tracked_objects[0]
                        if not atm_status.face_landmarks_visible:
                            custom_labels[p.track_id] = f"#{p.track_id} PERSON: Face covered"
                            custom_colors[p.track_id] = (0, 140, 255)  # Orange
                        else:
                            custom_labels[p.track_id] = f"#{p.track_id} PERSON: Safe"
                            custom_colors[p.track_id] = (0, 220, 100)  # Green

                elif self.mode == "restricted" and restricted_status is not None:
                    for p in tracked_objects:
                        if p.track_id in restricted_status.intruder_ids:
                            custom_labels[p.track_id] = f"#{p.track_id} PERSON: UNAUTHORIZED"
                            custom_colors[p.track_id] = (0, 0, 255)  # Red (Unauthorized)
                        else:
                            custom_labels[p.track_id] = f"#{p.track_id} PERSON: Normal"
                            custom_colors[p.track_id] = (255, 120, 0)  # Blue (Unrestricted / Normal)

                # Draw tracked bounding boxes with custom labels & colors
                annotated_frame = self.display.draw_tracked_objects(
                    annotated_frame,
                    tracked_objects,
                    custom_labels=custom_labels,
                    custom_colors=custom_colors,
                )

                # Draw mode-specific telemetry & overlays
                operational = True
                if atm_status:
                    operational = operational and atm_status.operational
                    if atm_status.person_count == 1:
                        show_dbg = getattr(self.config.atm_module, "show_debug_visibility", True)
                        annotated_frame = self.display.draw_atm_debug_telemetry(
                            annotated_frame,
                            visibility_score=atm_status.visibility_score,
                            rolling_visibility=atm_status.rolling_visibility_score,
                            is_covered=not atm_status.face_landmarks_visible,
                            enabled=show_dbg,
                        )

                if restricted_status:
                    operational = operational and (not restricted_status.is_breached)

                # Draw top HUD bar (Includes small live FPS counter)
                source_display_name = "LIVE WEBCAM" if str(self.source).isdigit() else Path(str(self.source)).name
                annotated_frame = self.display.draw_hud(
                    annotated_frame,
                    source_name=source_display_name,
                    mode_label=self.mode,
                    tracked_count=len(tracked_objects),
                    operational_flag=operational,
                )

                # Draw active alert banner if any
                active_alerts = self.alert_manager.get_current_active_alerts()
                if active_alerts:
                    top_alert = active_alerts[0]
                    alert_msg = top_alert.get("message", f"{top_alert['alert_type']}")
                    banner_msg = f"ALERT: {alert_msg}"
                    is_critical = (
                        top_alert.get("state") == "HARD_ALERT"
                        or top_alert.get("alert_type") == "CAMERA_BLACKOUT"
                        or top_alert.get("alert_type") == "ZONE_INTRUSION"
                        or (atm_status is not None and atm_status.blackout_active)
                    )
                    banner_sev = "CRITICAL" if is_critical else "WARNING"
                    annotated_frame = self.display.draw_alert_banner(
                        annotated_frame, banner_msg, severity=banner_sev
                    )
                elif self.mode == "atm" and atm_status and atm_status.is_safe and atm_status.person_count == 1:
                    annotated_frame = self.display.draw_alert_banner(
                        annotated_frame, "Safe to use ATM", severity="SAFE"
                    )
                elif self.mode == "restricted" and restricted_status and not restricted_status.is_breached:
                    annotated_frame = self.display.draw_alert_banner(
                        annotated_frame, "Restricted Zone Secure — Perimeter Clear", severity="SAFE"
                    )

                # 5. Display in OpenCV Window & Listen for Keys
                if self.gui_enabled:
                    key = self.display.render_and_wait(annotated_frame, wait_ms=1)
                    if key in (ord("q"), ord("Q"), 27):
                        logger.info("[USER] 'q' pressed. Shutting down surveillance.")
                        break
                    elif key in (ord("p"), ord("P")):
                        self.paused = not self.paused
                    elif key in (ord("s"), ord("S")):
                        # Manual snapshot
                        self.surveillance_logger.log_alert(
                            alert_type="MANUAL_SNAPSHOT",
                            severity="LOW",
                            module_name="user_action",
                            message="Manual snapshot captured by operator.",
                            frame=annotated_frame,
                        )

                # 6. Compress frame for WebSocket clients
                _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self.latest_frame_jpeg = buffer.tobytes()

                now = time.time()
                self.fps_tracker = 1.0 / max(0.001, (now - last_time))
                last_time = now

                custom_colors_list = {k: list(v) for k, v in custom_colors.items()}
                self.latest_telemetry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "fps": round(self.fps_tracker, 1),
                    "source": str(self.source),
                    "mode": self.mode,
                    "total_tracks": len(tracked_objects),
                    "active_alerts": active_alerts,
                    "operational": operational,
                    "is_safe": atm_status.is_safe if atm_status else (not restricted_status.is_breached if restricted_status else True),
                    "is_breached": restricted_status.is_breached if restricted_status else False,
                    "intruder_ids": restricted_status.intruder_ids if restricted_status else [],
                    "status_label": (
                        atm_status.status_label
                        if atm_status
                        else (
                            restricted_status.status_label
                            if restricted_status
                            else "Monitoring Active"
                        )
                    ),
                    "status_level": (
                        "critical"
                        if (restricted_status and restricted_status.is_breached)
                        or (atm_status and (not atm_status.operational or atm_status.blackout_active))
                        else ("warning" if atm_status and not atm_status.is_safe else "safe")
                    ),
                    "person_count": atm_status.person_count if atm_status else len(tracked_objects),
                    "atm_visibility_score": round(atm_status.visibility_score * 100.0, 1) if atm_status else None,
                    "atm_rolling_visibility": round(atm_status.rolling_visibility_score * 100.0, 1) if atm_status else None,
                    "zone_polygon": self.config.zones["restricted_zone"].polygon if "restricted_zone" in self.config.zones else [],
                    "tracked_objects": [
                        {
                            "track_id": o.track_id,
                            "bbox": [int(o.bbox[0]), int(o.bbox[1]), int(o.bbox[2]), int(o.bbox[3])],
                            "class_id": o.class_id,
                            "label": custom_labels.get(o.track_id, f"#{o.track_id} Person"),
                            "color": custom_colors_list.get(o.track_id, [0, 220, 100]),
                            "is_intruder": (
                                o.track_id in restricted_status.intruder_ids
                                if restricted_status
                                else False
                            ),
                        }
                        for o in tracked_objects
                    ],
                }

                # Real-time frame pacing to prevent CPU starvation
                target_dt = 1.0 / max(1.0, float(self.config.video.target_fps))
                elapsed = time.time() - now
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

        finally:
            self.running = False
            if cap is not None:
                cap.release()
            self.display.destroy()
            if self.atm_module is not None:
                self.atm_module.close()
            self.alert_manager.audio.shutdown()
            self.surveillance_logger.log_system_event(
                event_type="SYSTEM_SHUTDOWN",
                message=f"Surveillance pipeline ({self.mode.upper()}) gracefully terminated.",
            )
            logger.info("[SHUTDOWN] Surveillance pipeline terminated cleanly.")

    def switch_mode(self, mode: str, source: Optional[str] = None) -> bool:
        """Dynamically switch operational mode and video source."""
        mode = mode.lower()
        if mode not in ("atm", "restricted"):
            return False
        self.mode = mode
        if source is not None and str(source).strip() != "":
            self.source = str(source)
        else:
            if self.mode == "atm":
                self.source = self.config.video.sources.atm
            elif self.mode == "restricted":
                self.source = self.config.video.sources.restricted

        if self.mode == "atm" and self.atm_module is None:
            self.atm_module = ATMRuleModule(self.config.atm_module)
        elif self.mode == "restricted":
            if self.restricted_module is None:
                self.restricted_module = RestrictedZoneRuleModule(self.config.restricted_module)
            if "restricted_zone" in self.config.zones:
                zdef = self.config.zones["restricted_zone"]
                self.zone_engine.add_zone(
                    "restricted_zone",
                    PolygonZone(name=zdef.name, polygon=zdef.polygon, color=tuple(zdef.color)),
                )

        self._reopen_capture_flag = True
        logger.info(f"[PIPELINE] Switched mode to '{self.mode.upper()}' (source: {self.source})")
        return True

    def process_direct_frame(self, frame_bgr: np.ndarray, mode: Optional[str] = None) -> Dict[str, Any]:
        """Process a direct frame (e.g. from browser webcam or test feed) and return detection telemetry."""
        target_mode = (mode or self.mode).lower()
        h, w = frame_bgr.shape[:2]
        # Optimize frame size: 640x360 for high-speed sub-20ms client streaming
        if w != 640 or h != 360:
            frame_resized = cv2.resize(frame_bgr, (640, 360))
        else:
            frame_resized = frame_bgr

        status_label = "System Active"
        status_level = "safe"
        is_safe = True
        blackout_active = False
        is_breached = False
        intruder_ids: List[int] = []
        atm_telemetry: Dict[str, Any] = {}
        custom_labels: Dict[int, str] = {}
        custom_colors: Dict[int, List[int]] = {}
        tracked_objects: List[TrackedObject] = []

        if target_mode == "atm":
            if self.atm_module is None:
                self.atm_module = ATMRuleModule(self.config.atm_module)

            is_blackout, _ = self.atm_module._check_blackout_tamper(frame_resized)
            precomputed_face = None

            if not is_blackout:
                face_det, face_bbox, vis_score, lms, metrics = self.atm_module._detect_face_and_landmarks(
                    frame_resized, (0, 0, 640, 360)
                )
                
                if face_det and face_bbox:
                    virtual_person = TrackedObject(
                        track_id=1,
                        bbox=face_bbox,
                        class_id=0,
                        confidence=0.95,
                    )
                    atm_persons = [virtual_person]
                    tracked_objects = [virtual_person]
                    precomputed_face = (True, vis_score, lms)
                else:
                    atm_persons = [obj for obj in self.detector.track_frame(frame_resized) if obj.class_id == 0]
                    if len(atm_persons) > 0:
                        tracked_objects = atm_persons
                        precomputed_face = (True, 0.0, [])
                    else:
                        virtual_person = TrackedObject(
                            track_id=1,
                            bbox=(int(640 * 0.20), int(360 * 0.10), int(640 * 0.80), int(360 * 0.90)),
                            class_id=0,
                            confidence=0.90,
                        )
                        atm_persons = [virtual_person]
                        tracked_objects = [virtual_person]
                        precomputed_face = (True, 0.0, [])

            logger.info(f"[ATM-DEBUG] Biometric evaluation: found {len(atm_persons)} person(s)")

            atm_status, atm_conds, atm_landmarks = self.atm_module.evaluate_frame(
                frame_resized, atm_persons, zone_name="ATM Area", precomputed_face=precomputed_face
            )
            for cond in atm_conds:
                deb = 0.0 if cond.alert_type in ("FACE_COVERED", "CAMERA_BLACKOUT") else self.config.atm_module.multi_person_debounce_sec
                self.alert_manager.evaluate_condition(cond, frame=frame_resized.copy(), debounce_seconds=deb)

            status_label = atm_status.status_label
            is_safe = atm_status.is_safe
            blackout_active = atm_status.blackout_active
            if not atm_status.operational or blackout_active:
                status_level = "critical"
            elif not is_safe:
                status_level = "warning"
            else:
                status_level = "safe"

            face_object = None
            if atm_status.face_detected and atm_status.face_bbox:
                face_color = [0, 220, 100] if atm_status.face_landmarks_visible else [0, 140, 255]
                face_label = "FACE: SAFE" if atm_status.face_landmarks_visible else "FACE: COVERED"
                face_object = {
                    "bbox": list(atm_status.face_bbox),
                    "label": face_label,
                    "color": face_color,
                    "is_covered": not atm_status.face_landmarks_visible,
                }

            atm_telemetry = {
                "person_count": atm_status.person_count,
                "face_detected": atm_status.face_detected,
                "face_bbox": list(atm_status.face_bbox) if atm_status.face_bbox else None,
                "face_object": face_object,
                "face_landmarks_visible": atm_status.face_landmarks_visible,
                "visibility_score": round(atm_status.visibility_score * 100.0, 1),
                "rolling_visibility_score": round(atm_status.rolling_visibility_score * 100.0, 1),
                "blackout_active": atm_status.blackout_active,
                "landmarks": atm_landmarks,
                "debug_info": atm_status.debug_info,
            }

            if atm_status.person_count > 1:
                for p in tracked_objects:
                    custom_labels[p.track_id] = f"#{p.track_id} PERSON: Multiple people"
                    custom_colors[p.track_id] = [0, 140, 255]
            elif atm_status.person_count == 1:
                p = tracked_objects[0]
                if not atm_status.face_landmarks_visible:
                    custom_labels[p.track_id] = f"#{p.track_id} USER: Face Covered"
                    custom_colors[p.track_id] = [0, 140, 255]
                else:
                    custom_labels[p.track_id] = f"#{p.track_id} USER: Safe"
                    custom_colors[p.track_id] = [0, 220, 100]

        elif target_mode == "restricted":
            if self.restricted_module is None:
                self.restricted_module = RestrictedZoneRuleModule(self.config.restricted_module)
            tracked_objects = self.detector.track_frame(frame_resized)
            if "restricted_zone" in self.config.zones:
                zdef = self.config.zones["restricted_zone"]
                cur_w, cur_h = frame_resized.shape[1], frame_resized.shape[0]
                scaled_poly = [
                    [int(pt[0] * cur_w / 960.0), int(pt[1] * cur_h / 540.0)]
                    for pt in zdef.polygon
                ]
                self.zone_engine.add_zone(
                    "restricted_zone",
                    PolygonZone(name=zdef.name, polygon=scaled_poly, color=tuple(zdef.color)),
                )
            zone_memberships = self.zone_engine.process_objects(tracked_objects)
            res_zone_id = self.config.restricted_module.zone_id
            res_zone_objs = zone_memberships.get(res_zone_id, [])
            res_zone_name = (
                self.config.zones[res_zone_id].name
                if res_zone_id in self.config.zones
                else "Restricted Security Zone"
            )
            restricted_status, res_conds = self.restricted_module.evaluate_frame(
                frame_resized, res_zone_objs, zone_name=res_zone_name
            )
            for cond in res_conds:
                self.alert_manager.evaluate_condition(
                    cond,
                    frame=frame_resized.copy(),
                    debounce_seconds=0.0 if cond.is_active else self.config.restricted_module.intrusion_debounce_sec,
                )

            is_breached = restricted_status.is_breached
            intruder_ids = restricted_status.intruder_ids
            status_label = restricted_status.status_label
            status_level = "critical" if is_breached else "safe"

            for p in tracked_objects:
                if p.track_id in intruder_ids:
                    custom_labels[p.track_id] = f"#{p.track_id} PERSON: UNAUTHORIZED"
                    custom_colors[p.track_id] = [255, 45, 85]
                else:
                    custom_labels[p.track_id] = f"#{p.track_id} PERSON: Normal"
                    custom_colors[p.track_id] = [0, 200, 255]

        boxes = []
        for obj in tracked_objects:
            boxes.append({
                "track_id": obj.track_id,
                "bbox": [int(obj.bbox[0]), int(obj.bbox[1]), int(obj.bbox[2]), int(obj.bbox[3])],
                "class_id": obj.class_id,
                "label": custom_labels.get(obj.track_id, f"#{obj.track_id} Person"),
                "color": custom_colors.get(obj.track_id, [0, 220, 100]),
                "is_intruder": obj.track_id in intruder_ids,
            })

        active_alerts = self.alert_manager.get_current_active_alerts()
        zone_poly = []
        if "restricted_zone" in self.config.zones:
            zone_poly = self.config.zones["restricted_zone"].polygon

        return {
            "success": True,
            "mode": target_mode,
            "status_label": status_label,
            "status_level": status_level,
            "is_safe": is_safe,
            "is_breached": is_breached,
            "intruder_ids": intruder_ids,
            "person_count": len(tracked_objects),
            "tracked_objects": boxes,
            "active_alerts": active_alerts,
            "atm_telemetry": atm_telemetry,
            "zone_polygon": zone_poly,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global pipeline instance
pipeline_instance: Optional[SurveillancePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler with automatic background engine support."""
    global pipeline_instance
    logger.info("[SERVER] FastAPI Backend initialized.")
    if pipeline_instance is None:
        try:
            pipeline_instance = SurveillancePipeline(mode="restricted", gui_enabled=False)
            t = threading.Thread(target=pipeline_instance.run_pipeline, daemon=True, name="PipelineThread")
            t.start()
            logger.info("[SERVER] Background SurveillancePipeline started.")
        except Exception as e:
            logger.warning(f"[SERVER] Could not auto-start pipeline: {e}")
    yield
    if pipeline_instance:
        pipeline_instance.running = False


# FastAPI Web App
app = FastAPI(
    title="Vision Aura - Smart Video Surveillance and Analytics API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_system_status():
    """Retrieve operational health, active source, FPS, and alert state."""
    global pipeline_instance
    if not pipeline_instance:
        return {
            "status": "idle",
            "message": "Pipeline not running",
            "telemetry": {
                "fps": 0.0,
                "mode": "restricted",
                "total_tracks": 0,
                "active_alerts": [],
                "operational": True,
                "status_label": "System Standby",
                "status_level": "safe",
            },
        }
    return {
        "status": "running" if pipeline_instance.running else "stopped",
        "telemetry": pipeline_instance.latest_telemetry,
    }


@app.post("/api/mode")
def switch_system_mode(payload: Dict[str, Any]):
    """Dynamically switch operational mode ('atm' or 'restricted') and video source."""
    global pipeline_instance
    mode = payload.get("mode", "restricted")
    source = payload.get("source")
    if not pipeline_instance:
        pipeline_instance = SurveillancePipeline(source=source, mode=mode, gui_enabled=False)
        t = threading.Thread(target=pipeline_instance.run_pipeline, daemon=True)
        t.start()
        return {"status": "initialized", "mode": pipeline_instance.mode, "source": str(pipeline_instance.source)}
    success = pipeline_instance.switch_mode(mode=mode, source=source)
    return {
        "status": "switched" if success else "error",
        "mode": pipeline_instance.mode,
        "source": str(pipeline_instance.source),
    }


@app.post("/api/process_frame")
def process_client_frame(payload: Dict[str, Any]):
    """Process a single frame from the client webcam/simulation in real time."""
    global pipeline_instance
    if not pipeline_instance:
        pipeline_instance = SurveillancePipeline(mode="atm", gui_enabled=False)
        t = threading.Thread(target=pipeline_instance.run_pipeline, daemon=True)
        t.start()
    frame_data = payload.get("frame", "")
    mode = payload.get("mode", pipeline_instance.mode)
    if not frame_data:
        raise HTTPException(status_code=400, detail="Missing frame payload")
    import base64
    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        result = pipeline_instance.process_direct_frame(frame, mode=mode)
        return result
    except Exception as e:
        logger.error(f"[PROCESS FRAME] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zone")
def update_zone_polygon(payload: Dict[str, Any]):
    """Update polygon coordinates for the restricted zone."""
    global pipeline_instance
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    polygon = payload.get("polygon", [])
    if len(polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygon must contain at least 3 coordinate points")
    pipeline_instance.config.zones["restricted_zone"].polygon = polygon
    pipeline_instance.zone_engine.add_zone(
        "restricted_zone",
        PolygonZone(name="Custom Restricted Zone", polygon=polygon, color=(0, 0, 255)),
    )
    return {"status": "updated", "polygon": polygon}


@app.get("/api/scenarios")
def get_available_scenarios():
    """List sample test scenarios available for ATM and Restricted surveillance."""
    scenarios = [
        {
            "id": "safe_atm",
            "name": "ATM Safe Transaction (Bare Face)",
            "file": "atm_single_safe.mp4",
            "mode": "atm",
            "description": "Single user with clear visible face. Green status.",
        },
        {
            "id": "face_obscured",
            "name": "ATM Face Covered (Mask / Occlusion)",
            "file": "atm_face_obscured.mp4",
            "mode": "atm",
            "description": "User with face covered. Warning alert.",
        },
        {
            "id": "multi_person",
            "name": "ATM Multi-Person Violation",
            "file": "atm_multi_person.mp4",
            "mode": "atm",
            "description": "Multiple users standing near ATM. Warning alert.",
        },
        {
            "id": "blackout",
            "name": "ATM Camera Blackout / Tampering",
            "file": "camera_blackout.mp4",
            "mode": "atm",
            "description": "Camera blocked for 15+ seconds. Critical escalation.",
        },
        {
            "id": "pedestrians",
            "name": "Restricted Access Intrusion (Single Crossing)",
            "file": "pedestrians_surveillance_clean.mp4",
            "mode": "restricted",
            "description": "Perimeter surveillance with single clean intrusion event and normal pedestrian background flow.",
        },
        {
            "id": "pedestrians_full",
            "name": "Full Outdoor Pedestrian Stream",
            "file": "pedestrians_surveillance.mp4",
            "mode": "restricted",
            "description": "Continuous outdoor pedestrian flow across facility courtyard.",
        },
    ]
    return {"scenarios": scenarios}


@app.post("/api/shutdown")
def shutdown_surveillance():
    """Remotely shutdown the camera capture and surveillance pipeline."""
    if pipeline_instance:
        pipeline_instance.running = False
        pipeline_instance.display.shutdown_requested = True
    return {"status": "shutting_down", "message": "Camera surveillance system turning off."}


@app.get("/api/alerts")
def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    status: Optional[str] = None,
    module: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Query alert records from the database."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    alerts = pipeline_instance.surveillance_logger.query_alerts(
        severity=severity, alert_type=alert_type, status=status, module_name=module, limit=limit
    )
    return {"count": len(alerts), "alerts": alerts}


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    """Manually resolve an alert."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    record = pipeline_instance.surveillance_logger.resolve_alert(alert_id, resolution_status="RESOLVED")
    if not record:
        raise HTTPException(status_code=404, detail=f"Alert #{alert_id} not found")
    return {"status": "success", "alert": record.to_dict()}


@app.get("/api/reports/excel")
def download_excel_report():
    """Export and download Excel report of all logged alerts."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    file_path = pipeline_instance.surveillance_logger.export_to_excel()
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Excel report file not found")
    return FileResponse(
        path=file_path,
        filename="surveillance_alerts.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/config")
def get_config():
    """Return runtime configuration parameters."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"config": pipeline_instance.config}


@app.post("/api/config")
def update_config(updates: Dict[str, Any]):
    """Update runtime debounce and threshold settings."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if "face_covered_debounce_sec" in updates:
        pipeline_instance.config.atm_module.face_covered_debounce_sec = float(
            updates["face_covered_debounce_sec"]
        )
    if "multi_person_debounce_sec" in updates:
        pipeline_instance.config.atm_module.multi_person_debounce_sec = float(
            updates["multi_person_debounce_sec"]
        )
    if "intrusion_debounce_sec" in updates:
        pipeline_instance.config.restricted_module.intrusion_debounce_sec = float(
            updates["intrusion_debounce_sec"]
        )
    return {"status": "updated", "config": pipeline_instance.config}


@app.get("/api/screenshots/{filename}")
def get_screenshot(filename: str):
    """Serve captured alert screenshot image."""
    safe_name = os.path.basename(filename)
    path = Path("logs/screenshots") / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(path=str(path), media_type="image/jpeg")


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """WebSocket stream for real-time video frames and telemetry."""
    await websocket.accept()
    try:
        while True:
            if pipeline_instance and pipeline_instance.latest_frame_jpeg:
                import base64

                frame_b64 = base64.b64encode(pipeline_instance.latest_frame_jpeg).decode("utf-8")
                payload = {
                    "frame": f"data:image/jpeg;base64,{frame_b64}",
                    "telemetry": pipeline_instance.latest_telemetry,
                }
                await websocket.send_json(payload)
            await asyncio.sleep(1.0 / 20.0)  # 20 FPS stream
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WS] WebSocket client disconnected: {e}")


@app.websocket("/ws/client-feed")
async def websocket_client_stream(websocket: WebSocket):
    """
    High-throughput bidirectional WebSocket stream for real-time client frame analysis.
    Uses an async single-slot queue and thread-pool execution to drop stale frames and eliminate latency.
    """
    await websocket.accept()
    global pipeline_instance
    if not pipeline_instance:
        pipeline_instance = SurveillancePipeline(mode="atm", gui_enabled=False)
        t = threading.Thread(target=pipeline_instance.run_pipeline, daemon=True)
        t.start()

    loop = asyncio.get_running_loop()
    # Single-slot queue: Always drops older unprocessed frames to prevent queue backlog
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    is_active = True

    async def frame_worker():
        import base64
        while is_active:
            try:
                item = await frame_queue.get()
                if item is None:
                    break
                frame_data, mode, client_t = item

                if "," in frame_data:
                    frame_data = frame_data.split(",", 1)[1]
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None and pipeline_instance is not None:
                    t_infer_start = time.time()
                    # Execute CPU-intensive inference in thread pool without blocking event loop
                    result = await loop.run_in_executor(
                        None,
                        pipeline_instance.process_direct_frame,
                        frame,
                        mode,
                    )
                    t_infer_ms = (time.time() - t_infer_start) * 1000.0
                    result["backend_infer_ms"] = round(t_infer_ms, 1)
                    if client_t is not None:
                        result["client_t"] = client_t

                    logger.info(f"[ATM-DEBUG][Stage F] Backend sending WS response: is_safe={result.get('is_safe')}, person_count={result.get('person_count')}, backend_infer_ms={result.get('backend_infer_ms')}")
                    await websocket.send_json(result)
                frame_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[WS Worker] Error processing client frame: {e}")

    worker_task = asyncio.create_task(frame_worker())

    try:
        while True:
            msg = await websocket.receive_json()
            frame_data = msg.get("frame", "")
            mode = msg.get("mode", "atm")
            client_t = msg.get("t", None)

            if frame_data:
                logger.info(f"[ATM-DEBUG][Stage C] Backend WS received frame: payload_len={len(frame_data)}, mode={mode}, client_t={client_t}")
                # If queue already has a pending frame, evict it to always process the newest frame only
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        pass
                try:
                    frame_queue.put_nowait((frame_data, mode, client_t))
                except asyncio.QueueFull:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WS] Client feed disconnected: {e}")
    finally:
        is_active = False
        try:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except Exception:
                    pass
            await frame_queue.put(None)
        except Exception:
            pass
        worker_task.cancel()


# Mount static asset directories
sample_dir = Path("sample_videos")
if sample_dir.exists():
    app.mount("/sample_videos", StaticFiles(directory="sample_videos"), name="sample_videos")

screenshots_dir = Path("logs/screenshots")
screenshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/logs/screenshots", StaticFiles(directory="logs/screenshots"), name="screenshots_mount")

static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
audio_dir = Path("static/audio")
audio_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static_mount")
app.mount("/audio", StaticFiles(directory="static/audio"), name="audio_mount")
app.mount("/", StaticFiles(directory="static", html=True), name="static_root")


def prompt_mode_selection() -> str:
    """Present interactive console menu for selecting surveillance mode."""
    print("\n" + "=" * 68)
    print("      SMART VIDEO SURVEILLANCE & ANALYTICS - MODE SELECTION")
    print("=" * 68)
    print(" Please select an operational mode:\n")
    print("  [1] ATM Security Mode")
    print("      • Video Source: Live Laptop Webcam (Index: 0)")
    print("      • Active Rules: Multi-Person Violation, Face Obscured, Camera Blackout\n")
    print("  [2] Restricted Zone Intrusion Mode")
    print("      • Video Source: Outdoor Pedestrian Surveillance Video (pedestrians_surveillance.avi)")
    print("      • Active Rules: Restricted North Access Road Polygon Zone Intrusion")
    print("=" * 68)

    try:
        choice = input(" Enter selection [1, 2 or atm, restricted] (default: 1): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    mapping = {
        "1": "atm",
        "atm": "atm",
        "2": "restricted",
        "restricted": "restricted",
        "zone": "restricted",
    }
    selected_mode = mapping.get(choice, "atm")
    print(f"\n[INFO] Selected Mode: {selected_mode.upper()}\n")
    return selected_mode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Video Surveillance and Analytics Engine"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["atm", "restricted"],
        default=None,
        help="Active surveillance rule mode ('atm', 'restricted'). If omitted, a selection menu is displayed.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source override (webcam index '0' or path to video file / RTSP stream).",
    )
    parser.add_argument(
        "--draw-zone",
        action="store_true",
        help="Interactively draw polygon zone coordinates on the first video frame at startup.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable the OpenCV GUI window (headless/server mode).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI Web & WebSocket server port (default: 8000).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="FastAPI Web server host address (default: 0.0.0.0).",
    )
    return parser.parse_args()


def main():
    global pipeline_instance
    import atexit
    import signal

    args = parse_args()

    # If --mode was not passed via CLI, prompt user interactively
    mode = args.mode
    if mode is None:
        mode = prompt_mode_selection()

    pipeline_instance = SurveillancePipeline(
        source=args.source,
        mode=mode,
        gui_enabled=not args.no_gui,
        draw_zone=args.draw_zone,
    )

    def cleanup():
        global pipeline_instance
        if pipeline_instance is not None:
            pipeline_instance.running = False
            try:
                pipeline_instance.alert_manager.audio.shutdown()
            except Exception:
                pass

    atexit.register(cleanup)

    def sig_handler(signum, frame):
        cleanup()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)
    except Exception:
        pass

    print("\n" + "=" * 65)
    print(" SMART VIDEO SURVEILLANCE AND ANALYTICS SYSTEM")
    print("=" * 65)
    print(f" • Mode        : {pipeline_instance.mode.upper()}")
    print(f" • Video Source: {pipeline_instance.source}")
    print(f" • GUI Window  : {'DISABLED (--no-gui)' if args.no_gui else f'ENABLED ({pipeline_instance.config.video.display_window.title})'}")
    print(f" • Web Server  : http://localhost:{args.port}")
    print("=" * 65 + "\n")

    # Start FastAPI Web Server in a daemon background thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=args.host, port=args.port, log_level="warning"),
        daemon=True,
        name="SurveillanceAPIServer",
    )
    server_thread.start()

    try:
        # Start the core surveillance processing pipeline on the main thread
        pipeline_instance.run_pipeline()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
