"""
GUI Display Renderer and OpenCV Window Manager.
Responsible for rendering high-FPS overlays, HUD telemetry, bounding boxes,
polygon zones, facial landmarks, and alert banners with guaranteed window responsiveness.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import cv2
import numpy as np

from config import DisplayWindowConfig
from detection_model import TrackedObject

logger = logging.getLogger(__name__)

# Curated palette for high contrast and visual clarity
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 178, 50),     # Vibrant Sky Blue / Cyan (BGR)
    "car": (80, 200, 120),         # Emerald Green
    "motorcycle": (0, 165, 255),   # Orange
    "bicycle": (200, 130, 255),    # Lavender
}
DEFAULT_COLOR = (200, 200, 200)


class DisplayManager:
    """
    Manages OpenCV GUI window creation, overlay rendering, and keypress/mouse handling.
    Guarantees visible rendering and event pump via cv2.waitKey.
    """

    def __init__(self, config: Optional[DisplayWindowConfig] = None):
        self.config = config or DisplayWindowConfig()
        self.window_name = self.config.title
        self.width = self.config.width
        self.height = self.config.height
        self.is_initialized = False
        self.shutdown_requested = False
        self.quit_button_rect = (self.width - 125, 6, self.width - 10, 36)

        # FPS calculation tracking
        self.prev_time = time.time()
        self.current_fps = 0.0

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on GUI elements like the Quit button."""
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = self.quit_button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                logger.info("[DISPLAY] Quit button clicked by mouse. Initiating shutdown...")
                self.shutdown_requested = True

    def init_window(self):
        """Create and size the OpenCV display window and attach mouse listener."""
        if not self.config.enabled:
            return

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self.is_initialized = True
            logger.info(
                f"[DISPLAY] Window '{self.window_name}' created and active at {self.width}x{self.height}"
            )
            print(f"[INFO] Display window active ({self.width}x{self.height}): '{self.window_name}'")
        except Exception as e:
            logger.error(f"[DISPLAY] Failed to create OpenCV window: {e}")
            self.is_initialized = False

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        return PALETTE.get(class_name.lower(), DEFAULT_COLOR)

    def draw_hud(
        self,
        frame: np.ndarray,
        source_name: str,
        mode_label: str,
        tracked_count: int,
        operational_flag: bool = True,
    ) -> np.ndarray:
        """Render top status banner with telemetry and on-screen clickable Quit button."""
        h, w = frame.shape[:2]
        self.quit_button_rect = (w - 130, 6, w - 10, 36)

        # Top HUD semi-transparent bar
        header_height = 42
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, header_height), (w, header_height), (60, 70, 85), 1)

        # Mode Badge
        mode_color = (0, 200, 100) if operational_flag else (0, 140, 255)
        cv2.rectangle(frame, (12, 8), (140, 34), mode_color, -1)
        cv2.putText(
            frame,
            mode_label.upper(),
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Source & FPS Info
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)
        self.prev_time = now

        info_text = f"SRC: {source_name} | FPS: {self.current_fps:.1f} | TRACKS: {tracked_count}"
        cv2.putText(
            frame,
            info_text,
            (150, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 225, 230),
            1,
            cv2.LINE_AA,
        )

        # ATM / System Status Pill
        op_text = "NORMAL" if operational_flag else "ACTION REQ"
        op_bg = (40, 140, 50) if operational_flag else (30, 80, 220)
        cv2.rectangle(frame, (w - 270, 8), (w - 145, 34), op_bg, -1)
        cv2.putText(
            frame,
            op_text,
            (w - 260, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # ON-SCREEN CLICKABLE QUIT BUTTON [✖ QUIT (Q)]
        qx1, qy1, qx2, qy2 = self.quit_button_rect
        cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (0, 0, 220), -1)  # Red Button
        cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (255, 255, 255), 1)
        cv2.putText(
            frame,
            "QUIT (Q)",
            (qx1 + 10, qy1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

        return frame

    def draw_tracked_objects(
        self,
        frame: np.ndarray,
        objects: List[TrackedObject],
        custom_labels: Optional[Dict[int, str]] = None,
        custom_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Render labeled bounding boxes with tracking IDs and mode-specific highlight colors."""
        custom_labels = custom_labels or {}
        custom_colors = custom_colors or {}

        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            color = custom_colors.get(obj.track_id, self._get_class_color(obj.class_name))

            # Draw smooth bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label tag
            if obj.track_id in custom_labels:
                label = custom_labels[obj.track_id]
            else:
                label = f"#{obj.track_id} {obj.class_name.upper()} {int(obj.confidence * 100)}%"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            tag_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if (color[1] > 180 or color[2] > 200) else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Render facial landmark dots."""
        for pt in landmarks:
            cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)
        return frame

    def draw_atm_debug_telemetry(
        self,
        frame: np.ndarray,
        visibility_score: float,
        rolling_visibility: float,
        is_covered: bool,
        enabled: bool = True,
    ) -> np.ndarray:
        """
        Render small, corner on-screen debug text showing live rolling face visibility percentage
        and state (SAFE vs COVERED) for testing / evaluation verification.
        """
        if not enabled:
            return frame

        h, w = frame.shape[:2]
        # Position badge in top-right area below HUD
        pad_x = 12
        badge_w = 210
        badge_h = 28
        x2 = w - pad_x
        x1 = x2 - badge_w
        y1 = 48
        y2 = y1 + badge_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border color based on status
        border_color = (0, 70, 230) if is_covered else (0, 200, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

        state_label = "COVERED" if is_covered else "SAFE"
        pct_str = f"VIS: {rolling_visibility * 100.0:.1f}% [{state_label}]"
        cv2.putText(
            frame,
            pct_str,
            (x1 + 8, y1 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def draw_alert_banner(
        self,
        frame: np.ndarray,
        message: str,
        severity: str = "SAFE",  # "SAFE", "WARNING", "CRITICAL"
    ) -> np.ndarray:
        """Render wide bottom alert banner."""
        h, w = frame.shape[:2]
        banner_h = 48
        y_start = h - banner_h

        # Background color
        if severity == "SAFE":
            bg_color = (30, 140, 50)     # Green
            txt_color = (255, 255, 255)
        elif severity in ("WARNING", "SOFT_ALERT", "MEDIUM", "HIGH"):
            bg_color = (0, 120, 240)     # Amber / Orange
            txt_color = (0, 0, 0)
        else:
            bg_color = (0, 0, 220)       # Bright Red (CRITICAL)
            txt_color = (255, 255, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Border
        cv2.line(frame, (0, y_start), (w, y_start), (255, 255, 255), 1)

        # Text
        cv2.putText(
            frame,
            message,
            (20, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            txt_color,
            2,
            cv2.LINE_AA,
        )

        return frame

    def render_and_wait(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """
        Display the rendered frame in the named window and pump the OpenCV GUI loop.
        
        Returns:
            Key code pressed by user or triggered by mouse (ord('q') on quit, or -1).
        """
        if self.shutdown_requested:
            return ord("q")

        if not self.config.enabled:
            return -1

        if not self.is_initialized:
            self.init_window()

        # Resize frame to target display resolution if needed
        disp_frame = frame
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            disp_frame = cv2.resize(frame, (self.width, self.height))

        try:
            cv2.imshow(self.window_name, disp_frame)
            key = cv2.waitKey(wait_ms) & 0xFF
            if self.shutdown_requested:
                return ord("q")
            return key
        except Exception as e:
            logger.error(f"[DISPLAY] cv2.imshow render error: {e}")
            return -1

    def destroy(self):
        """Safely destroy OpenCV window."""
        if self.is_initialized:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.is_initialized = False
