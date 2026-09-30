"""
Polygon Zone & Region of Interest (ROI) Engine (Domain-Agnostic).
Provides geometric spatial containment tests for any tracked objects across customizable polygon areas.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from detection_model import TrackedObject


class PolygonZone:
    """Represents a 2D polygonal Region of Interest."""

    def __init__(
        self,
        name: str,
        polygon: List[List[int]],
        color: Tuple[int, int, int] = (0, 255, 255),  # BGR Yellow default
    ):
        self.name = name
        self.polygon_pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        self.color = color

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a coordinate point is inside the polygon."""
        if len(self.polygon_pts) < 3:
            return False
        # cv2.pointPolygonTest returns > 0 for inside, == 0 on edge, < 0 for outside
        result = cv2.pointPolygonTest(self.polygon_pts, (float(point[0]), float(point[1])), False)
        return result >= 0

    def contains_object(self, obj: TrackedObject, use_bottom_center: bool = True) -> bool:
        """Check if a tracked object is within this zone."""
        test_pt = obj.bottom_center if use_bottom_center else obj.center
        return self.contains_point(test_pt)

    def draw(
        self,
        frame: np.ndarray,
        alpha: float = 0.20,
        draw_border: bool = True,
        draw_label: bool = True,
    ) -> np.ndarray:
        """Render polygon overlay onto frame with semi-transparency."""
        if len(self.polygon_pts) < 3:
            return frame

        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon_pts], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if draw_border:
            cv2.polylines(frame, [self.polygon_pts], isClosed=True, color=self.color, thickness=2)

        if draw_label and len(self.polygon_pts) > 0:
            top_pt = self.polygon_pts[0][0]
            label_text = f"ZONE: {self.name}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (top_pt[0], top_pt[1] - h - 6),
                (top_pt[0] + w + 8, top_pt[1]),
                self.color,
                -1,
            )
            cv2.putText(
                frame,
                label_text,
                (top_pt[0] + 4, top_pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return frame


class ZoneEngine:
    """Manages collection of polygon zones and maps object memberships per frame."""

    def __init__(self, zones: Optional[Dict[str, PolygonZone]] = None):
        self.zones: Dict[str, PolygonZone] = zones or {}

    def add_zone(self, zone_id: str, zone: PolygonZone):
        self.zones[zone_id] = zone

    def process_objects(
        self,
        objects: List[TrackedObject],
    ) -> Dict[str, List[TrackedObject]]:
        """
        Evaluate which tracked objects are located within each defined zone.
        
        Returns:
            Dictionary mapping zone_id to list of TrackedObjects inside that zone.
        """
        memberships: Dict[str, List[TrackedObject]] = {zid: [] for zid in self.zones}

        for obj in objects:
            for zid, zone in self.zones.items():
                if zone.contains_object(obj):
                    memberships[zid].append(obj)

        return memberships

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.20) -> np.ndarray:
        """Render all configured zones on the image frame."""
        for zone in self.zones.values():
            frame = zone.draw(frame, alpha=alpha)
        return frame
