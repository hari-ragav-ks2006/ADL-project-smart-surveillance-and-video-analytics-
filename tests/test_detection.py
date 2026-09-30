"""Unit tests for Core Object Detection and ByteTrack Tracking Engine."""

import numpy as np
import pytest
from detection_model import COCO_CLASS_NAMES, ObjectDetectorTracker, TrackedObject


def test_tracked_object_geometry():
    """Verify centroid and bottom-center calculation."""
    obj = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(100, 200, 300, 600),
    )
    assert obj.center == (200, 400)
    assert obj.bottom_center == (200, 600)
    assert obj.width == 200
    assert obj.height == 400


def test_coco_class_mapping():
    """Verify COCO class mapping contains required classes."""
    assert COCO_CLASS_NAMES[0] == "person"
    assert len(COCO_CLASS_NAMES) > 0


def test_tracked_object_bounding_box():
    """Verify bounding box calculation and aspect ratio."""
    obj = TrackedObject(
        track_id=2,
        class_id=0,
        class_name="person",
        confidence=0.88,
        bbox=(50, 50, 150, 250),
    )
    assert obj.width == 100
    assert obj.height == 200
    assert obj.center == (100, 150)
    assert obj.bottom_center == (100, 250)
