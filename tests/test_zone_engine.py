"""Unit tests for Polygon Zone & ROI Engine."""

import pytest
from detection_model import TrackedObject
from zone_engine import PolygonZone, ZoneEngine


def test_polygon_zone_point_containment():
    """Verify point-in-polygon tests."""
    zone = PolygonZone(
        name="TestZone",
        polygon=[[100, 100], [400, 100], [400, 400], [100, 400]],
    )
    # Inside
    assert zone.contains_point((200, 200)) is True
    # Outside
    assert zone.contains_point((50, 50)) is False
    assert zone.contains_point((500, 300)) is False


def test_polygon_zone_object_containment():
    """Verify tracked object bottom-center containment."""
    zone = PolygonZone(
        name="ATM_Zone",
        polygon=[[100, 100], [500, 100], [500, 500], [100, 500]],
    )
    # Object with bottom center at (200, 450) -> Inside
    inside_obj = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(150, 200, 250, 450),
    )
    # Object with bottom center at (50, 300) -> Outside
    outside_obj = TrackedObject(
        track_id=2,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(0, 100, 100, 300),
    )

    assert zone.contains_object(inside_obj) is True
    assert zone.contains_object(outside_obj) is False


def test_zone_engine_mapping():
    """Verify ZoneEngine aggregates object memberships."""
    engine = ZoneEngine()
    zone_a = PolygonZone(name="ZoneA", polygon=[[0, 0], [200, 0], [200, 200], [0, 200]])
    zone_b = PolygonZone(name="ZoneB", polygon=[[300, 300], [500, 300], [500, 500], [300, 500]])

    engine.add_zone("zone_a", zone_a)
    engine.add_zone("zone_b", zone_b)

    obj1 = TrackedObject(1, 0, "person", 0.9, (50, 50, 100, 100))
    obj2 = TrackedObject(2, 0, "person", 0.9, (350, 350, 400, 400))

    memberships = engine.process_objects([obj1, obj2])
    assert len(memberships["zone_a"]) == 1
    assert memberships["zone_a"][0].track_id == 1
    assert len(memberships["zone_b"]) == 1
    assert memberships["zone_b"][0].track_id == 2
