"""Unit tests for Restricted Zone & Doorway Security Module."""

import numpy as np
import pytest
from config import RestrictedModuleConfig
from detection_model import TrackedObject
from restricted_zone_rules import RestrictedZoneRuleModule


def test_restricted_zone_doorway_intrusion_detection():
    """Verify intrusion alert triggers when tracked person enters restricted doorway polygon."""
    config = RestrictedModuleConfig(intrusion_debounce_sec=0.5, target_classes=[0])
    module = RestrictedZoneRuleModule(config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Person inside restricted doorway zone
    intruder = TrackedObject(
        track_id=101,
        class_id=0,
        class_name="person",
        confidence=0.91,
        bbox=(250, 200, 350, 450),
    )

    status, conditions = module.evaluate_frame(frame, [intruder], zone_name="Restricted Doorway")
    assert status.is_breached is True
    assert status.intruder_count == 1
    assert status.intruder_ids == [101]

    intrusion_cond = [c for c in conditions if c.alert_type == "ZONE_INTRUSION"][0]
    assert intrusion_cond.is_active is True
    assert intrusion_cond.tracked_ids == [101]
    assert intrusion_cond.zone_name == "Restricted Doorway"
    assert intrusion_cond.message == "Unauthorized entry detected in restricted area — please check"


def test_restricted_zone_clear_when_empty():
    """Verify zone reports secure and inactive alert when no intruders are present."""
    config = RestrictedModuleConfig()
    module = RestrictedZoneRuleModule(config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    status, conditions = module.evaluate_frame(frame, [], zone_name="Restricted Doorway")
    assert status.is_breached is False
    assert status.intruder_count == 0
    assert status.intruder_ids == []

    intrusion_cond = [c for c in conditions if c.alert_type == "ZONE_INTRUSION"][0]
    assert intrusion_cond.is_active is False
