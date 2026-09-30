"""Unit tests for ATM Security Module with Two-Signal Face Covering Logic."""

from datetime import datetime, timedelta
import numpy as np
import pytest
from atm_rules import ATMRuleModule
from config import ATMModuleConfig
from detection_model import TrackedObject


@pytest.fixture(scope="module")
def shared_atm_module():
    """Module-scoped ATMRuleModule instance (MediaPipe loaded once)."""
    config = ATMModuleConfig(
        visibility_window_size=30,
        covered_threshold=0.30,
        safe_recovery_threshold=0.70,
        multi_person_debounce_sec=1.0,
        blackout_duration_sec=0.1,
        blackout_variance_threshold=20.0,
    )
    module = ATMRuleModule(config)
    yield module
    module.close()


@pytest.fixture
def atm_instance(shared_atm_module):
    """Clean state wrapper per test."""
    shared_atm_module.face_visibility_history.clear()
    shared_atm_module.face_covered_state.clear()
    shared_atm_module.face_state_start_time.clear()
    shared_atm_module.blackout_start_time = None
    shared_atm_module.simulated_dispatch_logged = False
    return shared_atm_module


def test_atm_multi_person_rule(atm_instance):
    """Verify multi-person violation detection when >1 person is present."""
    atm = atm_instance
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    p1 = TrackedObject(1, 0, "person", 0.9, (10, 10, 50, 90))
    p2 = TrackedObject(2, 0, "person", 0.88, (55, 10, 95, 90))

    status, conditions, _ = atm.evaluate_frame(frame, [p1, p2])
    assert status.operational is False
    assert status.is_safe is False
    assert status.person_count == 2

    multi_cond = [c for c in conditions if c.alert_type == "MULTI_PERSON"][0]
    assert multi_cond.is_active is True
    assert multi_cond.tracked_ids == [1, 2]


def test_atm_single_person_face_covering_detected(atm_instance, monkeypatch):
    """Verify face covering requires rolling average visibility score < 30% before triggering alert."""
    atm = atm_instance
    frame = np.full((100, 100, 3), 40, dtype=np.uint8)
    p = TrackedObject(1, 0, "person", 0.92, (20, 10, 80, 90))

    # Mock face visibility: covered face (vis score = 0.0)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 0.0, []))
    monkeypatch.setattr(atm, "_check_blackout_tamper", lambda f: (False, 50.0))

    # Frame 1: Single frame -> NOT yet triggered
    status1, conds1, _ = atm.evaluate_frame(frame, [p])
    assert status1.person_count == 1
    assert status1.is_safe is True
    face_cond1 = [c for c in conds1 if c.alert_type == "FACE_COVERED"][0]
    assert face_cond1.is_active is False

    # Frames 2-15: Continuous absence sustained (vis score = 0.0 < 0.30) -> triggers alert
    for _ in range(14):
        status_sustained, conds_sustained, _ = atm.evaluate_frame(frame, [p])

    assert status_sustained.face_landmarks_visible is False
    assert status_sustained.is_safe is False
    face_cond_sustained = [c for c in conds_sustained if c.alert_type == "FACE_COVERED"][0]
    assert face_cond_sustained.is_active is True
    assert face_cond_sustained.message == "Face covering detected — please remove it to proceed"


def test_atm_face_visible_still_30_seconds_zero_false_flips(atm_instance, monkeypatch):
    """Test explicitly: face fully visible and still -> stays 'safe' 100% of 30-sec test with zero false flips."""
    atm = atm_instance
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    p = TrackedObject(1, 0, "person", 0.95, (20, 10, 80, 90))

    # Mock perfectly visible bare face (100% visibility score)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 1.0, [(50, 50)]))
    monkeypatch.setattr(atm, "_check_blackout_tamper", lambda f: (False, 50.0))

    # 30 seconds at 30 FPS = 900 frames
    for i in range(900):
        status, conds, _ = atm.evaluate_frame(frame, [p])
        assert status.is_safe is True, f"False flip at frame {i}"
        assert status.face_landmarks_visible is True
        assert status.rolling_visibility_score == 1.0
        face_cond = [c for c in conds if c.alert_type == "FACE_COVERED"][0]
        assert face_cond.is_active is False


def test_atm_face_talking_moving_naturally_zero_false_flips(atm_instance, monkeypatch):
    """Test explicitly: face fully visible but talking/moving naturally -> stays 'safe' throughout."""
    atm = atm_instance
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    p = TrackedObject(1, 0, "person", 0.95, (20, 10, 80, 90))

    # Simulate realistic speaking & natural motion variations:
    # Visibility score fluctuates between 75% and 100% with occasional single-frame drop to 55%
    np.random.seed(42)
    scores = np.random.choice([1.0, 0.89, 0.78, 0.89, 1.0, 0.67], size=900)
    monkeypatch.setattr(atm, "_check_blackout_tamper", lambda f: (False, 50.0))

    for i, score in enumerate(scores):
        monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox, s=score: (True, float(s), [(50, 50)]))
        status, conds, _ = atm.evaluate_frame(frame, [p])
        assert status.is_safe is True, f"False positive triggered while talking at frame {i} (score={score}, rolling={status.rolling_visibility_score})"
        assert status.face_landmarks_visible is True
        face_cond = [c for c in conds if c.alert_type == "FACE_COVERED"][0]
        assert face_cond.is_active is False


def test_atm_hand_mask_kerchief_transition_within_one_second(atm_instance, monkeypatch):
    """Test explicitly: hand/mask/kerchief over nose+mouth -> transitions to 'covered' within ~1s and stays stable."""
    atm = atm_instance
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    p = TrackedObject(1, 0, "person", 0.95, (20, 10, 80, 90))
    monkeypatch.setattr(atm, "_check_blackout_tamper", lambda f: (False, 50.0))

    # Phase 1: 30 frames of safe face (1.0 visibility score)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 1.0, [(50, 50)]))
    for _ in range(30):
        status, _, _ = atm.evaluate_frame(frame, [p])
    assert status.is_safe is True

    # Phase 2: Covering placed over nose+mouth (visibility score drops to 0.0-0.10)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 0.0, []))

    transition_frame = None
    for f in range(1, 35):
        status, conds, _ = atm.evaluate_frame(frame, [p])
        if not status.face_landmarks_visible and transition_frame is None:
            transition_frame = f

    # Must transition within ~30 frames (~1.0 second at 30 fps)
    assert transition_frame is not None, "Did not transition to covered state"
    assert transition_frame <= 30, f"Transition took too long: {transition_frame} frames"
    assert status.is_safe is False
    assert status.face_landmarks_visible is False

    # Must remain stably covered for next 100 frames
    for _ in range(100):
        status, conds, _ = atm.evaluate_frame(frame, [p])
        assert status.is_safe is False
        assert status.face_landmarks_visible is False
        face_cond = [c for c in conds if c.alert_type == "FACE_COVERED"][0]
        assert face_cond.is_active is True


def test_atm_hysteresis_and_recovery(atm_instance, monkeypatch):
    """Verify hysteresis: 30% down to trigger covered, 70% up to recover to safe."""
    atm = atm_instance
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    p = TrackedObject(1, 0, "person", 0.95, (20, 10, 80, 90))
    monkeypatch.setattr(atm, "_check_blackout_tamper", lambda f: (False, 50.0))

    # Trigger covered state (30 frames of 0.0)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 0.0, []))
    for _ in range(30):
        status, _, _ = atm.evaluate_frame(frame, [p])
    assert status.is_safe is False

    # Ambiguous intermediate score: 50% visibility (above 30% but below 70%)
    # Hysteresis must keep state as COVERED without flickering
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 0.50, []))
    for _ in range(30):
        status_mid, _, _ = atm.evaluate_frame(frame, [p])
    assert status_mid.is_safe is False, "Hysteresis failed: 50% score should remain covered"

    # Full recovery: 100% visibility for 30 frames -> rolling avg exceeds 70% -> recovers to SAFE
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 1.0, [(50, 50)]))
    for _ in range(30):
        status_rec, conds_rec, _ = atm.evaluate_frame(frame, [p])

    assert status_rec.is_safe is True
    assert status_rec.face_landmarks_visible is True
    face_cond_rec = [c for c in conds_rec if c.alert_type == "FACE_COVERED"][0]
    assert face_cond_rec.is_active is False


def test_atm_rolling_dropout_resilience(atm_instance, monkeypatch):
    """Verify single-frame dropouts or short glitches do NOT trigger false face covering alert."""
    atm = atm_instance
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    p = TrackedObject(1, 0, "person", 0.95, (20, 10, 80, 90))
    monkeypatch.setattr(atm, "_check_blackout_tamper", lambda f: (False, 50.0))

    # Feed 25 safe frames (100% visible)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 1.0, [(50, 50)]))
    for _ in range(25):
        status, conds, _ = atm.evaluate_frame(frame, [p])
    assert status.is_safe is True

    # 1-3 momentary frame dropouts (glitches)
    monkeypatch.setattr(atm, "_check_face_visibility", lambda f, bbox: (True, 0.0, []))
    for _ in range(3):
        status_drop, conds_drop, _ = atm.evaluate_frame(frame, [p])

    # Rolling visibility is 25/28 = ~89%, well above 30% covered threshold
    assert status_drop.is_safe is True
    face_cond = [c for c in conds_drop if c.alert_type == "FACE_COVERED"][0]
    assert face_cond.is_active is False


def test_atm_blackout_tamper_rule(atm_instance):
    """Verify blackout tamper alert triggers on near-zero variance."""
    atm = atm_instance

    # Completely uniform black frames
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 1
    atm.evaluate_frame(black_frame, [])
    if atm.blackout_start_time:
        atm.blackout_start_time -= timedelta(seconds=1.0)

    # Frame 2
    status, conditions, _ = atm.evaluate_frame(black_frame, [])
    blackout_cond = [c for c in conditions if c.alert_type == "CAMERA_BLACKOUT"][0]
    assert blackout_cond.is_active is True
    assert status.blackout_active is True
    assert blackout_cond.trigger_phone_call is True
