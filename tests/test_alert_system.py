from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from alert_system import (
    ActiveAlertTracker,
    AlertCondition,
    AlertManager,
    AlertSeverity,
    AlertState,
)
from config import AlertSystemConfig
from database import DatabaseManager
from logger import SurveillanceLogger

def get_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_alert_debouncing_and_escalation(tmp_path):
    """Verify alert transitions from NORMAL -> SOFT_ALERT -> HARD_ALERT."""
    db_file = tmp_path / "test_surveillance.db"
    db_manager = DatabaseManager(str(db_file))
    logger_inst = SurveillanceLogger(
        db_manager,
        screenshots_dir=str(tmp_path / "screenshots"),
        excel_path=str(tmp_path / "alerts.xlsx"),
    )

    config = AlertSystemConfig(escalation_timeout_sec=0.2, auto_resolve_debounce_sec=0.1)
    alert_mgr = AlertManager(logger_instance=logger_inst, config=config, tts_enabled=False)

    cond = AlertCondition(
        condition_key="test_key",
        alert_type="TEST_ALERT",
        severity=AlertSeverity.HIGH,
        module_name="test_mod",
        message="Test alert message",
        is_active=True,
    )

    # 1. First evaluation: initiates tracker in NORMAL state
    alert_mgr.evaluate_condition(cond, debounce_seconds=0.1)
    tracker = alert_mgr.active_trackers.get("test_key")
    assert tracker is not None
    assert tracker.state == AlertState.NORMAL

    # 2. Shift time past debounce -> triggers SOFT_ALERT
    tracker.first_detected_at = get_utc_now() - timedelta(seconds=0.15)
    alert_mgr.evaluate_condition(cond, debounce_seconds=0.1)
    assert tracker.state == AlertState.SOFT_ALERT

    # 3. Shift time past escalation timeout -> escalates to HARD_ALERT
    tracker.first_detected_at = get_utc_now() - timedelta(seconds=0.45)
    alert_mgr.evaluate_condition(cond, debounce_seconds=0.1)
    assert tracker.state == AlertState.HARD_ALERT

    # 4. Condition clears -> Auto-resolves
    cond.is_active = False
    tracker.last_cleared_at = get_utc_now() - timedelta(seconds=0.2)
    alert_mgr.evaluate_condition(cond, debounce_seconds=0.1)
    assert "test_key" not in alert_mgr.active_trackers
