"""
Alert Management System: State Machine, Debouncer, Audio Dispatcher, Looping Alarm,
and Twilio Emergency Phone Call Escalation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import os
import queue
import threading
import time
import cv2
import numpy as np

def get_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

from config import AlertSystemConfig, TwilioConfig
from database import AlertRecord
from logger import SurveillanceLogger

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertState(str, Enum):
    NORMAL = "NORMAL"
    SOFT_ALERT = "SOFT_ALERT"
    HARD_ALERT = "HARD_ALERT"
    RESOLVED = "RESOLVED"


@dataclass
class AlertCondition:
    """Represents the raw active state of a specific rule condition."""

    condition_key: str
    alert_type: str
    severity: AlertSeverity
    module_name: str
    message: str
    is_active: bool
    tracked_ids: List[int] = field(default_factory=list)
    zone_name: Optional[str] = field(default=None)
    details: Dict[str, Any] = field(default_factory=dict)
    sound_siren: bool = False
    voice_announcement: Optional[str] = None
    trigger_phone_call: bool = False


@dataclass
class ActiveAlertTracker:
    """Tracks state and timing for an active alert condition."""

    condition_key: str
    alert_type: str
    module_name: str
    zone_name: Optional[str]
    first_detected_at: datetime
    last_detected_at: datetime
    last_cleared_at: Optional[datetime] = None
    state: AlertState = AlertState.NORMAL
    db_alert_id: Optional[int] = None
    last_spoken_at: Optional[datetime] = None
    phone_call_dispatched: bool = False


class NonBlockingAudioDispatcher:
    """
    Dedicated managed background worker for Text-To-Speech (TTS), single-shot beeps,
    and continuous looping alarm sirens with strict lifecycle and shutdown controls.
    """

    def __init__(self, enabled: bool = True):
        import atexit
        self.enabled = enabled
        self.msg_queue: queue.Queue = queue.Queue(maxsize=10)
        self.looping_alarm = False
        self.running = True
        self.engine = None
        self._engine_lock = threading.Lock()
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="SurveillanceAudioThread"
        )
        self.worker_thread.start()
        # Register atexit handler to ensure immediate audio shutdown on program termination
        atexit.register(self.shutdown)

    def _worker_loop(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        if pyttsx3 and self.enabled:
            try:
                with self._engine_lock:
                    self.engine = pyttsx3.init()
                    self.engine.setProperty("rate", 160)
            except Exception as e:
                logger.warning(f"[AUDIO] Failed to initialize pyttsx3: {e}")
                self.engine = None

        last_loop_beep = 0.0

        while self.running:
            try:
                # 1. Process queued speech or single beeps
                try:
                    item = self.msg_queue.get(timeout=0.15)
                    if not self.running:
                        self.msg_queue.task_done()
                        break
                    action_type, payload = item

                    if action_type == "speak" and self.engine and self.enabled and self.running:
                        try:
                            with self._engine_lock:
                                if self.engine and self.running:
                                    self.engine.say(payload)
                                    self.engine.runAndWait()
                        except Exception as e:
                            logger.warning(f"[AUDIO] TTS speech error: {e}")
                    elif action_type == "beep" and self.enabled and self.running:
                        self._play_beep(1200, 300)

                    self.msg_queue.task_done()
                except queue.Empty:
                    pass

                if not self.running:
                    break

                # 2. Handle continuous looping alarm if active
                if self.looping_alarm and self.enabled and self.running:
                    now = time.time()
                    if now - last_loop_beep >= 0.4:
                        self._play_beep(1400, 250)
                        last_loop_beep = time.time()

            except Exception as e:
                logger.warning(f"[AUDIO] Audio worker exception: {e}")

    def _play_beep(self, freq: int, duration_ms: int):
        """Play hardware tone safely on Windows without raising."""
        if not self.running or not self.enabled:
            return
        try:
            import winsound
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass

    def speak(self, text: str):
        """Enqueue speech text asynchronously."""
        if not self.enabled or not self.running or not text:
            return
        try:
            self.msg_queue.put_nowait(("speak", text))
        except queue.Full:
            pass

    def sound_alarm(self):
        """Enqueue a single alarm buzzer."""
        if not self.enabled or not self.running:
            return
        try:
            self.msg_queue.put_nowait(("beep", None))
        except queue.Full:
            pass

    def start_looping_alarm(self):
        """Start continuous looping alarm sound."""
        if self.running:
            self.looping_alarm = True

    def stop_looping_alarm(self):
        """Stop continuous looping alarm sound."""
        self.looping_alarm = False
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def shutdown(self):
        """Immediately stop any looping alarms, purge audio queue, stop engine, and signal worker thread to exit."""
        self.looping_alarm = False
        self.running = False
        self.enabled = False

        # Drain message queue
        try:
            while not self.msg_queue.empty():
                try:
                    self.msg_queue.get_nowait()
                    self.msg_queue.task_done()
                except Exception:
                    break
        except Exception:
            pass

        # Stop TTS engine if running
        try:
            with self._engine_lock:
                if self.engine:
                    try:
                        self.engine.stop()
                    except Exception:
                        pass
        except Exception:
            pass

        # Purge Windows sound buffer
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        # If pygame is present, stop mixer
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
        except Exception:
            pass


class AlertManager:
    """
    Coordinates alert debouncing, state machine progression, escalation,
    looping audible alarms, Twilio phone calls, and database persistence.
    """

    def __init__(
        self,
        logger_instance: SurveillanceLogger,
        config: Optional[AlertSystemConfig] = None,
        twilio_config: Optional[TwilioConfig] = None,
        tts_enabled: bool = True,
    ):
        self.surveillance_logger = logger_instance
        self.config = config or AlertSystemConfig()
        self.twilio_config = twilio_config or TwilioConfig()
        self.audio = NonBlockingAudioDispatcher(enabled=tts_enabled)
        self.active_trackers: Dict[str, ActiveAlertTracker] = {}
        self.active_alerts_display: List[Dict[str, Any]] = []

    def dispatch_twilio_phone_call(self, alert_type: str, message: str, zone_name: Optional[str] = None):
        """
        Place automated emergency escalation phone call via Twilio Voice API.
        Fails gracefully with clear logging if credentials or numbers are missing/invalid.
        """
        if not self.twilio_config.enabled:
            return False

        account_sid = self.twilio_config.account_sid.strip()
        auth_token = self.twilio_config.auth_token.strip()
        from_number = self.twilio_config.from_number.strip()
        to_number = self.twilio_config.to_number.strip()

        if not account_sid or not auth_token or not from_number or not to_number:
            logger.info(
                f"[TWILIO ESCALATION] Automated phone call skipped: Twilio credentials/phone numbers "
                f"not fully configured in config.yaml / .env. Alert '{alert_type}' recorded in DB."
            )
            return False

        if not TwilioClient:
            logger.warning("[TWILIO] 'twilio' Python package not installed. Skipping phone call.")
            return False

        try:
            client = TwilioClient(account_sid, auth_token)
            twiml_content = (
                f"<Response><Say voice='alice' loop='2'>"
                f"Emergency security alert from Smart Surveillance System. "
                f"A critical {alert_type.replace('_', ' ')} incident has occurred in {zone_name or 'the monitored zone'}. "
                f"Immediate response is required."
                f"</Say></Response>"
            )

            call = client.calls.create(
                twiml=twiml_content,
                to=to_number,
                from_=from_number,
            )
            logger.critical(
                f"[TWILIO DISPATCHED] Emergency phone escalation call placed successfully to {to_number}! "
                f"Call SID: {call.sid}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"[TWILIO ERROR] Failed to place Twilio emergency call to {to_number}: {e}. "
                f"Surveillance pipeline continues uninterrupted."
            )
            return False

    def evaluate_condition(
        self,
        condition: AlertCondition,
        frame: Optional[np.ndarray] = None,
        debounce_seconds: float = 1.5,
    ):
        """
        Evaluate an incoming rule condition, applying debouncing, escalation,
        looping audible siren, phone escalation, and auto-resolution.
        """
        now = get_utc_now()
        key = condition.condition_key
        tracker = self.active_trackers.get(key)

        if condition.is_active:
            if tracker is None:
                tracker = ActiveAlertTracker(
                    condition_key=key,
                    alert_type=condition.alert_type,
                    module_name=condition.module_name,
                    zone_name=condition.zone_name,
                    first_detected_at=now,
                    last_detected_at=now,
                    state=AlertState.NORMAL,
                )
                self.active_trackers[key] = tracker

            tracker.last_detected_at = now
            tracker.last_cleared_at = None
            active_duration = (now - tracker.first_detected_at).total_seconds()

            # Debounce check before triggering soft alert
            if active_duration >= debounce_seconds and tracker.state == AlertState.NORMAL:
                tracker.state = AlertState.SOFT_ALERT
                logger.warning(
                    f"[ALERT TRIGGERED] {condition.alert_type} ({condition.severity.value}) "
                    f"Message: {condition.message}"
                )

                # Persist to database and screenshot
                record = self.surveillance_logger.log_alert(
                    alert_type=condition.alert_type,
                    severity=condition.severity.value,
                    module_name=condition.module_name,
                    zone_name=condition.zone_name,
                    message=condition.message,
                    tracked_ids=condition.tracked_ids,
                    frame=frame,
                    details=condition.details,
                )
                tracker.db_alert_id = record.id if record else None

                # Speech / Siren dispatch
                if condition.voice_announcement:
                    self.audio.speak(condition.voice_announcement)
                    tracker.last_spoken_at = now
                if condition.sound_siren:
                    self.audio.sound_alarm()

                # If condition severity is CRITICAL or explicit phone call requested on trigger
                if condition.severity == AlertSeverity.CRITICAL or condition.trigger_phone_call:
                    if not tracker.phone_call_dispatched:
                        self.dispatch_twilio_phone_call(
                            condition.alert_type, condition.message, condition.zone_name
                        )
                        tracker.phone_call_dispatched = True
                    if condition.sound_siren:
                        self.audio.start_looping_alarm()

            # Escalation check: Soft Alert -> Hard Alert
            elif tracker.state == AlertState.SOFT_ALERT:
                if active_duration >= (
                    debounce_seconds + self.config.escalation_timeout_sec
                ):
                    tracker.state = AlertState.HARD_ALERT
                    logger.critical(
                        f"[ALERT ESCALATED] {condition.alert_type} ESCALATED TO HARD ALERT."
                    )
                    if tracker.db_alert_id:
                        self.surveillance_logger.resolve_alert(
                            tracker.db_alert_id, resolution_status="ESCALATED"
                        )

                    # Start continuous looping alarm sound for Hard Alert
                    self.audio.start_looping_alarm()

                    # Trigger Twilio phone escalation on Hard Alert
                    if not tracker.phone_call_dispatched:
                        self.dispatch_twilio_phone_call(
                            condition.alert_type, condition.message, condition.zone_name
                        )
                        tracker.phone_call_dispatched = True

            # Periodic speech reminder for persistent soft alert (every 6 seconds)
            if (
                tracker.state in (AlertState.SOFT_ALERT, AlertState.HARD_ALERT)
                and condition.voice_announcement
                and tracker.last_spoken_at
                and (now - tracker.last_spoken_at).total_seconds() > 6.0
            ):
                self.audio.speak(condition.voice_announcement)
                tracker.last_spoken_at = now

        else:
            # Condition is NOT active
            if tracker is not None:
                if tracker.last_cleared_at is None:
                    tracker.last_cleared_at = now

                clear_duration = (now - tracker.last_cleared_at).total_seconds()

                if clear_duration >= self.config.auto_resolve_debounce_sec:
                    if tracker.state in (AlertState.SOFT_ALERT, AlertState.HARD_ALERT):
                        logger.info(
                            f"[ALERT RESOLVED] Condition cleared: {condition.alert_type}"
                        )
                        if tracker.db_alert_id:
                            self.surveillance_logger.resolve_alert(
                                tracker.db_alert_id, resolution_status="RESOLVED"
                            )

                    # Stop looping alarm when condition is resolved
                    self.audio.stop_looping_alarm()

                    # Clean up tracker
                    del self.active_trackers[key]

    def get_current_active_alerts(self) -> List[Dict[str, Any]]:
        """Returns list of currently active triggered alerts for GUI overlay."""
        active = []
        for key, tracker in list(self.active_trackers.items()):
            if tracker.state in (AlertState.SOFT_ALERT, AlertState.HARD_ALERT):
                active.append(
                    {
                        "key": key,
                        "alert_type": tracker.alert_type,
                        "module": tracker.module_name,
                        "zone": tracker.zone_name,
                        "state": tracker.state.value,
                        "duration_sec": round(
                            (get_utc_now() - tracker.first_detected_at).total_seconds(),
                            1,
                        ),
                        "db_id": tracker.db_alert_id,
                    }
                )
        return active
