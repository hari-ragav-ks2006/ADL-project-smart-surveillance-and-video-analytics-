"""
Restricted Zone & Doorway Security Rule Module (Isolated Plug-in).
Enforces security perimeter and doorway access control:
1. Spatial polygon zone containment detection.
2. Intruder classification and tracking within protected doorway / restricted rooms.
3. Automated audible alarms, Text-To-Speech warnings, and intrusion alert logging.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import cv2
import numpy as np

from alert_system import AlertCondition, AlertSeverity
from config import RestrictedModuleConfig
from detection_model import TrackedObject

logger = logging.getLogger(__name__)


def get_utc_now() -> datetime:
    """Return timezone-naive UTC timestamp for consistent SQLite compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class RestrictedStatus:
    """Consolidated state evaluation for restricted zone security overlay."""

    is_breached: bool = False
    intruder_count: int = 0
    intruder_ids: List[int] = field(default_factory=list)
    status_label: str = "Restricted Area Secure — Perimeter Clear"
    active_alerts: List[str] = field(default_factory=list)


class RestrictedZoneRuleModule:
    """
    Restricted Zone & Perimeter Breach Intelligence Module.
    Completely decoupled from ATM security surveillance module.
    """

    def __init__(self, config: Optional[RestrictedModuleConfig] = None):
        self.config = config or RestrictedModuleConfig()

    def evaluate_frame(
        self,
        frame: np.ndarray,
        zone_objects: List[TrackedObject],
        zone_name: str = "Restricted Security Zone",
    ) -> Tuple[RestrictedStatus, List[AlertCondition]]:
        """
        Evaluate perimeter intrusion rules on objects currently located inside the zone.

        Args:
            frame: Video frame image.
            zone_objects: Objects currently contained within the designated restricted polygon.
            zone_name: Display name of the monitored zone.

        Returns:
            (RestrictedStatus, List[AlertCondition])
        """
        conditions: List[AlertCondition] = []

        # Filter objects by target classes (default class 0 = person)
        intruders = [
            obj for obj in zone_objects if obj.class_id in self.config.target_classes
        ]
        intruder_count = len(intruders)
        intruder_ids = [obj.track_id for obj in intruders]

        is_breached = intruder_count > 0

        status = RestrictedStatus(
            is_breached=is_breached,
            intruder_count=intruder_count,
            intruder_ids=intruder_ids,
            status_label=(
                "Unauthorized entry detected in restricted area — please check"
                if is_breached
                else "Restricted Area Secure — Perimeter Clear"
            ),
            active_alerts=[f"Intrusion #{tid}" for tid in intruder_ids] if is_breached else [],
        )

        if is_breached:
            conditions.append(
                AlertCondition(
                    condition_key="restricted_zone_intrusion",
                    alert_type="ZONE_INTRUSION",
                    severity=AlertSeverity.CRITICAL,
                    module_name="restricted_zone_rules",
                    message="Unauthorized entry detected in restricted area — please check",
                    is_active=True,
                    tracked_ids=intruder_ids,
                    zone_name=zone_name,
                    sound_siren=self.config.sound_siren_on_intrusion,
                    voice_announcement=(
                        "Unauthorized entry detected in restricted area — please check"
                        if self.config.tts_enabled
                        else None
                    ),
                    details={
                        "intruder_count": intruder_count,
                        "intruder_ids": intruder_ids,
                        "zone_name": zone_name,
                    },
                )
            )
        else:
            # Generate inactive condition so alert manager can auto-resolve after debounce
            conditions.append(
                AlertCondition(
                    condition_key="restricted_zone_intrusion",
                    alert_type="ZONE_INTRUSION",
                    severity=AlertSeverity.CRITICAL,
                    module_name="restricted_zone_rules",
                    message="Perimeter clear",
                    is_active=False,
                    zone_name=zone_name,
                )
            )

        return status, conditions
