"""
Configuration Loader and Schema Manager.
Parses config.yaml and provides type-safe access across ATM and Restricted Zone modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SystemConfig:
    app_name: str = "Smart Video Surveillance & Analytics System"
    version: str = "1.0.0"
    log_level: str = "INFO"
    device: str = "auto"


@dataclass
class DisplayWindowConfig:
    enabled: bool = True
    title: str = "Smart Surveillance Engine"
    width: int = 960
    height: int = 540
    show_fps: bool = True
    show_hud: bool = True
    show_zones: bool = True


@dataclass
class VideoSourcesConfig:
    atm: str = "0"
    restricted: str = "sample_videos/pedestrians_surveillance.avi"


@dataclass
class VideoConfig:
    default_source: str = "0"
    sources: VideoSourcesConfig = field(default_factory=VideoSourcesConfig)
    frame_width: int = 960
    frame_height: int = 540
    target_fps: int = 30
    frame_skip: int = 1
    display_window: DisplayWindowConfig = field(default_factory=DisplayWindowConfig)


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    tracker_type: str = "bytetrack.yaml"
    target_classes: List[int] = field(default_factory=lambda: [0])


@dataclass
class ZoneDefinition:
    name: str
    color: List[int]
    polygon: List[List[int]]


@dataclass
class ATMModuleConfig:
    enabled: bool = True
    face_detection_confidence: float = 0.50
    face_covered_debounce_sec: float = 1.5
    multi_person_debounce_sec: float = 1.5
    blackout_variance_threshold: float = 12.0
    blackout_duration_sec: float = 20.0
    tts_enabled: bool = True
    sound_siren_on_tamper: bool = True
    looping_alarm_on_hard_alert: bool = True
    simulated_dispatch_enabled: bool = True
    authority_contact: str = "Central Security Control & Precinct 4"
    visibility_window_size: int = 30
    covered_threshold: float = 0.30
    safe_recovery_threshold: float = 0.70
    landmark_confidence_threshold: float = 0.50
    show_debug_visibility: bool = True


@dataclass
class TwilioConfig:
    enabled: bool = True
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    to_number: str = ""


@dataclass
class RestrictedModuleConfig:
    enabled: bool = True
    intrusion_debounce_sec: float = 0.5
    target_classes: List[int] = field(default_factory=lambda: [0])
    sound_siren_on_intrusion: bool = True
    tts_enabled: bool = True
    zone_id: str = "restricted_zone"


@dataclass
class AlertSystemConfig:
    escalation_timeout_sec: float = 15.0
    auto_resolve_debounce_sec: float = 2.0
    audio_announcements: bool = True


@dataclass
class PersistenceConfig:
    db_path: str = "logs/surveillance.db"
    screenshots_dir: str = "logs/screenshots"
    excel_export_path: str = "logs/surveillance_alerts.xlsx"
    auto_export_excel: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_stream_fps: int = 20


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    zones: Dict[str, ZoneDefinition] = field(default_factory=dict)
    atm_module: ATMModuleConfig = field(default_factory=ATMModuleConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    restricted_module: RestrictedModuleConfig = field(default_factory=RestrictedModuleConfig)
    alert_system: AlertSystemConfig = field(default_factory=AlertSystemConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and parse YAML configuration into type-safe AppConfig."""
    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config file '{config_path}' not found. Using default configurations.")
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sys_raw = raw.get("system", {})
    system = SystemConfig(
        app_name=sys_raw.get("app_name", "Smart Video Surveillance & Analytics System"),
        version=sys_raw.get("version", "1.0.0"),
        log_level=sys_raw.get("log_level", "INFO"),
        device=sys_raw.get("device", "auto"),
    )

    vid_raw = raw.get("video", {})
    disp_raw = vid_raw.get("display_window", {})
    display_win = DisplayWindowConfig(
        enabled=disp_raw.get("enabled", True),
        title=disp_raw.get("title", "Smart Surveillance Engine"),
        width=disp_raw.get("width", 960),
        height=disp_raw.get("height", 540),
        show_fps=disp_raw.get("show_fps", True),
        show_hud=disp_raw.get("show_hud", True),
        show_zones=disp_raw.get("show_zones", True),
    )
    srcs_raw = vid_raw.get("sources", {})
    sources = VideoSourcesConfig(
        atm=str(srcs_raw.get("atm", "0")),
        restricted=str(srcs_raw.get("restricted", "sample_videos/pedestrians_surveillance.avi")),
    )
    video = VideoConfig(
        default_source=str(vid_raw.get("default_source", "0")),
        sources=sources,
        frame_width=vid_raw.get("frame_width", 960),
        frame_height=vid_raw.get("frame_height", 540),
        target_fps=vid_raw.get("target_fps", 30),
        frame_skip=vid_raw.get("frame_skip", 1),
        display_window=display_win,
    )

    det_raw = raw.get("detection", {})
    detection = DetectionConfig(
        model_name=det_raw.get("model_name", "yolov8n.pt"),
        confidence_threshold=float(det_raw.get("confidence_threshold", 0.45)),
        iou_threshold=float(det_raw.get("iou_threshold", 0.45)),
        tracker_type=det_raw.get("tracker_type", "bytetrack.yaml"),
        target_classes=det_raw.get("target_classes", [0]),
    )

    zones_dict: Dict[str, ZoneDefinition] = {}
    for zone_id, z_data in raw.get("zones", {}).items():
        zones_dict[zone_id] = ZoneDefinition(
            name=z_data.get("name", zone_id),
            color=z_data.get("color", [0, 255, 0]),
            polygon=z_data.get("polygon", []),
        )

    atm_raw = raw.get("atm_module", {})
    atm_module = ATMModuleConfig(
        enabled=atm_raw.get("enabled", True),
        face_detection_confidence=float(atm_raw.get("face_detection_confidence", 0.50)),
        face_covered_debounce_sec=float(atm_raw.get("face_covered_debounce_sec", 1.5)),
        multi_person_debounce_sec=float(atm_raw.get("multi_person_debounce_sec", 1.5)),
        blackout_variance_threshold=float(atm_raw.get("blackout_variance_threshold", 12.0)),
        blackout_duration_sec=float(atm_raw.get("blackout_duration_sec", 20.0)),
        tts_enabled=atm_raw.get("tts_enabled", True),
        sound_siren_on_tamper=atm_raw.get("sound_siren_on_tamper", True),
        looping_alarm_on_hard_alert=atm_raw.get("looping_alarm_on_hard_alert", True),
        simulated_dispatch_enabled=atm_raw.get("simulated_dispatch_enabled", True),
        authority_contact=atm_raw.get("authority_contact", "Central Security Control & Precinct 4"),
        visibility_window_size=int(atm_raw.get("visibility_window_size", 30)),
        covered_threshold=float(atm_raw.get("covered_threshold", 0.30)),
        safe_recovery_threshold=float(atm_raw.get("safe_recovery_threshold", 0.70)),
        landmark_confidence_threshold=float(atm_raw.get("landmark_confidence_threshold", 0.50)),
        show_debug_visibility=bool(atm_raw.get("show_debug_visibility", True)),
    )

    twi_raw = raw.get("twilio", {})
    twilio_config = TwilioConfig(
        enabled=twi_raw.get("enabled", True),
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", twi_raw.get("account_sid", "")),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", twi_raw.get("auth_token", "")),
        from_number=os.getenv("TWILIO_FROM_NUMBER", twi_raw.get("from_number", "")),
        to_number=os.getenv("TWILIO_TO_NUMBER", twi_raw.get("to_number", "")),
    )

    res_raw = raw.get("restricted_module", {})
    restricted_module = RestrictedModuleConfig(
        enabled=res_raw.get("enabled", True),
        intrusion_debounce_sec=float(res_raw.get("intrusion_debounce_sec", 0.5)),
        target_classes=res_raw.get("target_classes", [0]),
        sound_siren_on_intrusion=res_raw.get("sound_siren_on_intrusion", True),
        tts_enabled=res_raw.get("tts_enabled", True),
        zone_id=res_raw.get("zone_id", "restricted_zone"),
    )

    alt_raw = raw.get("alert_system", {})
    alert_system = AlertSystemConfig(
        escalation_timeout_sec=float(alt_raw.get("escalation_timeout_sec", 15.0)),
        auto_resolve_debounce_sec=float(alt_raw.get("auto_resolve_debounce_sec", 2.0)),
        audio_announcements=alt_raw.get("audio_announcements", True),
    )

    per_raw = raw.get("persistence", {})
    persistence = PersistenceConfig(
        db_path=per_raw.get("db_path", "logs/surveillance.db"),
        screenshots_dir=per_raw.get("screenshots_dir", "logs/screenshots"),
        excel_export_path=per_raw.get("excel_export_path", "logs/surveillance_alerts.xlsx"),
        auto_export_excel=per_raw.get("auto_export_excel", True),
    )

    srv_raw = raw.get("server", {})
    server = ServerConfig(
        host=srv_raw.get("host", "0.0.0.0"),
        port=int(srv_raw.get("port", 8000)),
        websocket_stream_fps=int(srv_raw.get("websocket_stream_fps", 20)),
    )

    return AppConfig(
        system=system,
        video=video,
        detection=detection,
        zones=zones_dict,
        atm_module=atm_module,
        twilio=twilio_config,
        restricted_module=restricted_module,
        alert_system=alert_system,
        persistence=persistence,
        server=server,
    )


# Global instance
CONFIG = load_config()
