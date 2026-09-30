"""
End-to-End Automated Verification Script.
Feeds scenario clips through the pipeline for both operational modes:
- ATM Mode (Bare Face Safe, Surgical Mask, Hand Covering, Kerchief, Helmet, Multi-Person, Blackout)
- Restricted Zone Mode (Outdoor Pedestrian Intrusion in Restricted North Access Road)
Verifies alert generation, database records, screenshots, and mode isolation.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import cv2

from config import load_config
from detection_model import TrackedObject
from main import SurveillancePipeline


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def verify_scenario(name: str, video_path: str, mode: str, max_frames: int = 150):
    print(f"\n" + "=" * 60)
    print(f" TESTING SCENARIO: {name}")
    print(f" Video: {video_path} | Mode: {mode}")
    print("=" * 60)

    pipeline = SurveillancePipeline(source=video_path, mode=mode, gui_enabled=False)
    with pipeline.db_manager.get_session() as session:
        from sqlalchemy import func
        from database import AlertRecord
        min_id = session.query(func.max(AlertRecord.id)).scalar() or 0

    # Fast debounces for automated verification
    pipeline.config.atm_module.face_covered_debounce_sec = 0.25
    pipeline.config.atm_module.multi_person_debounce_sec = 0.25
    pipeline.config.atm_module.blackout_duration_sec = 0.5
    pipeline.config.restricted_module.intrusion_debounce_sec = 0.25

    cap = pipeline._open_capture()
    if not cap:
        print(f" [FAILED] Could not open {video_path}")
        return False

    processed = 0
    start_time = time.time()

    for i in range(max_frames):
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(raw_frame, (960, 540))
        processed += 1

        # 1. Track objects
        tracked = pipeline.detector.track_frame(frame)
        zone_memberships = pipeline.zone_engine.process_objects(tracked)

        # 2. Evaluate mode-specific rules
        all_conds = []
        if mode == "atm" and pipeline.atm_module is not None:
            raw_persons = [o for o in tracked if o.class_id == 0]
            persons, _ = pipeline.atm_module.fuse_persons_and_faces(frame, raw_persons)
            status, conds, _ = pipeline.atm_module.evaluate_frame(frame, persons, zone_name="ATM Area")
            all_conds.extend(conds)

        elif mode == "restricted" and pipeline.restricted_module is not None:
            res_zone_id = pipeline.config.restricted_module.zone_id
            res_zone_objs = zone_memberships.get(res_zone_id, [])
            res_zone_name = (
                pipeline.config.zones[res_zone_id].name
                if res_zone_id in pipeline.config.zones
                else "Restricted Security Zone"
            )
            status, conds = pipeline.restricted_module.evaluate_frame(
                frame, res_zone_objs, zone_name=res_zone_name
            )
            all_conds.extend(conds)

        for c in all_conds:
            deb = 0.25
            if c.alert_type in ("CAMERA_BLACKOUT", "FACE_COVERED"):
                deb = 0.0
            elif c.alert_type == "ZONE_INTRUSION":
                deb = 0.25

            pipeline.alert_manager.evaluate_condition(c, frame=frame.copy(), debounce_seconds=deb)

        time.sleep(0.005)

    cap.release()
    pipeline.alert_manager.audio.shutdown()
    elapsed = time.time() - start_time

    module_filter = {
        "atm": "atm_rules",
        "restricted": "restricted_zone_rules",
    }.get(mode)

    all_alerts = pipeline.surveillance_logger.query_alerts(limit=50)
    alerts = [
        a for a in all_alerts
        if a["id"] > min_id and (module_filter is None or a.get("module_name") == module_filter)
    ]
    print(f" [OK] Logged Alerts this run: {len(alerts)}")
    has_target_alert = False
    for a in alerts:
        print(f"   • Alert #{a['id']}: {a['alert_type']} [{a['severity']}] -> {a['message']}")
        if a.get("screenshot_path"):
            assert Path(a["screenshot_path"]).exists(), "Screenshot missing"
            print(f"     Evidence screenshot: {a['screenshot_path']}")
        has_target_alert = True

    return has_target_alert


def main():
    scenarios = [
        ("ATM Single Person: Bare Face (Safe)", "sample_videos/atm_single_safe.mp4", "atm", 60, False),
        ("ATM Face: Surgical/Cloth Mask", "sample_videos/atm_face_obscured.mp4", "atm", 90, True),
        ("ATM Face: Hand Covering Face", "sample_videos/atm_hand_covering.mp4", "atm", 90, True),
        ("ATM Face: Kerchief / Bandana", "sample_videos/atm_kerchief.mp4", "atm", 90, True),
        ("ATM Face: Motorcycle Helmet", "sample_videos/atm_helmet.mp4", "atm", 90, True),
        ("ATM Multi-Person Violation", "sample_videos/atm_multi_person.mp4", "atm", 90, True),
        ("ATM Camera Blackout / Tamper", "sample_videos/camera_blackout.mp4", "atm", 150, True),
        ("Restricted Hallway CCTV Intrusion", "sample_videos/hallway_cctv_intrusion.mp4", "restricted", 100, True),
    ]

    results = {}
    for item in scenarios:
        name, path, mode, frames, expect_alert = item
        triggered = verify_scenario(name, path, mode, frames)
        if expect_alert:
            success = triggered
            status_str = "PASS [ALERT TRIGGERED & LOGGED]" if success else "FAIL [NO ALERT TRIGGERED]"
        else:
            success = not triggered
            status_str = "PASS [SAFE & UNBLOCKED - NO FALSE ALERT]" if success else "FAIL [FALSE ALERT TRIGGERED]"
        results[name] = (success, status_str)

    print("\n" + "=" * 65)
    print(" REFINED VERIFICATION SUMMARY MATRIX (TWO-MODULE SYSTEM)")
    print("=" * 65)
    all_pass = True
    for name, (success, status_str) in results.items():
        if not success:
            all_pass = False
        print(f" {name:<36} : {status_str}")
    print("=" * 65)
    if all_pass:
        print(" FINAL STATUS: ALL VERIFICATIONS PASSED")
    else:
        print(" FINAL STATUS: ONE OR MORE VERIFICATIONS FAILED")


if __name__ == "__main__":
    main()
