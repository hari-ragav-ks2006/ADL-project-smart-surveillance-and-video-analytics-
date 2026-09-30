# Surveillance System Verification & Testing Report

This document records the verification methodology, threshold tuning calibrations, unit test results, and end-to-end scenario validations for the **Smart Video Surveillance and Analytics System**.

---

## 1. GUI Display & Window Rendering Verification

### Issue Addressed
The previous version of the application failed to open a visible window and printed only to the terminal.

### Root Cause Analysis & Fix Implemented
1. **Window Sizing & Lifecycle**: Created [display.py](file:///d:/Smsrt surveillance (ADL)/display.py) with dedicated `DisplayManager`. Uses `cv2.namedWindow('Smart Surveillance Engine - Live View', cv2.WINDOW_NORMAL)` with explicit sizing to `960x540` (half-screen).
2. **Guaranteed Event Loop**: Implemented a mandatory `cv2.waitKey(1) & 0xFF` event pump inside `render_and_wait()` called on every frame of the main thread.
3. **Graceful Fallback & Diagnostics**: On startup, the pipeline verifies stream availability and logs:
   ```
   [INFO] Display window active (960x540): 'Smart Surveillance Engine - Live View'
   [SOURCE] Active stream confirmed: resolution 960x540
   ```
   If no physical camera is connected, the engine gracefully offers fallback test videos or reports an actionable error without silent termination.

---

## 2. Threshold Calibration & Configuration

All thresholds are centralized in [config.yaml](file:///d:/Smsrt surveillance (ADL)/config.yaml) and calibrated as follows:

| Module / Parameter | Calibrated Value | Real-World Deployment Value | Rationale |
|---|---|---|---|
| **YOLOv8 Confidence** | `0.40` | `0.45` | Balances recall for partially occluded persons with low false positive rate. |
| **YOLOv8 IOU** | `0.45` | `0.45` | Standard NMS threshold for multi-person separation. |
| **Face Obscured Threshold** | `30% coverage` | `30%–35%` | Rolling 30-frame window MediaPipe nose & mouth visibility score. |
| **Face Obscured Debounce** | `0.5 seconds` | `0.5–1.0 seconds` | Rapid alerting on mask/helmet donning while avoiding single-frame noise. |
| **Multi-Person Debounce** | `0.5 seconds` | `1.0–1.5 seconds` | Reliable detection of multiple people approaching transaction area. |
| **Camera Blackout Variance** | `15.0` | `12.0–15.0` | Frames with Laplacian variance < 15 indicate uniform darkness or lens obstruction. |
| **Camera Blackout Duration** | `0.5 seconds` | `1.0–2.0 seconds` | Debounce window before escalating to critical siren alarm and emergency dispatch. |
| **Restricted Zone Breach** | `0.5 seconds` | `0.5–1.0 seconds` | Instant detection of person centroid entering restricted polygon coordinates. |
| **Alert Escalation Timeout** | `15.0 seconds` | `30.0 seconds` | Duration an unresolved soft alert takes before escalating to hard alert state. |

---

## 3. Automated Unit & Integration Tests

The test suite is located in `tests/` and can be executed with:
```powershell
pytest tests/ -v
```

### Test Results Summary:
```
tests/test_alert_system.py::test_alert_debouncing_and_escalation PASSED  [  7%]
tests/test_atm_rules.py::test_atm_multi_person_rule PASSED               [ 15%]
tests/test_atm_rules.py::test_atm_single_person_face_obscured PASSED     [ 23%]
tests/test_atm_rules.py::test_atm_blackout_tamper_rule PASSED            [ 30%]
tests/test_database.py::test_database_crud_and_excel PASSED              [ 38%]
tests/test_detection.py::test_tracked_object_geometry PASSED             [ 46%]
tests/test_detection.py::test_detector_initialization PASSED             [ 53%]
tests/test_detection.py::test_detector_track_dummy_frame PASSED          [ 61%]
tests/test_mode_isolation.py::test_mode_configurations PASSED            [ 69%]
tests/test_mode_isolation.py::test_mode_isolation_switch PASSED          [ 76%]
tests/test_restricted_rules.py::test_restricted_zone_rule PASSED         [ 84%]
tests/test_zone_engine.py::test_polygon_zone_point_containment PASSED    [ 92%]
tests/test_zone_engine.py::test_zone_engine_mapping PASSED               [100%]

============================= 13 passed in 4.54s ==============================
```

---

## 4. End-to-End Scenario Verification Matrix

Run the automated scenario test script:
```powershell
python verify_e2e.py
```

### Scenario Matrix:

| Scenario | Input Clip | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| **ATM Bare Face (Safe)** | `sample_videos/atm_single_safe.mp4` | Detects person with bare face (>70% landmark visibility); no alerts triggered. | 0 alerts; safe operational status maintained. | **PASS** |
| **ATM Face: Mask** | `sample_videos/atm_face_obscured.mp4` | Detects person with surgical mask (<30% visibility); triggers `FACE_COVERED`. | `FACE_COVERED [MEDIUM]` logged to SQLite & screenshot saved. | **PASS** |
| **ATM Face: Hand Covered** | `sample_videos/atm_hand_covering.mp4` | Detects hand covering mouth/nose; triggers `FACE_COVERED`. | `FACE_COVERED [MEDIUM]` logged to SQLite & screenshot saved. | **PASS** |
| **ATM Face: Kerchief** | `sample_videos/atm_kerchief.mp4` | Detects kerchief/bandana covering nose+mouth; triggers `FACE_COVERED`. | `FACE_COVERED [MEDIUM]` logged to SQLite & screenshot saved. | **PASS** |
| **ATM Face: Helmet** | `sample_videos/atm_helmet.mp4` | Detects motorcycle helmet / visor; triggers `FACE_COVERED`. | `FACE_COVERED [MEDIUM]` logged to SQLite & screenshot saved. | **PASS** |
| **ATM Multi-Person** | `sample_videos/atm_multi_person.mp4` | Detects 2 people at ATM; triggers `MULTI_PERSON` alert. | `MULTI_PERSON [MEDIUM]` logged to SQLite & screenshot saved. | **PASS** |
| **ATM Camera Blackout** | `sample_videos/camera_blackout.mp4` | Frame variance drops to ~0; triggers red `CAMERA_BLACKOUT` hard alert & authority dispatch. | `CAMERA_BLACKOUT [CRITICAL]` logged to SQLite & dispatch confirmed. | **PASS** |
| **Restricted Zone Intrusion** | `sample_videos/pedestrians_surveillance.avi` | Pedestrian enters marked North Access Road polygon; triggers red box + warning + audio alarm. | `ZONE_INTRUSION [CRITICAL]` logged to SQLite & screenshot saved. | **PASS** |

---

## 5. Persistence & Output Verification

1. **SQLite Database**: Verified in `logs/surveillance.db` via SQLAlchemy. All fields (`id`, `timestamp`, `alert_type`, `severity`, `tracked_ids`, `screenshot_path`, `resolution_status`) populated correctly.
2. **Excel Mirror**: Verified in `logs/surveillance_alerts.xlsx` via `openpyxl`. Header row styled with dark theme and auto-sized column widths.
3. **Evidence Photos**: High-resolution annotated JPEG files verified in `logs/screenshots/ALERT_*.jpg`.
