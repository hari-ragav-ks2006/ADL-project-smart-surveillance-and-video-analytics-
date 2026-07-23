# Smart Surveillance and Video Analytics System - Implementation Plan

## 1. Project Understanding & Module Overview

The objective of this project is to build an automated, deep learning-driven real-time video surveillance and analytics system tailored for ATM cabins and airport environments. Traditional CCTV systems only passively record footage, requiring continuous manual monitoring. Our system automates threat detection across four primary modules:

1. **Face Cover / Concealment Detection (ATM Cabin)**:
   Detects whether individuals inside an ATM cabin have covered their faces (e.g., masks, helmets, scarves, hoodies) to prevent fraudulent activities and unmasked identity evasion.
2. **Multiple-Person Detection (ATM Cabin)**:
   Monitors the designated ATM interior zone. If more than one person is detected inside the ATM boundary, a multi-person violation alert is raised to prevent shoulder surfing or physical robbery.
3. **Restricted-Area Intrusion Detection (Airport/Security Zones)**:
   Monitors configurable virtual boundary polygons (geofence). Triggers immediate intrusion alerts when any unauthorized human presence is detected inside the restricted region.
4. **Abandoned / Unattended Baggage Detection (Airport Terminals)**:
   Tracks baggage items (backpacks, suitcases, handbags) and people. If a baggage item remains stationary in a monitored area without an associated person within a spatial distance threshold $D_{\text{near}}$ for longer than a temporal duration $T_{\text{abandon}}$ (e.g., > 5–10 seconds), an abandoned baggage alert is triggered.

## 2. Technology Stack & Rationale

- **Programming Language**: Python 3.11 (standard for computer vision & ML pipelines)
- **Object Detection**: `ultralytics` YOLOv8 (YOLOv8n / YOLOv8s pretrained on COCO dataset). Provides real-time inference speed on CPU with pre-trained classes: `person` (0), `backpack` (24), `handbag` (26), `suitcase` (28).
- **Multi-Object Tracking**: `deep-sort-realtime` (DeepSORT algorithm). Maintains persistent tracking IDs (`track_id`) across consecutive video frames to track movement, stationarity, and owner proximity.
- **Face Concealment Detector**: Fine-tuned lightweight YOLOv8 classification / detection head or dedicated face-mask/cover detection pipeline operating on cropped person ROI inside ATM zone.
- **Spatial / Geometry Calculations**: `shapely` & OpenCV `cv2.pointPolygonTest` for polygon geofencing, centroid tracking, and bounding box IoU / distance calculation.
- **Dashboard & User Interface**: `streamlit` for interactive live video feed monitoring, real-time alert logs, metric counters, snapshot previews, and dynamic zone configuration.
- **Data & Video Handling**: OpenCV (`cv2`) for frame ingestion, preprocessing, drawing annotations (bounding boxes, track IDs, ROI zones, alert banners), and video output rendering.

## 3. Datasets & Pretrained Weights

- **General Detection & Tracking (Person & Baggage)**: Pretrained `yolov8n.pt` on COCO 80-class dataset (free, open source).
- **Face Cover / Mask Dataset**: Roboflow Universe / Kaggle Face Mask Detection Dataset (open-source MIT/Apache-licensed dataset containing masked vs. unmasked face images). Fine-tuned lightweight YOLOv8 head or pre-trained classification model. Note: Mask/face covering serves as a stand-in for face concealment for demo purposes.
- **Demo CCTV Footage**: 
  - Real-world stock footage / synthesized video feeds simulating ATM cabin (single/multiple person enter, face cover), restricted area (perimeter crossing), and airport baggage area (person drops suitcase and walks away).
  - All test feeds will be processed and exported with visual overlays as verification artifacts.

## 4. Step-by-Step Build Order & Time Estimates

| Step | Task | Description | Time Est. |
|---|---|---|---|
| 1 | Infrastructure & Dependencies | Setup virtual environment, install packages (`ultralytics`, `deep-sort-realtime`, `streamlit`, `shapely`, `opencv-python`). | 20 mins |
| 2 | Core Pipeline (Ingestion + YOLOv8 + DeepSORT) | Build frame ingestion, YOLOv8 detection, and DeepSORT tracking module returning persistent object tracks (`track_id`, `class`, `bbox`). | 45 mins |
| 3 | Event Analysis Engine - Rule 2 & Rule 3 | Implement Multiple-Person ATM Detection and Restricted-Area Intrusion modules using polygon geofencing (`shapely` / `cv2`). | 40 mins |
| 4 | Event Analysis Engine - Rule 4 | Implement Abandoned Baggage module with stationarity logic, person-bag proximity distance check, and temporal counter ($T > N$ sec). | 50 mins |
| 5 | Event Analysis Engine - Rule 1 | Implement Face Concealment module on cropped person bounding boxes in ATM zones using fine-tuned/pretrained classifier. | 50 mins |
| 6 | Alert Manager & Central Log | Build alert dispatcher, snapshot saver, CSV/JSON logger, and alert deduplication engine. | 30 mins |
| 7 | Streamlit Central Dashboard | Create dashboard with video player, live detection feed, alert stream, zone tuner, and stats counters. | 60 mins |
| 8 | Video Testing & Artifact Generation | Run full pipeline on test clips for each scenario, render annotated video outputs, and capture snapshots. | 40 mins |
| 9 | Documentation (`PROJECT_REVIEW.md`) | Prepare detailed documentation covering architecture, methodology, results, and limitations. | 45 mins |

**Total Estimated Time**: ~6.5 hours

## 5. Definition of "80–90% Complete"

For this project, "80–90% complete" is defined as:
1. End-to-end operational pipeline ingesting video feeds, running YOLOv8 + DeepSORT, analyzing events, and displaying results in Streamlit dashboard.
2. All 4 core detection modules implemented and functionally verified:
   - Multiple person in ATM cabin alert triggers when person count > 1 in ATM polygon.
   - Restricted area intrusion alert triggers immediately upon human entry into restricted zone.
   - Abandoned baggage alert triggers when a bag is left unattended without an owner nearby for > 5 seconds.
   - Face cover alert triggers when face concealment is detected on a person inside the ATM cabin.
3. Centralized dashboard featuring live video rendering, real-time alert logs with timestamps and snapshot previews, and configurable parameters.
4. Annotated video outputs generated and saved for all 4 scenarios as empirical evidence.
5. Complete documentation in `PLAN.md` and `PROJECT_REVIEW.md`.

## 6. Fallback Priority Order (If Time Constrained)

1. Working detection + tracking pipeline running end-to-end on sample video.
2. Restricted-area intrusion + Multiple-person in ATM zone modules (Rule-based).
3. Abandoned-baggage detection module (Stationarity + Proximity).
4. Face-cover detection module (Classifier/detector on face ROI).
5. Streamlit Dashboard polish, live alert feed, and parameter controls.
6. Fine-tuning YOLOv8 on custom ATM footage (Stretch goal).
