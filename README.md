# Smart Surveillance and Video Analytics System Using Deep Learning

**Team:** B.E. CSE (AI & ML), Chennai Institute of Technology  
**Project Scope:** ATM Security & Video Analytics (Real Footage Benchmark)  

---

## 📌 Executive Overview

An autonomous, deep learning-powered video surveillance and real-time analytics system designed for high-security environments including ATM cabins and airport terminals. The system replaces passive CCTV recording with proactive event detection, tracking, and instant alerts across four critical security modules:

1. **Face Concealment Detection**: Detects face masks, helmets, and face coverings inside monitored cabin zones.
2. **Restricted-Area Intrusion Detection**: Real-time virtual geofence perimeter protection with immediate alerting upon zone entry.
3. **Multiple Person Detection**: Detects shoulder-surfing and unauthorized multiple occupancy inside cabin zones.
4. **Single-Person Control Benchmark**: Evaluates single occupant footage to verify zero false-positive multi-person alerts.

---

## 🛠️ System Architecture & Folder Layout

```
adl project/
├── PLAN.md                     # Initial Project Scope & Time Estimates
├── PROJECT_REVIEW.md           # Comprehensive Technical Review & Methodology
├── README.md                   # Setup & Execution Instructions
├── requirements.txt            # Python Dependencies
├── app.py                      # Root Dashboard Runner Wrapper
├── pipeline.py                 # Core Video Analytics Pipeline Engine
├── test_all_scenarios.py       # Batch Benchmark Suite
├── config/
│   └── config.py               # Global System Configuration & Zone Polygons
├── data/
│   ├── demo_clips/             # The 4 Real Demo Videos
│   └── alerts/                 # CSV/JSON Logs & Snapshot Captures
├── models/
│   └── yolov8n.pt              # YOLOv8 Pretrained Model Weights
├── src/
│   ├── ingest.py               # Frame Extraction Module
│   ├── detection.py            # Clean YOLOv8 Detector
│   ├── tracking.py             # DeepSORT Object Tracker
│   ├── face_cover.py           # Face Concealment Classifier
│   ├── event_rules.py          # The 4 Module Event Rules
│   ├── alert_manager.py        # Alert Logger & Deduplicator
│   └── utils.py                # Geometry & Visual Overlay Utilities
├── dashboard/
│   └── app.py                  # Streamlit Interactive Dashboard
└── outputs/
    └── (Annotated Output Video Clips & Screenshots)
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.11
- OpenCV, PyTorch, Ultralytics YOLOv8, DeepSORT Realtime, Streamlit

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Real Video Benchmark Suite
To run all 4 real demo video clips through the detection, tracking, and rule engine pipeline:
```bash
python test_all_scenarios.py
```
Annotated video outputs are saved to `outputs/` and alert logs are exported to `data/alerts/alert_log.csv`.

### 4. Launch Interactive Central Dashboard
Start the Streamlit live surveillance dashboard:
```bash
streamlit run app.py
```
or
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Real Demo Clips Registry

| Clip Name | Description | Source & License |
|---|---|---|
| `data/demo_clips/face_concealment.mp4` | Person wearing face mask approaching counter | [Qengineering Repo](https://github.com/Qengineering/Face-Mask-Detection-Jetson-Nano/raw/main/Face_Mask_Video.mp4) (MIT License) |
| `data/demo_clips/restricted_intrusion.mp4` | Person walking into corridor zone | [Intel IoT DevKit](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4) (BSD/MIT License) |
| `data/demo_clips/multi_person.mp4` | Multiple people standing/walking facing camera | [Intel IoT DevKit](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4) (BSD/MIT License) |
| `data/demo_clips/single_person_control.mp4` | Single person standing/walking facing camera | [Intel IoT DevKit](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4) (BSD/MIT License) |

---

## 📄 License & Citation
Developed for Chennai Institute of Technology - Department of Computer Science & Engineering (AI & ML).
