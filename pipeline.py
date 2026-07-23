import os
import cv2
import time
import logging
import config.config as config
from src.ingest import VideoStreamIngestion
from src.detection import YOLOV8Detector
from src.tracking import ObjectTracker
from src.event_rules import EventAnalysisEngine
from src.alert_manager import AlertManager
from src.utils import scale_polygon, draw_polygon_zone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SurveillancePipeline")

class SurveillancePipeline:
    def __init__(self, yolo_model_path=config.YOLO_MODEL_PATH):
        logger.info("Initializing Surveillance Pipeline components...")
        self.detector = YOLOV8Detector(model_name=yolo_model_path)
        self.tracker = ObjectTracker()
        self.event_engine = EventAnalysisEngine()
        self.alert_manager = AlertManager()

    def process_video(self, video_source, output_save_path=None, display_live=False):
        """
        Runs full video analytics pipeline on video_source.
        Optionally saves annotated video feed to output_save_path.
        Returns dictionary with processing stats and logged alerts list.
        """
        ingest = VideoStreamIngestion(video_source)
        w, h, fps = ingest.width, ingest.height, ingest.fps
        
        # Scale polygon zones to actual video dimensions
        atm_poly = scale_polygon(config.ATM_ZONE_NORMALIZED, w, h)
        restricted_poly = scale_polygon(config.RESTRICTED_ZONE_NORMALIZED, w, h)
        airport_poly = scale_polygon(config.ATM_ZONE_NORMALIZED, w, h)

        writer = None
        if output_save_path:
            os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_save_path, fourcc, fps, (w, h))

        start_time = time.time()
        processed_frames = 0
        total_alerts_count = 0

        logger.info(f"Processing video: {video_source}")
        
        for frame_idx, frame in ingest.read_frames():
            processed_frames += 1
            current_time_sec = frame_idx / fps

            # 1. Run YOLOv8 Detection
            detections = self.detector.detect(frame)

            # 2. Run DeepSORT Tracking
            tracked_objects = self.tracker.update(detections, frame)

            # 3. Event Analysis Engine Rules Evaluation
            alerts = self.event_engine.analyze_frame(
                frame, tracked_objects, atm_poly, restricted_poly, airport_poly, current_time_sec=current_time_sec
            )

            # 4. Visualization Overlays Creation
            annotated_frame = frame.copy()
            
            # Draw Monitoring Zones
            draw_polygon_zone(annotated_frame, atm_poly, "ATM / Monitoring Zone", color=(0, 200, 255), alpha=0.15)
            draw_polygon_zone(annotated_frame, restricted_poly, "RESTRICTED SECURITY ZONE", color=(0, 0, 255), alpha=0.20)

            # Count total person objects in current frame
            person_tracks = [obj for obj in tracked_objects if obj["class_name"] == "person"]
            num_persons_in_frame = len(person_tracks)

            # Extract track IDs flagged for face concealment in current frame
            concealed_track_ids = set()
            for a in alerts:
                if a["event_type"] == "FACE_CONCEALMENT":
                    tid = a.get("track_id")
                    if tid is not None:
                        if isinstance(tid, list):
                            for t in tid: concealed_track_ids.add(str(t))
                        else:
                            concealed_track_ids.add(str(tid))

            # Draw Tracked Bounding Boxes with Priority Colors:
            # RED (face concealed) > ORANGE (multiple people in frame) > GREEN (single person, face visible)
            for obj in tracked_objects:
                x1, y1, x2, y2 = [int(v) for v in obj["bbox_ltrb"]]
                track_id = str(obj["track_id"])
                cls_name = obj["class_name"]
                
                if cls_name == "person":
                    is_concealed = track_id in concealed_track_ids
                    if not is_concealed:
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, y2_c = min(w, x2), min(h, y2)
                        crop = frame[y1_c:y2_c, x1_c:x2_c]
                        if crop.size > 0:
                            is_concealed, _, _ = self.event_engine.face_detector.predict_concealment(crop)

                    if is_concealed:
                        box_color = (0, 0, 255)  # RED (BGR) - Face Concealed
                    elif num_persons_in_frame > 1:
                        box_color = (0, 140, 255)  # ORANGE (BGR) - Multiple People in Frame
                    else:
                        box_color = (0, 255, 0)  # GREEN (BGR) - Single Person, Face Visible
                else:
                    box_color = (255, 165, 0)  # Cyan/Blue for Non-Person Objects

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"ID #{track_id}: {cls_name}"
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 22)), (x1 + len(label)*10, y1), box_color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw Active Alert Banners if alerts in current frame
            if alerts:
                top_alert = alerts[0]
                banner_color = (0, 0, 220) if top_alert["severity"] == "CRITICAL" else (0, 140, 255)
                cv2.rectangle(annotated_frame, (0, 0), (w, 45), banner_color, -1)
                cv2.putText(annotated_frame, f"🚨 ALERT ({top_alert['severity']}): {top_alert['details']}", 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # 5. Alert Logging & Snapshot Management (Pass fully annotated frame containing bounding boxes & labels)
            logged_alerts = self.alert_manager.process_alerts(alerts, annotated_frame, frame_idx=frame_idx)
            total_alerts_count += len(logged_alerts)

            if writer:
                writer.write(annotated_frame)

        ingest.release()
        if writer:
            writer.release()

        elapsed = time.time() - start_time
        processing_fps = processed_frames / elapsed if elapsed > 0 else 0
        logger.info(f"Finished {video_source}: {processed_frames} frames in {elapsed:.2f}s ({processing_fps:.1f} FPS), {total_alerts_count} new alerts logged.")

        return {
            "source": video_source,
            "processed_frames": processed_frames,
            "elapsed_sec": elapsed,
            "fps": processing_fps,
            "total_alerts": total_alerts_count,
            "output_path": output_save_path,
            "alerts": self.alert_manager.get_all_alerts()
        }

if __name__ == "__main__":
    pipeline = SurveillancePipeline()
    res = pipeline.process_video(os.path.join(config.DEMO_CLIPS_DIR, "face_concealment.mp4"), 
                                 output_save_path=os.path.join(config.OUTPUT_DIR, "annotated_face_concealment.mp4"))
    print("Test run completed:", res)
