import os
import time
import config.config as config
from pipeline import SurveillancePipeline

def run_all_tests():
    print("==================================================================")
    print("   SMART SURVEILLANCE PIPELINE - REAL VIDEO DEMO BENCHMARK")
    print("==================================================================")
    
    pipeline = SurveillancePipeline()

    test_cases = [
        ("1. Face Concealment Detection", "face_concealment.mp4", "annotated_face_concealment.mp4"),
        ("2. Restricted-Area Intrusion Detection", "restricted_intrusion.mp4", "annotated_restricted_intrusion.mp4"),
        ("3. Multiple-Person Detection (ATM)", "multi_person.mp4", "annotated_multi_person.mp4"),
        ("4. Single-Person Control (No Alert Expected)", "single_person_control.mp4", "annotated_single_person_control.mp4")
    ]

    summary_results = []

    for title, input_filename, output_filename in test_cases:
        input_path = os.path.join(config.DEMO_CLIPS_DIR, input_filename)
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        
        print(f"\n[RUNNING TEST]: {title}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        res = pipeline.process_video(input_path, output_save_path=output_path)
        summary_results.append({
            "title": title,
            "frames": res["processed_frames"],
            "fps": res["fps"],
            "alerts": res["total_alerts"]
        })

    print("\n==================================================================")
    print("                    REAL VIDEO BENCHMARK SUMMARY RESULTS           ")
    print("==================================================================")
    for s in summary_results:
        print(f" - {s['title']}: {s['frames']} frames @ {s['fps']:.1f} FPS | Total Alerts Triggered: {s['alerts']}")
    print("==================================================================\n")

if __name__ == "__main__":
    run_all_tests()
