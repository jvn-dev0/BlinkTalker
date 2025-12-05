# module1_eye_tracking.py
# Module 1 - Webcam Eye Tracking + EAR Calculation
# Requirements: opencv-python, mediapipe, numpy

import cv2
import mediapipe as mp
import math
import argparse

# ----------------- Helper functions -----------------
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_EAR(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))
    vertical_1 = euclidean_distance(pts[1], pts[5])
    vertical_2 = euclidean_distance(pts[2], pts[4])
    horizontal = euclidean_distance(pts[0], pts[3])
    if horizontal == 0:
        return 0.0
    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return EAR

# ----------------- Main -----------------
def main(cam_index: int = 0, ear_threshold: float = 0.21):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try changing cam_index.")
        return

    print("Press 'q' to quit. Press 'c' to recalibrate threshold (see README).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_EAR = calculate_EAR(landmarks, LEFT_EYE, w, h)
            right_EAR = calculate_EAR(landmarks, RIGHT_EYE, w, h)
            EAR = (left_EAR + right_EAR) / 2.0

            # Draw EAR and open/closed status
            cv2.putText(frame, f"EAR: {EAR:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            status = "OPEN" if EAR >= ear_threshold else "CLOSED"
            color = (0, 255, 0) if status == "OPEN" else (0, 0, 255)
            cv2.putText(frame, f"EYES: {status}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # (Optional) draw small circles on eye landmark points
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)

        cv2.imshow("BlinkTalker - Module 1 (Eye Tracking)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # allow quick recalibration aid (prints EAR)
        if key == ord('c'):
            print("** Calibration: look with eyes open and press any key **")
            # capture a few frames to estimate open-eye EAR could be added here

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1: Eye Tracking")
    parser.add_argument("--cam", type=int, default=0, help="webcam index")
    parser.add_argument("--threshold", type=float, default=0.21, help="EAR threshold")
    args = parser.parse_args()
    main(cam_index=args.cam, ear_threshold=args.threshold)
