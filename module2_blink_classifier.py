# Module 2 – Blink Classifier (Short / Long / Double Blink)
# Depends on: EAR detection (Module 1)
import cv2
import mediapipe as mp
import time
import math
import argparse

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_EAR(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    h1 = euclidean_distance(pts[0], pts[3])
    if h1 == 0:
        return 0
    return (v1 + v2) / (2 * h1)

def main(cam_index=0, threshold=0.21):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [362, 385, 387, 263, 373, 380]

    cap = cv2.VideoCapture(cam_index)

    blink_start = None
    last_blink_time = 0
    blink_sequence = ""

    SHORT_MIN = 0.10   # 100ms
    SHORT_MAX = 0.25   # 250ms
    LONG_MIN = 0.30    # 300ms
    DOUBLE_GAP = 0.60  # 600ms

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            leftEAR = calculate_EAR(lm, LEFT, w, h)
            rightEAR = calculate_EAR(lm, RIGHT, w, h)
            EAR = (leftEAR + rightEAR) / 2

            eyes_closed = EAR < threshold

            # --- blink timing tracking ---
            if eyes_closed and blink_start is None:
                blink_start = time.time()

            if not eyes_closed and blink_start:
                duration = time.time() - blink_start
                now = time.time()

                # Short blink
                if SHORT_MIN <= duration <= SHORT_MAX:
                    # detect double blink
                    if now - last_blink_time < DOUBLE_GAP:
                        blink_sequence += " "   # space
                    else:
                        blink_sequence += "."
                    last_blink_time = now

                # Long blink
                elif duration >= LONG_MIN:
                    blink_sequence += "-"
                    last_blink_time = now

                blink_start = None

            # Display
            cv2.putText(frame, f"EAR: {EAR:.3f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(frame, f"Sequence: {blink_sequence}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("BlinkTalker - Module 2 (Blink Classification)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('c'):
            blink_sequence = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.21)
    args = parser.parse_args()
    main(args.cam, args.threshold)
