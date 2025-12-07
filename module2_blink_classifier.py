# BlinkTalker – Module 2 + 4 Integrated
# Blink Classifier + Morse Decoder + Real-Time Text Builder

import cv2
import mediapipe as mp
import time
import math
from module4_morse_decoder import MorseDecoder  # <-- correct import only


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

    # Initialize modules
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    decoder = MorseDecoder()
    blink_buffer = ""  # raw dot/dash buffer for display only

    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [362, 385, 387, 263, 373, 380]

    cap = cv2.VideoCapture(cam_index)

    blink_start = None
    last_blink_time = time.time()

    # Blink thresholds
    SHORT_MIN = 0.10  # 100ms
    SHORT_MAX = 0.25  # 250ms
    LONG_MIN = 0.30  # 300ms
    LETTER_GAP = 1.0  # 1 sec = auto finalize letter
    WORD_GAP = 2.0  # 2 sec = auto finalize word

    print("\nBlinkTalker running...\n'q' = quit | 'c' = clear\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            leftEAR = calculate_EAR(lm, LEFT, w, h)
            rightEAR = calculate_EAR(lm, RIGHT, w, h)
            EAR = (leftEAR + rightEAR) / 2

            eyes_closed = EAR < threshold

            # Start blink timer
            if eyes_closed and blink_start is None:
                blink_start = time.time()

            # Blink ended
            if not eyes_closed and blink_start:
                duration = now - blink_start

                # Short blink = DOT
                if SHORT_MIN <= duration <= SHORT_MAX:
                    blink_buffer += "."
                    decoder.add_symbol(".")
                    last_blink_time = now

                # Long blink = DASH
                elif duration >= LONG_MIN:
                    blink_buffer += "-"
                    decoder.add_symbol("-")
                    last_blink_time = now

                blink_start = None

            # Auto-letter detection
            if now - last_blink_time >= LETTER_GAP and decoder.buffer:
                decoded = decoder.end_letter()
                blink_buffer += f" [{decoded}] "
                last_blink_time = now

            # Auto-word detection
            if now - last_blink_time >= WORD_GAP and not decoder.buffer:
                decoder.end_word()
                blink_buffer += " / "
                last_blink_time = now

            # Display EAR
            cv2.putText(
                frame,
                f"EAR: {EAR:.3f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Display Raw Morse Buffer
            cv2.putText(
                frame,
                f"Morse: {blink_buffer[-35:]}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Display Output Text
            cv2.putText(
                frame,
                f"Text: {decoder.get_text()}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("BlinkTalker – Integrated (Blink + Morse)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            blink_buffer = ""
            decoder.current_word = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
