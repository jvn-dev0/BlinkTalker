import cv2
import mediapipe as mp

def get_eye_direction(landmarks, w, h):
    # Iris indices (Mediapipe)
    RIGHT_IRIS = 468
    LEFT_IRIS = 473
    
    right_iris = landmarks[RIGHT_IRIS]
    left_iris = landmarks[LEFT_IRIS]

    rx, ry = int(right_iris.x * w), int(right_iris.y * h)
    lx, ly = int(left_iris.x * w), int(left_iris.y * h)

    avg_x = (rx + lx) / 2

    # Threshold zones
    if avg_x < w * 0.40:
        return "LEFT"
    elif avg_x > w * 0.60:
        return "RIGHT"
    else:
        return "CENTER"

def main(cam_index=0):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(cam_index)
    print("Gaze tracking running... Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            gaze = get_eye_direction(lm, w, h)

            cv2.putText(frame, f"GAZE: {gaze}", (20, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("BlinkTalker - Module 3 Gaze Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
