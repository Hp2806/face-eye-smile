import cv2
import os
import time
import sys

# Classifiers
face_classifier  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Verify classifiers loaded
for name, clf in [("Face", face_classifier), ("Eye", eye_classifier), ("Smile", smile_classifier)]:
    if clf.empty():
        print(f"[ERROR] Could not load {name} classifier. Check your OpenCV installation.")
        sys.exit(1)

# Camera setup  
camera_index = None
for idx in range(3):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            camera_index = idx
            cap.release()
            break
    cap.release()

if camera_index is None:
    print("[ERROR] No working camera found. Check if your webcam is connected.")
    sys.exit(1)

print(f"[OK] Camera found at index {camera_index}")
video_capture = cv2.VideoCapture(camera_index)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Output 
output_dir = "captured_detections"
os.makedirs(output_dir, exist_ok=True)

# Colours
COLOR_FACE  = (0,   255,   0)   # green
COLOR_EYE   = (255,   0,   0)   # blue
COLOR_SMILE = (0,   165, 255)   # orange

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
FONT_THICK  = 1


def draw_labeled_box(frame, x, y, w, h, color, label):
    fh, fw = frame.shape[:2]

    # Clamp coordinates to frame boundaries
    x, y = max(0, x), max(0, y)
    w = min(w, fw - x)
    h = min(h, fh - y)
    if w <= 0 or h <= 0:
        return

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label 
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    badge_h   = text_h + baseline + 6
    badge_top = max(0, y - badge_h)
    badge_bot = badge_top + badge_h

    cv2.rectangle(frame,
                  (x, badge_top),
                  (x + text_w + 8, badge_bot),
                  color, cv2.FILLED)

    cv2.putText(frame, label,
                (x + 4, badge_bot - baseline - 2),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)


def detect_and_draw(frame):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # improves detection in poor lighting

    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    smile_detected = False

    for (x, y, w, h) in faces:
        draw_labeled_box(frame, x, y, w, h, COLOR_FACE, "Face")

        # Upper half for eyes  
        eye_roi_gray  = gray[y : y + h // 2,       x : x + w]
        eye_roi_color = frame[y : y + h // 2,       x : x + w]

        # Lower half for smiles 
        smile_roi_gray  = gray[y + h // 2 : y + h, x : x + w]
        smile_roi_color = frame[y + h // 2 : y + h, x : x + w]

        # Eyes 
            eyes = eye_classifier.detectMultiScale(
            eye_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            draw_labeled_box(eye_roi_color, ex, ey, ew, eh, COLOR_EYE, "Eye")

        # Smiles 
        smiles = smile_classifier.detectMultiScale(
            smile_roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
        )
        for (sx, sy, sw, sh) in smiles:
            draw_labeled_box(smile_roi_color, sx, sy, sw, sh, COLOR_SMILE, "Smile")
            smile_detected = True

    return len(faces), smile_detected


# Saving state 
last_saved_time  = time.time()
capture_interval = 2          # periodic snapshot every 2 s when face is present

smile_start_time = None
smile_burst_secs = 2          # save every frame for 2 s after smile detected

print("[INFO] Detection running. Press 'Q' in the window to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[WARNING] Failed to grab frame, retrying...")
        time.sleep(0.05)
        continue

    num_faces, smile_detected = detect_and_draw(frame)

    # Status overlay at top-left (drawn twice for outline effect)
    status = f"Faces: {num_faces}   {'SMILE :)' if smile_detected else ''}"
    cv2.putText(frame, status, (10, 28), FONT, 0.65, (0, 0, 0),       2, cv2.LINE_AA)
    cv2.putText(frame, status, (10, 28), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Periodic snapshot every 2 seconds
    now = time.time()
    if num_faces > 0 and (now - last_saved_time) >= capture_interval:
        ts   = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(output_dir, f"detection_{ts}.jpg")
        cv2.imwrite(path, frame)
        last_saved_time = now

    # Smile
    if smile_detected:
        if smile_start_time is None:
            smile_start_time = time.time()
        if (time.time() - smile_start_time) <= smile_burst_secs:
            ms   = int(time.time() * 1000) % 1000
            ts   = time.strftime("%Y%m%d-%H%M%S") + f"_{ms:03d}"
            path = os.path.join(output_dir, f"smile_{ts}.jpg")
            cv2.imwrite(path, frame)
        else:
            smile_start_time = None   # reset after burst window
    else:
        smile_start_time = None       # reset when smile disappears

    cv2.imshow("Face | Eye | Smile Detection  [Q to quit]", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")
