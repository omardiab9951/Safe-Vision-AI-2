import cv2
import time
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/runs/face_shield_train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    raise SystemExit

# Webcam size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Settings
CONFIDENCE = 0.4
IMG_SIZE = 416
NO_GUARD_ALERT_SECONDS = 3

# Update these if your class names are different in data.yaml
FACE_GUARD_LABELS = {
    "face shield",
    "face_guard",
    "face-shield",
    "with_face_guard",
    "with shield"
}

NO_FACE_GUARD_LABELS = {
    "no face shield",
    "no_face_guard",
    "without_face_guard",
    "no shield",
    "without shield"
}

no_guard_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read webcam frame")
        break

    # Flip webcam horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Run detection
    results = model(frame, conf=CONFIDENCE, imgsz=IMG_SIZE)
    annotated_frame = results[0].plot()

    wearing_guard = False
    no_guard_detected = False

    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(names[cls_id]).strip().lower()

            if class_name in FACE_GUARD_LABELS:
                wearing_guard = True
            elif class_name in NO_FACE_GUARD_LABELS:
                no_guard_detected = True

    # Decide status
    status_text = "No Detection"
    alert_text = ""
    status_color = (0, 255, 255)  # yellow

    if wearing_guard:
        status_text = "Wearing Face Guard"
        status_color = (0, 255, 0)  # green
        no_guard_start_time = None

    elif no_guard_detected:
        status_text = "No Face Guard"
        status_color = (0, 0, 255)  # red

        if no_guard_start_time is None:
            no_guard_start_time = time.time()

        elapsed = time.time() - no_guard_start_time
        if elapsed >= NO_GUARD_ALERT_SECONDS:
            alert_text = "ERROR: Wear Face Guard"

    else:
        # No relevant detection found
        no_guard_start_time = None

    # Draw status text
    cv2.putText(
        annotated_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    # Draw alert text if needed
    if alert_text:
        cv2.putText(
            annotated_frame,
            alert_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    cv2.imshow("Face Guard Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()