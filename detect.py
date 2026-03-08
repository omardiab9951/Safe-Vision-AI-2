from ultralytics import YOLO
import cv2
import os
import time
import threading
from datetime import datetime

# Windows sound
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/train2/weights/best.pt"   # change if needed
CAMERA_INDEX = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(BASE_DIR, "alerts")
os.makedirs(SAVE_FOLDER, exist_ok=True)

CONF_THRESHOLD = 0.60
TRIGGER_DELAY = 2.0       # trigger only after 2 seconds of continuous no-vest
RESET_DELAY = 1.0         # reset after 1 second of normal detection
FLASH_INTERVAL = 0.35     # text flashing speed

NO_VEST_NAMES = {"no vest", "no-vest", "no_vest", "novest"}


# =========================
# ALARM THREAD
# =========================
alarm_stop_event = threading.Event()
alarm_thread = None


def alarm_loop():
    """
    Repeating alarm pattern:
    BEEP BEEP ... pause ... BEEP BEEP ...
    Runs in a separate thread so video does not freeze.
    """
    while not alarm_stop_event.is_set():
        if SOUND_AVAILABLE:
            try:
                winsound.Beep(1400, 250)   # beep 1
            except RuntimeError:
                pass

        if alarm_stop_event.wait(0.10):
            break

        if SOUND_AVAILABLE:
            try:
                winsound.Beep(1400, 250)   # beep 2
            except RuntimeError:
                pass

        if alarm_stop_event.wait(0.45):
            break


def start_alarm():
    global alarm_thread
    if not SOUND_AVAILABLE:
        return

    if alarm_thread is None or not alarm_thread.is_alive():
        alarm_stop_event.clear()
        alarm_thread = threading.Thread(target=alarm_loop, daemon=True)
        alarm_thread.start()


def stop_alarm():
    alarm_stop_event.set()


# =========================
# HELPER FUNCTION
# =========================
def is_no_vest_detected(result, model_names, conf_threshold=0.6):
    """
    Returns True if at least one 'no vest' detection is found
    above the confidence threshold.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return False

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = str(model_names[cls_id]).lower().strip()

        if conf >= conf_threshold and class_name in NO_VEST_NAMES:
            return True

    return False


# =========================
# MAIN
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Alert folder:", SAVE_FOLDER)
print("Press 'q' to quit.")

# State variables
no_vest_start_time = None
normal_start_time = None
event_active = False
snapshot_taken_for_event = False
alarm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read frame from camera.")
        break

    # Run detection
    result = model(frame, verbose=False)[0]
    annotated = result.plot()

    current_time = time.time()
    no_vest_now = is_no_vest_detected(result, model.names, CONF_THRESHOLD)

    # =========================
    # 2-SECOND NO-VEST TIMER
    # =========================
    if no_vest_now:
        normal_start_time = None

        if no_vest_start_time is None:
            no_vest_start_time = current_time

        no_vest_duration = current_time - no_vest_start_time

        if no_vest_duration >= TRIGGER_DELAY and not event_active:
            event_active = True
            snapshot_taken_for_event = False
            print("[ALERT] No vest detected for 2 seconds.")

    else:
        no_vest_start_time = None

        if normal_start_time is None:
            normal_start_time = current_time

        normal_duration = current_time - normal_start_time

        if normal_duration >= RESET_DELAY:
            if event_active:
                print("[INFO] Alert reset.")

            event_active = False
            snapshot_taken_for_event = False

            if alarm_active:
                stop_alarm()
                alarm_active = False

    # =========================
    # SAVE SNAPSHOT ONCE PER EVENT
    # =========================
    if event_active and not snapshot_taken_for_event:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.abspath(os.path.join(SAVE_FOLDER, f"no_vest_{timestamp}.jpg"))

        saved = cv2.imwrite(filename, annotated)
        if saved:
            print(f"[ALERT] Snapshot saved to: {filename}")
        else:
            print("[ERROR] Snapshot was not saved.")

        snapshot_taken_for_event = True

    # =========================
    # START / STOP ALARM
    # =========================
    if event_active and not alarm_active:
        start_alarm()
        alarm_active = True

    if not event_active and alarm_active:
        stop_alarm()
        alarm_active = False

    # =========================
    # FLASHING WARNING TEXT
    # =========================
    if event_active:
        flash_on = int(current_time / FLASH_INTERVAL) % 2 == 0

        if flash_on:
            cv2.rectangle(annotated, (10, 10), (540, 90), (0, 0, 255), -1)
            cv2.putText(
                annotated,
                "NO VEST DETECTED",
                (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                4
            )

    # Show video
    cv2.imshow("Vest Detection", annotated)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()