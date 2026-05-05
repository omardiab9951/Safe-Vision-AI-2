from ultralytics import YOLO
import cv2
import os
import time
import threading
import csv
import argparse
from datetime import datetime

# =========================
# SOUND SETUP
# =========================
try:
    import winsound
    SOUND_BACKEND = "winsound"
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        SOUND_BACKEND = "pygame"
    except ImportError:
        SOUND_BACKEND = None
        print("[WARN] No sound backend found. Install pygame")

# =========================
# CONFIG
# =========================
MODEL_PATH   = "runs/detect/train2/weights/best.pt"
CAMERA_SOURCE = "0"
# "0" = webcam
# OR: "rtsp://admin:password@192.168.1.105:554/H.264"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(BASE_DIR, "alerts")
LOG_FILE    = os.path.join(BASE_DIR, "violations.csv")

os.makedirs(SAVE_FOLDER, exist_ok=True)

CONF_THRESHOLD  = 0.60
TRIGGER_DELAY   = 2.0
RESET_DELAY     = 1.0
FLASH_INTERVAL  = 0.35
ALERT_COOLDOWN  = 10.0
PROCESS_EVERY_N = 3

NO_VEST_NAMES = {"no vest", "no-vest", "no_vest", "novest"}

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--source", type=str, default=CAMERA_SOURCE)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--cooldown", type=float, default=ALERT_COOLDOWN)
    return parser.parse_args()

# =========================
# CAMERA HANDLER
# =========================
def open_camera(source):
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Using webcam {idx}")
            return cap
        else:
            return None
    else:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"[INFO] Connected to stream:\n{source}")
            return cap
        else:
            return None

# =========================
# LOGGER
# =========================
class ViolationLogger:
    def __init__(self, path):
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if os.stat(path).st_size == 0:
            self.writer.writerow(["timestamp", "camera", "event", "confidence", "image"])

    def log(self, cam, event, conf, img):
        self.writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cam, event, f"{conf:.2f}", img
        ])
        self.file.flush()

    def close(self):
        self.file.close()

# =========================
# SOUND
# =========================
alarm_stop = threading.Event()

def beep(freq, duration):
    if SOUND_BACKEND == "winsound":
        winsound.Beep(freq, duration)

def alarm_loop():
    while not alarm_stop.is_set():
        beep(1400, 250)
        time.sleep(0.1)
        beep(1400, 250)
        time.sleep(0.5)

def start_alarm():
    alarm_stop.clear()
    threading.Thread(target=alarm_loop, daemon=True).start()

def stop_alarm():
    alarm_stop.set()

# =========================
# DETECTION
# =========================
def get_no_vest(result, names, conf):
    if result.boxes is None:
        return False, 0, 0

    max_conf = 0
    count = 0

    for b in result.boxes:
        cls = int(b.cls[0])
        c = float(b.conf[0])
        name = names[cls].lower()

        if c >= conf and name in NO_VEST_NAMES:
            count += 1
            max_conf = max(max_conf, c)

    return count > 0, max_conf, count

# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    print("[INFO] Loading model...")
    model = YOLO(args.model)

    print(f"[INFO] Opening source: {args.source}")
    cap = open_camera(args.source)

    if cap is None:
        raise RuntimeError("Camera/stream not accessible.")

    logger = ViolationLogger(LOG_FILE)

    no_vest_start = None
    normal_start = None
    event = False
    alarm_on = False
    last_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        result = model(frame)[0]
        detected, conf, count = get_no_vest(result, model.names, args.conf)

        now = time.time()

        if detected:
            normal_start = None
            if no_vest_start is None:
                no_vest_start = now

            if now - no_vest_start > TRIGGER_DELAY and not event:
                if now - last_alert > args.cooldown:
                    event = True
                    last_alert = now

                    filename = os.path.join(
                        SAVE_FOLDER,
                        f"alert_{datetime.now().strftime('%H%M%S')}.jpg"
                    )
                    cv2.imwrite(filename, frame)
                    logger.log("cam", "no_vest", conf, filename)

        else:
            no_vest_start = None
            if normal_start is None:
                normal_start = now

            if now - normal_start > RESET_DELAY:
                event = False

        if event and not alarm_on:
            start_alarm()
            alarm_on = True

        if not event and alarm_on:
            stop_alarm()
            alarm_on = False

        annotated = result.plot()
        cv2.imshow("Detection", annotated)

        if cv2.waitKey(1) == ord("q"):
            break

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    logger.close()

if __name__ == "__main__":
    main()