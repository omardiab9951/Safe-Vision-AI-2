from ultralytics import YOLO
import cv2
import os
import time
import threading
import csv
import argparse
from datetime import datetime

# Cross-platform sound
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
        print("[WARN] No sound backend found. Install pygame for cross-platform audio: pip install pygame")


# =========================
# CONFIG
# =========================
MODEL_PATH       = "runs/detect/train2/weights/best.pt"
CAMERA_INDEX     = 0

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER      = os.path.join(BASE_DIR, "alerts")
LOG_FILE         = os.path.join(BASE_DIR, "violations.csv")

os.makedirs(SAVE_FOLDER, exist_ok=True)

CONF_THRESHOLD        = 0.60
TRIGGER_DELAY         = 2.0    # seconds of continuous no-vest before alert
RESET_DELAY           = 1.0    # seconds of clear detection before reset
FLASH_INTERVAL        = 0.35   # warning text flash speed (seconds)
ALERT_COOLDOWN        = 10.0   # minimum seconds between repeated alerts
PROCESS_EVERY_N       = 3      # run inference every N frames (skip the rest)

NO_VEST_NAMES = {"no vest", "no-vest", "no_vest", "novest"}


# =========================
# ARGUMENT PARSER (optional overrides)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Vest Detection")
    parser.add_argument("--model",   type=str, default=MODEL_PATH,   help="Path to YOLO model weights")
    parser.add_argument("--camera",  type=int, default=CAMERA_INDEX, help="Camera device index")
    parser.add_argument("--conf",    type=float, default=CONF_THRESHOLD, help="Confidence threshold (0-1)")
    parser.add_argument("--cooldown",type=float, default=ALERT_COOLDOWN, help="Cooldown between alerts (s)")
    return parser.parse_args()


# =========================
# CSV LOG (opened once, kept open)
# =========================
class ViolationLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._lock = threading.Lock()
        file_exists = os.path.exists(log_path)
        self._file = open(log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(["timestamp", "camera", "event", "confidence", "image"])
            self._file.flush()

    def log(self, camera_name, event_name, confidence, image_path):
        with self._lock:
            self._writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                camera_name,
                event_name,
                f"{confidence:.2f}",
                image_path
            ])
            self._file.flush()

    def close(self):
        self._file.close()


# =========================
# ALARM THREAD
# =========================
alarm_stop_event = threading.Event()
alarm_thread     = None


def alarm_loop():
    """Repeating double-beep alarm. Runs in a daemon thread."""
    while not alarm_stop_event.is_set():
        _play_beep(1400, 250)
        if alarm_stop_event.wait(0.10):
            break
        _play_beep(1400, 250)
        if alarm_stop_event.wait(0.45):
            break


def _play_beep(freq, duration_ms):
    if SOUND_BACKEND == "winsound":
        try:
            winsound.Beep(freq, duration_ms)
        except RuntimeError:
            pass
    elif SOUND_BACKEND == "pygame":
        sample_rate = 44100
        import numpy as np
        t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        wave = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(wave)
        sound.play()
        time.sleep(duration_ms / 1000)


def start_alarm():
    global alarm_thread
    if SOUND_BACKEND is None:
        return
    if alarm_thread is None or not alarm_thread.is_alive():
        alarm_stop_event.clear()
        alarm_thread = threading.Thread(target=alarm_loop, daemon=True)
        alarm_thread.start()


def stop_alarm():
    alarm_stop_event.set()


# =========================
# DETECTION HELPERS
# =========================
def get_no_vest_info(result, model_names, conf_threshold):
    """
    Returns (detected: bool, max_confidence: float, count: int)
    for all 'no vest' boxes above the threshold.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return False, 0.0, 0

    max_conf = 0.0
    count    = 0

    for box in result.boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        class_name = str(model_names[cls_id]).lower().strip()

        if conf >= conf_threshold and class_name in NO_VEST_NAMES:
            count   += 1
            max_conf = max(max_conf, conf)

    return count > 0, max_conf, count


def find_camera(preferred_index):
    """Try the preferred index first, then fall back to 0–3."""
    for idx in [preferred_index] + [i for i in range(4) if i != preferred_index]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            if idx != preferred_index:
                print(f"[WARN] Camera {preferred_index} not found. Using camera {idx}.")
            return cap
        cap.release()
    return None


def draw_overlay(frame, event_active, current_time, no_vest_duration,
                 max_conf, det_count, trigger_delay):
    """Draw all HUD elements on the frame in-place."""

    h, w = frame.shape[:2]

    # --- Status bar background ---
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)

    # --- Detection count & confidence (bottom bar) ---
    status_text = (
        f"No-Vest detections: {det_count}  |  "
        f"Conf: {max_conf:.0%}  |  "
        f"Press Q to quit"
    )
    cv2.putText(frame, status_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Progress bar (fills up while waiting to trigger) ---
    if no_vest_duration > 0 and not event_active:
        progress = min(no_vest_duration / trigger_delay, 1.0)
        bar_w    = int((w - 20) * progress)
        cv2.rectangle(frame, (10, h - 48), (w - 10, h - 43), (60, 60, 60), -1)
        color = (0, 165 + int(90 * progress), 255 - int(255 * progress))
        cv2.rectangle(frame, (10, h - 48), (10 + bar_w, h - 43), color, -1)

    # --- Flashing alert banner ---
    if event_active:
        flash_on = int(current_time / FLASH_INTERVAL) % 2 == 0
        if flash_on:
            cv2.rectangle(frame, (10, 10), (560, 90), (0, 0, 220), -1)
            cv2.putText(frame, "! NO VEST DETECTED !",
                        (22, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 4, cv2.LINE_AA)


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Opening camera {args.camera} ...")
    cap = find_camera(args.camera)
    if cap is None:
        raise RuntimeError("Could not open any camera. Check connections.")

    logger = ViolationLogger(LOG_FILE)
    print(f"[INFO] Alert folder : {SAVE_FOLDER}")
    print(f"[INFO] Log file     : {LOG_FILE}")
    print("[INFO] Press 'q' to quit.\n")

    # State
    no_vest_start_time      = None
    normal_start_time       = None
    event_active            = False
    snapshot_taken          = False
    alarm_active            = False
    last_alert_time         = 0.0
    frame_count             = 0
    last_result             = None
    last_no_vest_duration   = 0.0
    last_max_conf           = 0.0
    last_det_count          = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 1 = horizontal flip (mirror)
        if not ret:
            print("[ERROR] Could not read frame. Exiting.")
            break

        frame_count  += 1
        current_time  = time.time()

        # -------------------------------------------------------
        # FRAME SKIPPING: run inference only every N frames
        # -------------------------------------------------------
        if frame_count % PROCESS_EVERY_N == 0:
            last_result = model(frame, verbose=False)[0]
            no_vest_now, last_max_conf, last_det_count = get_no_vest_info(
                last_result, model.names, args.conf
            )

            # --- No-vest timer logic ---
            if no_vest_now:
                normal_start_time = None
                if no_vest_start_time is None:
                    no_vest_start_time = current_time

                last_no_vest_duration = current_time - no_vest_start_time

                if (last_no_vest_duration >= TRIGGER_DELAY
                        and not event_active
                        and (current_time - last_alert_time) >= args.cooldown):
                    event_active   = True
                    snapshot_taken = False
                    last_alert_time = current_time
                    print(f"[ALERT] No vest for {TRIGGER_DELAY}s — alert triggered.")

            else:
                no_vest_start_time    = None
                last_no_vest_duration = 0.0

                if normal_start_time is None:
                    normal_start_time = current_time

                normal_duration = current_time - normal_start_time

                if normal_duration >= RESET_DELAY:
                    if event_active:
                        print("[INFO] Alert cleared.")
                    event_active   = False
                    snapshot_taken = False

        # Use the last annotated frame if we have a result
        annotated = last_result.plot() if last_result is not None else frame.copy()

        # -------------------------------------------------------
        # SNAPSHOT + LOG (once per event, non-blocking)
        # -------------------------------------------------------
        if event_active and not snapshot_taken:
            ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.abspath(
                os.path.join(SAVE_FOLDER, f"no_vest_{ts}.jpg")
            )
            if cv2.imwrite(filename, annotated):
                print(f"[ALERT] Snapshot → {filename}")
                logger.log("cam0", "no_vest", last_max_conf, filename)
                print(f"[LOG]   Recorded in {LOG_FILE}")
            else:
                print("[ERROR] Snapshot could not be saved.")
            snapshot_taken = True

        # -------------------------------------------------------
        # ALARM
        # -------------------------------------------------------
        if event_active and not alarm_active:
            start_alarm()
            alarm_active = True

        if not event_active and alarm_active:
            stop_alarm()
            alarm_active = False

        # -------------------------------------------------------
        # HUD OVERLAY
        # -------------------------------------------------------
        draw_overlay(
            annotated, event_active, current_time,
            last_no_vest_duration, last_max_conf,
            last_det_count, TRIGGER_DELAY
        )

        cv2.imshow("Vest Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()