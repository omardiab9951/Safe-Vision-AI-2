from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # small, fast starter model

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=416,
        batch=8,
        project="runs",
        name="face_shield_train"
    )

if __name__ == "__main__":
    main()