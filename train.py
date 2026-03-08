from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="vest-no-vest-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)