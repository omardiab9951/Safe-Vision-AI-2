from ultralytics import YOLO
import cv2
import os

model = YOLO('best (1).pt')
model.to('cpu')

# Your test images
files = {
    'suit': 'images.JFIF',
    'vest': 'safetyvest.JFIF',
    'normal': 'nnormalclothes.JFIF'
}

print("🔍 CONFIDENCE SWEEP: Finding the perfect threshold")
print("="*55)

thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

for conf in thresholds:
    print(f"\n📊 Threshold: {conf:.2f}")
    for name, path in files.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            results = model.predict(img, conf=conf, verbose=False)
            boxes = results[0].boxes
            count = len(boxes)
            # Show exact confidence of each box
            confs = [f"{float(b.conf):.3f}" for b in boxes]
            status = "✅" if (name == 'suit' and count > 0) or (name != 'suit' and count == 0) else "❌"
            print(f"   {status} {name:8s} → {count} box(es) | confidences: [{', '.join(confs) or 'none'}]")
        else:
            print(f"   ⚠️  {name:8s} → File not found")

print("\n" + "="*55)
print("💡 HOW TO PICK YOUR THRESHOLD:")
print("   1. Find the lowest threshold where 'suit' shows ✅")
print("   2. Make sure 'vest' and 'normal' show ❌ at that same threshold")
print("   3. That's your production CONF_THRESHOLD!")