from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load your EXACT model
model = YOLO('best (1).pt')
model.to('cpu')

CONF = 0.70  # Must match app.py

print("🧪 Direct Model Test (conf=0.70)")
print("="*50)

# Test cases - replace with your actual test images
test_images = {
    'images.JFIF': 'Should detect ✅',
    'safetyvest.JFIF': 'Should NOT detect ❌',
    'nnormalclothes.JFIF': 'Should NOT detect ❌',
    'empty.jpg': 'Should NOT detect ❌',
}

for filename, expected in test_images.items():
    if os.path.exists(filename):
        img = cv2.imread(filename)
        results = model.predict(img, conf=CONF, verbose=False)
        count = len(results[0].boxes)
        
        # Determine pass/fail
        should_detect = 'detect' in expected and 'NOT' not in expected
        passed = (count > 0) == should_detect
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{status} {filename}: {count} detections ({expected})")
        
        # Save result image
        annotated = results[0].plot()
        cv2.imwrite(f'out_{filename}', annotated)
    else:
        print(f"⚠️  {filename} not found (create this test image)")

print("\n💡 If tests FAIL but training showed Precision=1.00:")
print("   → Your web app might be using a different confidence value")
print("   → Check app.py: CONF_THRESHOLD must be 0.70")