from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2, numpy as np, base64, os, sys

app = Flask(__name__)
MODEL_PATH = 'best.pt'

if not os.path.exists(MODEL_PATH):
    print(f"❌ FATAL: {MODEL_PATH} not found in {os.getcwd()}")
    sys.exit(1)

print("✅ Loading model...")
model = YOLO(MODEL_PATH)
model.to('cpu')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    try:
        img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
        results = model.predict(img, conf=0.45, verbose=False)
        
        img_out = img.copy()
        count = 0
        for box in results[0].boxes:
            # 🔥 CRITICAL: Skip Class 1 (Vests), Keep Class 0 (Suits)
            if int(box.cls) == 1:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1
            
        _, buf = cv2.imencode('.jpg', img_out)
        return jsonify({'success': True, 'image': base64.b64encode(buf).decode(), 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/webcam', methods=['POST'])
def detect_webcam():
    try:
        data = request.get_json()
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data['image'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        results = model.predict(img, conf=0.45, verbose=False)
        img_out = img.copy()
        count = 0
        for box in results[0].boxes:
            if int(box.cls) == 1: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1
        _, buf = cv2.imencode('.jpg', img_out)
        return jsonify({'success': True, 'image': base64.b64encode(buf).decode(), 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)