from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# 🔧 CONFIGURATION
MODEL_PATH = 'best.pt'  # Make sure your model is named exactly this
CONF_THRESHOLD = 0.45

# 🔌 LOAD MODEL
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    model.to('cpu')
    print(f"✅ Model loaded: {MODEL_PATH}")
else:
    print(f"❌ Model NOT FOUND: {MODEL_PATH}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        # Read image bytes
        file_bytes = np.frombuffer(request.files['image'].read(), np.uint8)
        
        # Decode to BGR (OpenCV default - DO NOT convert to RGB)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run prediction (YOLO handles all preprocessing internally)
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        
        # Draw boxes on the image
        img_out = img.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Encode result to base64
        _, buffer = cv2.imencode('.jpg', img_out)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'count': len(results[0].boxes)
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect/webcam', methods=['POST'])
def detect_webcam():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        
        img_out = img.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img_out)
        return jsonify({
            'success': True,
            'image': base64.b64encode(buffer).decode('utf-8'),
            'count': len(results[0].boxes)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"\n🛡️ Safety Suit Detector")
    print(f"📂 Model: {MODEL_PATH} | Threshold: {CONF_THRESHOLD}")
    print(f"🌐 http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)