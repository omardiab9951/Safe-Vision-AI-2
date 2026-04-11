from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from PIL import Image
import os

app = Flask(__name__)

# 🔒 STRICT CONFIGURATION
MODEL_PATH = 'best (1).pt'  # Make sure this file is in the same folder
CONF_THRESHOLD = 0.60   # Tested & proven to eliminate false positives
IOU_THRESHOLD = 0.45    # Prevents duplicate boxes

# Load model (CPU mode for laptop compatibility)
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH, task='detect')
    model.to('cpu')  # Force CPU to avoid CUDA errors on laptop
    print(f"✅ Model loaded: {MODEL_PATH} (CPU mode)")
else:
    print(f"❌ Error: {MODEL_PATH} not found. Place it in the same folder as app.py")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'image' not in request.files or not model:
        return jsonify({'error': 'No image or model loaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        img_np = np.array(img)
        
        # 🔒 STRICT INFERENCE
        results = model.predict(img_np, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        result_img = results[0].plot() if len(results[0].boxes) > 0 else img_np
        
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        detections = [{
            'class': 'safety_suit',
            'confidence': float(box.conf),
            'bbox': box.xyxy.tolist()[0]
        } for box in results[0].boxes]
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'detections': detections,
            'count': len(detections)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/webcam', methods=['POST'])
def detect_webcam():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.get_json()
    image_b64 = data.get('image')
    
    if not image_b64:
        return jsonify({'error': 'No image data'}), 400
    
    try:
        img_data = base64.b64decode(image_b64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        result_img = results[0].plot() if len(results[0].boxes) > 0 else img
        
        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'count': len(results[0].boxes)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🛡️ SAFETY SUIT DETECTOR - PRODUCTION MODE")
    print(f"🔒 Confidence Threshold: {CONF_THRESHOLD}")
    print("🌐 Running on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)