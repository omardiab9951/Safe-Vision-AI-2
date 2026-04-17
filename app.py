# app.py - CLEAN & ACCURATE VERSION
import os, cv2, base64, numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = YOLO('best.pt')

# 🔑 VERY STRICT SETTINGS to prevent false positives
CONF = 0.65  # High confidence required
MIN_BOX_SIZE = 30  # Goggles box must be at least 30x30 pixels

def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
        
        # Save input
        if 'image' in request.files:
            request.files['image'].save(path)
        elif 'webcam_frame' in request.form:
            data = request.form['webcam_frame'].split(',')[1]
            img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), 1)
            cv2.imwrite(path, img)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Load and save ORIGINAL image (no modifications)
        img_cv = cv2.imread(path)
        if img_cv is None:
            return jsonify({'error': 'Failed to read image'}), 400
            
        h, w, _ = img_cv.shape
        
        # Run inference
        results = model(path, conf=CONF, verbose=False)
        
        # --- STRICT DETECTION LOGIC ---
        has_goggles = False
        detection_conf = 0.0
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get box properties
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Calculate box size
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Calculate center position
                y_center = (y1 + y2) / 2
                x_center = (x1 + x2) / 2
                
                # STRICT FILTERS:
                # 1. Must be class 0 (safety_goggles)
                # 2. Must have high confidence
                # 3. Must be in upper half of image (face region)
                # 4. Must be reasonable size (not too small, not too big)
                # 5. Must be somewhat centered horizontally
                
                if (cls == 0 and 
                    conf >= CONF and
                    y_center < h * 0.6 and  # Upper 60% of image
                    box_area > (MIN_BOX_SIZE * MIN_BOX_SIZE) and
                    box_area < (h * w * 0.15) and  # Not more than 15% of image
                    0.2 * w < x_center < 0.8 * w):  # Centered horizontally
                    
                    has_goggles = True
                    detection_conf = conf
                    break
        
        # Save ORIGINAL image (no overlays)
        out_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
        cv2.imwrite(out_path, img_cv)
        
        # Return status separately (NOT on image)
        if has_goggles:
            status = "✅ COMPLIANT - Safety Goggles Detected"
            status_class = "compliant"
            confidence_text = f"Confidence: {detection_conf:.1%}"
        else:
            status = "❌ VIOLATION - No Safety Goggles"
            status_class = "violation"
            confidence_text = "No goggles detected"
        
        return jsonify({
            'status': 'success',
            'image_base64': img_to_b64(out_path),
            'is_compliant': has_goggles,
            'status_text': status,
            'status_class': status_class,
            'confidence': confidence_text
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    print("="*60)
    print("🛡️  SAFETY GOGGLES DETECTOR (STRICT MODE)")
    print("="*60)
    print(f"📂 Model: best.pt")
    print(f"🎯 Confidence: {CONF} (Very High)")
    print(f"📏 Min box size: {MIN_BOX_SIZE}px")
    print("🌐 Server: http://localhost:5000")
    print("="*60)
    print("⚠️  STRICT MODE: Only detects clear, obvious goggles")
    app.run(host='0.0.0.0', port=5000, debug=True)