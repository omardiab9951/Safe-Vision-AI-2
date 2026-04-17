document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // Update buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`${tab}-tab`).classList.add('active');
        });
    });
    
    // Image Upload
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const detectBtn = document.getElementById('detect-btn');
    const imageResult = document.getElementById('image-result');
    const resultImage = document.getElementById('result-image');
    const detectionInfo = document.getElementById('detection-info');
    
    // Click to upload
    dropZone.addEventListener('click', () => imageInput.click());
    
    // Drag & drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });
    
    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            imageInput.files = files;
            handleFileSelect();
        }
    }
    
    imageInput.addEventListener('change', handleFileSelect);
    
    function handleFileSelect() {
        if (imageInput.files && imageInput.files[0]) {
            detectBtn.disabled = false;
            dropZone.querySelector('.upload-prompt p').textContent = 
                `✅ Selected: ${imageInput.files[0].name}`;
        }
    }
    
    detectBtn.addEventListener('click', async () => {
        if (!imageInput.files[0]) return;
        
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        
        showStatus('🔄 Detecting...', 'loading');
        detectBtn.disabled = true;
        
        try {
            const response = await fetch('/detect/image', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                resultImage.src = `data:image/jpeg;base64,${data.image}`;
                detectionInfo.innerHTML = `
                    <strong>🎯 ${data.count} safety suit(s) detected!</strong><br>
                    <small>Confidence threshold: 45%</small>
                `;
                imageResult.style.display = 'block';
                showStatus('✅ Detection complete!', 'success');
            } else {
                showStatus(`❌ Error: ${data.error}`, 'error');
            }
        } catch (error) {
            showStatus('❌ Connection error', 'error');
            console.error(error);
        } finally {
            detectBtn.disabled = false;
        }
    });
    
    // Webcam
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const startWebcamBtn = document.getElementById('start-webcam');
    const captureBtn = document.getElementById('capture-btn');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const webcamResult = document.getElementById('webcam-result');
    const webcamResultImage = document.getElementById('webcam-result-image');
    const webcamStatus = document.getElementById('webcam-status');
    
    let stream = null;
    
    startWebcamBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            webcamVideo.srcObject = stream;
            
            startWebcamBtn.disabled = true;
            captureBtn.disabled = false;
            stopWebcamBtn.disabled = false;
            showWebcamStatus('🎥 Camera active', 'success');
        } catch (err) {
            showWebcamStatus('❌ Camera access denied', 'error');
            console.error('Camera error:', err);
        }
    });
    
    captureBtn.addEventListener('click', async () => {
        if (!stream) return;
        
        // Draw video frame to canvas
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0);
        
        // Convert to base64
        const imageData = webcamCanvas.toDataURL('image/jpeg', 0.9);
        
        showWebcamStatus('🔄 Detecting...', 'loading');
        captureBtn.disabled = true;
        
        try {
            const response = await fetch('/detect/webcam', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const data = await response.json();
            
            if (data.success) {
                webcamResultImage.src = `data:image/jpeg;base64,${data.image}`;
                webcamResult.style.display = 'block';
                showWebcamStatus(`✅ ${data.count} suit(s) detected!`, 'success');
            } else {
                showWebcamStatus(`❌ Error: ${data.error}`, 'error');
            }
        } catch (error) {
            showWebcamStatus('❌ Connection error', 'error');
            console.error(error);
        } finally {
            captureBtn.disabled = false;
        }
    });
    
    stopWebcamBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        webcamVideo.srcObject = null;
        webcamResult.style.display = 'none';
        
        startWebcamBtn.disabled = false;
        captureBtn.disabled = true;
        stopWebcamBtn.disabled = true;
        showWebcamStatus('⏹️ Camera stopped', '');
    });
    
    // Helper functions
    function showStatus(message, type) {
        // You can add a status element for image tab if needed
        console.log(`${type}: ${message}`);
    }
    
    function showWebcamStatus(message, type) {
        webcamStatus.textContent = message;
        webcamStatus.className = `status ${type}`;
    }
});