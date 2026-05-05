<div align="center">

<img src="https://img.shields.io/badge/Safety%20AI-Full%20PPE%20Detection-FF4500?style=for-the-badge" />
<img src="https://img.shields.io/badge/Model-YOLOv8-00C4CC?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Type-Group%20Project-8A2BE2?style=for-the-badge" />
<img src="https://img.shields.io/badge/Stack-Python%20%7C%20HTML%20%7C%20JS%20%7C%20Docker-228B22?style=for-the-badge" />

# 🛡️ Safe Vision AI
### AI-Powered Full-Body PPE Compliance & Fatigue Detection System

*A full-stack deep learning system that monitors workers in real time — detecting PPE violations and fatigue before they turn into accidents.*

[![Author](https://img.shields.io/badge/Omar%20Diab-DL%20%7C%20Frontend-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/omar9951)
[![GitHub](https://img.shields.io/badge/GitHub-omardiab9951-181717?style=flat-square&logo=github)](https://github.com/omardiab9951)

</div>

---

## 🔍 Overview

Safe Vision AI is a collaborative group project delivering a **comprehensive, real-time worker safety monitoring platform** built on top of deep learning and computer vision. Unlike single-item PPE detectors, this system covers the **full spectrum of personal protective equipment** — and goes a step further by incorporating **fatigue detection** to flag workers who are at risk of accidents due to exhaustion.

The system is deployed as a web application with both a live camera feed and file upload pipeline, giving industrial facilities an end-to-end compliance monitoring tool that requires no manual intervention.

### 💡 The Problem
- Industrial accidents caused by missing PPE or worker fatigue cost billions globally in injuries, insurance claims, and lost productivity
- Monitoring the full PPE suite (helmet, goggles, vest, gloves, safety suit, shoes, face shield) manually is impossible at scale
- Worker fatigue is invisible to the naked eye until an incident has already occurred

### ✅ The Solution
A unified, always-on AI monitoring platform that simultaneously:
- Detects **8 PPE categories** across the full body in real time
- Identifies **worker fatigue** through visual behavioral cues
- Streams detection results to a live dashboard
- Accepts uploaded images and videos for retrospective compliance audits

---

## 🎯 Detected Classes

| # | PPE Item | Detection Target |
|---|---|---|
| 1 | 🪖 **Helmet** | Head protection compliance |
| 2 | 🥽 **Goggles** | Eye protection compliance |
| 3 | 🦺 **Safety Vest** | High-visibility vest compliance |
| 4 | 🧤 **Gloves** | Hand protection compliance |
| 5 | 🥼 **Safety Suit** | Full-body protection compliance |
| 6 | 👟 **Safety Shoes** | Foot protection compliance |
| 7 | 🛡️ **Face Shield** | Face protection compliance |
| 8 | 😴 **Fatigue** | Worker alertness & drowsiness detection |

---

## 🏗️ System Architecture

```
Safe Vision AI
│
├── 🤖  AI Model Layer          — YOLOv8 multi-class PPE + fatigue detection
├── ⚙️  Backend                 — Python API serving model inference
├── 🌐  Frontend (websiteinteg) — Real-time dashboard (HTML / CSS / JS)
└── 🐳  Docker                  — Containerized deployment
```

The AI core is a **YOLOv8** model trained to simultaneously detect all 8 classes in a single inference pass. The backend exposes the model through an API that feeds both the live camera stream and the file upload endpoints. The frontend dashboard renders bounding boxes, class labels, and compliance status in real time.

---

## 🏋️ Model Details

| Component | Details |
|---|---|
| **Architecture** | YOLOv8, YOLOv8m, SwinV2 |
| **Task** | Multi-class Object Detection (8 classes) |
| **Framework** | PyTorch (via Ultralytics) |
| **Input Resolution** | 640 × 640 |
| **Inference Modes** | Live camera feed · Image upload · Video upload |

---

## 📦 Dataset

- **Collection:** Mix of self-collected images curated by the team and publicly available PPE datasets
- **Labeling:** Annotated using **Roboflow** with per-class bounding boxes across all 8 PPE and fatigue categories
- **Format:** YOLO-format bounding box annotations
- **Split:** Train / validation split configured via Roboflow's export pipeline

---

## 🌐 Web Application

The system ships with a full-stack web interface built by the team:

**Live Camera Mode**
- Connect any CCTV or webcam feed
- Real-time frame-by-frame inference with bounding box overlay
- Instant per-class compliance status on the dashboard

**Upload Mode**
- Upload static images or recorded video files
- Full PPE audit report generated per frame
- Suitable for retrospective safety reviews and compliance logging

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install ultralytics
pip install opencv-python
pip install flask        # or fastapi, depending on backend setup
```

### Clone the Repository
```bash
git clone https://github.com/omardiab9951/Safe-Vision-AI-1.git
cd Safe-Vision-AI-1
```

### Run with Docker
```bash
docker build -t safevision-ai .
docker run -p 5000:5000 safevision-ai
```

### Run Backend Manually
```bash
cd backend
python app.py
```

### Run Inference Directly
```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

# On an image
model.predict(source="worker_image.jpg", conf=0.5, save=True)

# On a live webcam
model.predict(source=0, conf=0.5, show=True)

# On a video file
model.predict(source="site_footage.mp4", conf=0.5, save=True)
```

---

## 🎯 Target Use Cases

| Environment | Application |
|---|---|
| 🏭 Manufacturing Plants | Full-body PPE compliance across production floors |
| 🏗️ Construction Sites | Multi-item safety audit via existing CCTV |
| ⚗️ Chemical & Oil Facilities | Hazard suit, gloves, and face shield enforcement |
| 🚧 Road & Infrastructure Works | High-vis vest and helmet monitoring |
| 🏥 Medical & Lab Facilities | Gloves, face shield, and protective suit checks |
| 🌙 Night Shifts | Fatigue detection to prevent exhaustion-related incidents |

---

## 📈 Why This Matters

> The International Labour Organization estimates **2.3 million workers** die annually from work-related accidents. A large proportion of non-fatal injuries are directly linked to PPE non-compliance or worker fatigue — both of which are preventable with real-time monitoring.

Safe Vision AI addresses both failure modes simultaneously — making it one of the most comprehensive open-source industrial safety systems available.

---

## 👥 Team & Roles

This is a group project developed collaboratively. Contributions span the full stack — from dataset curation and model training to backend development and frontend integration.

| Contributor | Role |
|---|---|
| **Omar Diab** | Team Leader · DL Model Training & Deployment · Frontend Development |
| **Ahmed Baher** | DL Model Training & Deployment · Full Stack AI Engineer |
| **Shaza Alaa** | DL Model Training & Deployment |
| **Haneen Ahmed** | DL Model Training & Deployment |



## 🔗 Related Projects

Part of a broader PPE safety detection initiative:

- 🥽 [Goggles Detection Model](https://github.com/omardiab9951/Googles-Detection-Model) — Standalone goggles compliance detector
- 🦺 [Safety Suit Detection Model](https://github.com/omardiab9951/Safety-Suit-Detection) — Standalone safety suit compliance detector

---

## 👤 Author (Omar Diab)

**Data Science & AI Student — ElSewedy University of Technology, Polytechnic of Egypt**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-omar9951-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/omar9951)
[![GitHub](https://img.shields.io/badge/GitHub-omardiab9951-181717?style=flat-square&logo=github)](https://github.com/omardiab9951)
[![Email](https://img.shields.io/badge/Email-omarkamaldiab9951@gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:omarkamaldiab9951@gmail.com)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

*Eight classes. One model. Zero blind spots.*

⭐ If this project helped you, consider giving it a star!

</div>
