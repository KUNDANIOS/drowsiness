# 🛡️ DrowseGuard — Real-Time Drowsiness Detection

A browser-based drowsiness and distraction detection system for drivers, powered by **MediaPipe** face & hand landmark models and deployed as a **Flask** web app.

🚀 **Live Demo:** [https://drowsiness-production.up.railway.app/](https://drowsiness-production.up.railway.app/)

---

## 🧠 How It Works

The app captures webcam frames in the browser and sends them to a Python backend for analysis. Each frame is processed through MediaPipe's Face Landmarker and Hand Landmarker models, and multiple drowsiness signals are computed in real time.

### Detection Signals

| Signal | Method | Alert Condition |
|---|---|---|
| **Eye closure** | EAR (Eye Aspect Ratio) + PERCLOS | >35% of frames closed over 4s window |
| **Yawning** | MediaPipe `jawOpen` blendshape / geometric MAR | Jaw open sustained for ~1.4s |
| **Head nod** | Nose-Y normalized drop from baseline | Drop >0.07 from median baseline |
| **Distraction** | Face yaw geometry (side-face detection) | Yaw >35° for 0.8s |
| **Hands off wheel** | Hand landmark centroid height vs chin | Both hands raised above chin for 1.5s |

### Calibration

On startup, the system runs a **9-second calibration phase** — it records your personal EAR and nod baselines (using median values for robustness) so thresholds adapt to your face, not a fixed number. Frames where your face is turned are automatically excluded from calibration.

---

## 🗂️ Project Structure

```
drowsiness/
├── app.py              # Flask backend — routes & API endpoints
├── detector.py         # Core DrowsinessDetector class (v4.0)
├── index.html          # Landing page
├── detect.html         # Live detector UI
├── face_landmarker.task   # MediaPipe Face Landmarker model
├── hand_landmarker.task   # MediaPipe Hand Landmarker model
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container setup
└── nixpacks.toml       # Railway deployment config
```

---

## 🔌 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/detect.html` | Live detector page |
| `POST` | `/detect` | Process a JPEG frame → returns JSON with all metrics |
| `POST` | `/reset` | Reset detector state |
| `GET` | `/status` | Health check + session stats |

### Sample `/detect` Response

```json
{
  "status": "drowsy",
  "ear": 0.2134,
  "perclos": 0.42,
  "eye_alert": true,
  "yawn_alert": false,
  "nod_alert": false,
  "distract_alert": false,
  "hand_alert": false,
  "alert": true,
  "calibrated": true,
  "ear_baseline": 0.312,
  "ear_threshold": 0.192,
  "message": "Eyes closing!"
}
```

---

## 🛠️ Local Setup

### Prerequisites

- Python 3.9+
- Webcam

### Install & Run

```bash
git clone https://github.com/KUNDANIOS/drowsiness.git
cd drowsiness

pip install -r requirements.txt

python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

> The MediaPipe model files (`face_landmarker.task`, `hand_landmarker.task`) are downloaded automatically on first run if not present (~30 MB each).

### Docker

```bash
docker build -t drowseguard .
docker run -p 5000:5000 drowseguard
```

---

## 🚢 Deployment

This project is deployed on **Railway** using `nixpacks.toml` for build config. The Flask server reads the `PORT` environment variable automatically:

```python
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
```

---

## 🧰 Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **CV / AI:** MediaPipe Face Landmarker, MediaPipe Hand Landmarker, OpenCV, SciPy
- **Frontend:** Vanilla HTML/JS (webcam capture via `getUserMedia`)
- **Deployment:** Railway (Docker / Nixpacks)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
