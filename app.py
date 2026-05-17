"""
DrowseGuard — Flask Backend
Routes:
  GET  /              → index.html  (landing page)
  GET  /detect.html   → detect.html (live detector)
  GET  /detector      → detect.html (alias — works from old links too)
  POST /detect        → process frame, return JSON
  POST /reset         → reset detector state
  GET  /status        → health check JSON
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from detector import DrowsinessDetector
import os

app = Flask(__name__)
CORS(app)

detector = DrowsinessDetector()

# ─────────────────────────────────────────────
#  SERVE PAGES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page"""
    return send_from_directory(".", "index.html")


@app.route("/detect.html")
def detect_page():
    """Live detector page"""
    return send_from_directory(".", "detect.html")


@app.route("/detector")
def detector_alias():
    """Alias — in case any link still uses /detector"""
    return send_from_directory(".", "detect.html")


@app.route("/detector.html")
def live_detector_alias():
    """Alias — in case any old link uses /detector.html"""
    return send_from_directory(".", "detect.html")


# ─────────────────────────────────────────────
#  CORE DETECTION
# ─────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """
    Receives a JPEG frame as raw bytes in request body.
    Returns full JSON with EAR, MAR, nod_drop, alerts, totals.
    """
    frame_bytes = request.data
    if not frame_bytes:
        return jsonify({"error": "No frame received"}), 400
    result = detector.process_frame(frame_bytes)
    return jsonify(result)


# ─────────────────────────────────────────────
#  RESET
# ─────────────────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset():
    global detector
    detector = DrowsinessDetector()
    return jsonify({"message": "Detector reset successfully"})


# ─────────────────────────────────────────────
#  STATUS / HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "server":           "running",
        "frames_processed": detector.frames_processed,
        "current_status":   detector.status,
        "total_eye_alerts": detector.total_eye_alerts,
        "total_yawn_alerts": detector.total_yawn_alerts,
        "total_nod_alerts": detector.total_nod_alerts,
    })


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n==========================================")
    print("  DrowseGuard Server running!")
    print("  Landing page : http://localhost:5000")
    print("  Detector     : http://localhost:5000/detect.html")
    print("  API status   : http://localhost:5000/status")
    print("==========================================\n")
    app.run(host="0.0.0.0", port=5000, debug=True)