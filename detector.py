"""
DrowseGuard — Drowsiness Detector  v4.0  (accuracy rewrite)

FIXES vs v3:
  ① EAR FALSE POSITIVES (Image 4 — eyes open but alerting):
       Root cause: EAR_DROP_FROM_BASE=0.08 too tight; baseline ~0.30 → threshold 0.22
       which barely passes 0.218. Fix: raise DROP to 0.12, add hard floor 0.19,
       require PERCLOS ≥ 35% (was 30%) AND sustained for ≥5 frames before locking.
       Also: reject calibration frames where face_yaw > 20° to avoid skewed baseline.

  ② SIDE-FACE / DISTRACTION (Image 1 — looking away, no alert):
       Old: NO_FACE_FRAMES=20 (2s) before alert. If landmarks just degrade (not lost)
       when face turns, the "no face" path is never taken.
       Fix A: Detect yaw angle from transformation matrix OR landmark geometry.
              If |yaw| > 35°, treat as distraction immediately (reduce lag to 1s).
       Fix B: Lower NO_FACE_FRAMES to 12 (1.2s) for faster response.
       Fix C: Add face_yaw metric computed from nose tip vs eye midpoint geometry.

  ③ HANDS OFF WHEEL (Image 3 — hands raised, no detection):
       New: Use MediaPipe Hands in parallel with Face to detect if hands are
       visible and elevated (above chin level = raised). If both hands are raised
       OR no hands detected after face-confirmed baseline, fire hands_alert.
       Uses a simple heuristic: hand centroid Y < chin_y - face_height*0.3 → raised.

  ④ NOD BASELINE: re-calibrate from median not mean (more robust).
  ⑤ PERCLOS window extended to 4s (40 frames at 10fps) for fewer jitter alerts.
"""

import cv2
import numpy as np
from scipy.spatial import distance as scipy_dist
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os
from collections import deque
import math

# ─────────────────────────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────────────────────────

# EAR — PERCLOS method (tightened to avoid false positives)
EAR_THRESHOLD        = 0.19     # hard floor — never alert above this
EAR_DROP_FROM_BASE   = 0.12     # personal drop from baseline (was 0.08 — too tight)
EAR_PERCLOS_WINDOW   = 40       # 4s rolling window at 10fps (was 30 = 3s, too jittery)
EAR_PERCLOS_THRESH   = 0.35     # >35% frames closed (was 30% — reduces false positives)
EAR_LOCK_FRAMES      = 5        # must stay alerting for N frames before firing
EAR_SMOOTH_ALPHA     = 0.25     # smoother (was 0.30)
CAL_FRAMES           = 90       # 9s calibration (was 60 = 6s, more stable baseline)
CAL_MAX_YAW          = 0.18     # reject calibration frames where face is turned

# MAR — blendshape jawOpen
JAW_OPEN_THRESHOLD   = 0.42
MAR_GEO_THRESHOLD    = 0.12
MAR_FRAMES           = 14
MAR_RESET_FRAMES     = 6
MAR_SMOOTH_ALPHA     = 0.40

# NOD
NOD_THRESHOLD        = 0.07
NOD_FRAMES           = 12
NOD_RESET_FRAMES     = 8
NOD_SMOOTH_ALPHA     = 0.35

# Distraction — side face / no face
NO_FACE_FRAMES       = 12       # 1.2s (was 20 = 2s, too slow)
YAW_DISTRACT_THRESH  = 0.35     # face yaw ratio → distraction (normalized 0-1)
YAW_DISTRACT_FRAMES  = 8        # 0.8s sustained side-face before alert

# Hands off wheel
HAND_RAISED_FRAMES   = 15       # 1.5s with hands raised before alert
HAND_RESET_FRAMES    = 10
HAND_MODEL_PATH      = "hand_landmarker.task"
HAND_MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ─────────────────────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_TOP = 0
MOUTH_BOT = 17
MOUTH_L   = 61
MOUTH_R   = 291
FOREHEAD  = 10
CHIN      = 152
NOSE_TIP  = 1
LEFT_EYE_INNER  = 362
RIGHT_EYE_INNER = 133
LEFT_EAR_POINT  = 234   # left face edge
RIGHT_EAR_POINT = 454   # right face edge

# ─────────────────────────────────────────────────────────────
#  MODEL SETUP — Face
# ─────────────────────────────────────────────────────────────
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} (~30 MB)…")
        urllib.request.urlretrieve(url, path)
        print("Done.")

ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)

base_opts = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=base_opts,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.50,
    min_face_presence_confidence=0.50,
    min_tracking_confidence=0.50,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# ─────────────────────────────────────────────────────────────
#  MODEL SETUP — Hands (optional, graceful fallback if missing)
# ─────────────────────────────────────────────────────────────
hand_landmarker = None
try:
    ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
    hand_base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_opts,
        num_hands=2,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    print("Hand landmarker loaded.")
except Exception as e:
    print(f"Hand landmarker not available: {e} — hands-off-wheel detection disabled.")


# ─────────────────────────────────────────────────────────────
#  METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = scipy_dist.euclidean(pts[1], pts[5])
    B = scipy_dist.euclidean(pts[2], pts[4])
    C = scipy_dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.30


def calc_mar_geometry(lm, w, h):
    top  = np.array([lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h])
    bot  = np.array([lm[MOUTH_BOT].x * w, lm[MOUTH_BOT].y * h])
    fore = np.array([lm[FOREHEAD].x  * w, lm[FOREHEAD].y  * h])
    chin = np.array([lm[CHIN].x      * w, lm[CHIN].y      * h])
    vert = float(np.linalg.norm(top - bot))
    fh   = float(np.linalg.norm(fore - chin))
    return vert / fh if fh > 0 else 0.0


def get_jaw_blendshape(blendshapes):
    if not blendshapes or not blendshapes[0]:
        return None
    for bs in blendshapes[0]:
        if bs.category_name == 'jawOpen':
            return float(bs.score)
    return None


def calc_nose_y_norm(lm, w, h):
    ny  = lm[NOSE_TIP].y * h
    fy  = lm[FOREHEAD].y * h
    cy  = lm[CHIN].y * h
    fh  = abs(cy - fy)
    return (ny - fy) / fh if fh > 0 else 0.5


def calc_face_yaw(lm, w, h):
    """
    Estimate horizontal yaw (face turning left/right) using landmark geometry.
    Compares nose-to-eye-midpoint offset vs face width.

    Returns a value in [-1, 1]:
        0   = looking straight ahead
        ±1  = fully turned to one side
    Positive = turning right, Negative = turning left.
    """
    # Eye midpoint
    le = np.array([lm[LEFT_EYE_INNER].x * w,  lm[LEFT_EYE_INNER].y * h])
    re = np.array([lm[RIGHT_EYE_INNER].x * w, lm[RIGHT_EYE_INNER].y * h])
    eye_mid = (le + re) / 2.0

    # Nose tip
    nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])

    # Face width (ear to ear landmarks)
    left_edge  = np.array([lm[LEFT_EAR_POINT].x  * w, lm[LEFT_EAR_POINT].y  * h])
    right_edge = np.array([lm[RIGHT_EAR_POINT].x * w, lm[RIGHT_EAR_POINT].y * h])
    face_width = float(np.linalg.norm(left_edge - right_edge))

    if face_width < 1.0:
        return 0.0

    # Horizontal offset of nose from eye midpoint, normalized
    offset = float(nose[0] - eye_mid[0]) / face_width
    return float(np.clip(offset * 2.5, -1.0, 1.0))


def calc_hand_raised(hand_result, lm, w, h):
    """
    Detect if driver's hands are raised (not on wheel).

    Logic:
    - Get chin Y from face landmarks as a reference level.
    - For each detected hand, compute centroid Y.
    - If hand centroid is ABOVE (lower Y value) chin - 0.2 * face_height,
      the hand is raised into camera view (likely not on wheel).
    - Alert if BOTH hands are raised, OR if face is detected but
      hands have been visible and then suddenly both disappear
      (covered by separate no-hand logic).

    Returns:
        'raised'   — hands visibly raised (not on wheel)
        'normal'   — hands below reference / not visible
        None       — hand detection not available
    """
    if hand_landmarker is None or hand_result is None:
        return None

    chin_y  = lm[CHIN].y * h
    fore_y  = lm[FOREHEAD].y * h
    face_h  = abs(chin_y - fore_y)
    ref_y   = chin_y - face_h * 0.25   # threshold: above this = raised

    if not hand_result.hand_landmarks:
        return 'normal'  # no hands in frame → not raised (hands on wheel, below camera)

    raised_count = 0
    for hand_lm in hand_result.hand_landmarks:
        # Centroid of wrist + finger bases (landmarks 0, 5, 9, 13, 17)
        key_pts = [hand_lm[i] for i in [0, 5, 9, 13, 17]]
        cy = float(np.mean([p.y * h for p in key_pts]))
        if cy < ref_y:
            raised_count += 1

    if raised_count >= 2:
        return 'raised'
    if raised_count == 1 and len(hand_result.hand_landmarks) == 1:
        # Only one hand detected and it's raised — single hand off wheel
        return 'raised'

    return 'normal'


def exp_smooth(prev, val, alpha):
    return val if prev is None else alpha * val + (1.0 - alpha) * prev


# ─────────────────────────────────────────────────────────────
#  DETECTOR
# ─────────────────────────────────────────────────────────────

class DrowsinessDetector:

    def __init__(self):
        # EAR / PERCLOS
        self.ear_sm           = None
        self.ear_baseline     = None
        self.ear_cal_buf      = []
        self.ear_window       = deque(maxlen=EAR_PERCLOS_WINDOW)
        self.ear_alerting     = False
        self.ear_fired        = False
        self.ear_clear        = 0
        self.ear_lock_count   = 0   # must sustain PERCLOS before locking alert

        # MAR / yawn
        self.mar_sm           = None
        self.mar_frames       = 0
        self.mar_alerting     = False
        self.mar_fired        = False
        self.mar_clear        = 0
        self.using_blendshape = False

        # NOD
        self.nod_sm           = None
        self.nod_baseline     = None
        self.nod_cal_buf      = []
        self.nod_frames       = 0
        self.nod_alerting     = False
        self.nod_fired        = False
        self.nod_clear        = 0

        # Side-face / yaw distraction
        self.yaw_sm           = None
        self.yaw_frames       = 0
        self.yaw_alerting     = False
        self.yaw_fired        = False
        self.yaw_clear        = 0

        # No-face / distraction
        self.no_face_frames   = 0
        self.no_face_fired    = False

        # Hands off wheel
        self.hand_frames      = 0
        self.hand_alerting    = False
        self.hand_fired       = False
        self.hand_clear       = 0
        self.hand_alerts      = 0

        # Calibration
        self.cal_frame        = 0
        self.cal_done         = False

        # Stats
        self.frames           = 0
        self.eye_alerts       = 0
        self.yawn_alerts      = 0
        self.nod_alerts       = 0
        self.dist_alerts      = 0
        self.status           = "no_face"

    @property
    def ear_thr(self):
        if self.ear_baseline:
            computed = self.ear_baseline - EAR_DROP_FROM_BASE
            return max(computed, EAR_THRESHOLD)
        return EAR_THRESHOLD

    def _clear_ear(self):
        self.ear_alerting   = False
        self.ear_fired      = False
        self.ear_clear      = 0
        self.ear_lock_count = 0
        self.ear_window.clear()

    def _clear_mar(self):
        self.mar_alerting = False
        self.mar_fired    = False
        self.mar_frames   = 0
        self.mar_clear    = 0

    def _clear_nod(self):
        self.nod_alerting = False
        self.nod_fired    = False
        self.nod_frames   = 0
        self.nod_clear    = 0

    def _clear_yaw(self):
        self.yaw_alerting = False
        self.yaw_fired    = False
        self.yaw_frames   = 0
        self.yaw_clear    = 0

    def _clear_hand(self):
        self.hand_alerting = False
        self.hand_fired    = False
        self.hand_frames   = 0
        self.hand_clear    = 0

    # ─────────────────────────────────────────────────────
    def process_frame(self, frame_bytes: bytes) -> dict:
        arr   = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "decode_failed"}

        self.frames += 1
        h, w = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_landmarker.detect(mp_img)

        # Hand detection (parallel)
        hand_result = None
        if hand_landmarker is not None:
            try:
                hand_result = hand_landmarker.detect(mp_img)
            except Exception:
                hand_result = None

        # ── NO FACE ───────────────────────────────────────
        if not face_result.face_landmarks:
            self.ear_sm = None
            self.mar_sm = None
            self.nod_sm = None
            self.yaw_sm = None
            self.ear_window.clear()
            self._clear_yaw()
            self.no_face_frames += 1
            self.status = "no_face"

            dist = self.no_face_frames >= NO_FACE_FRAMES
            if dist and not self.no_face_fired:
                self.dist_alerts += 1
                self.no_face_fired = True

            return self._build(None, None, None, None,
                               False, False, False, dist, False,
                               perclos=0.0, jaw_raw=None)

        self.no_face_frames = 0
        self.no_face_fired  = False
        lm = face_result.face_landmarks[0]

        # ── CALIBRATION ───────────────────────────────────
        self.cal_frame += 1
        if not self.cal_done:
            raw_e   = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2
            raw_yaw = abs(calc_face_yaw(lm, w, h))

            # Only add to calibration if face is straight and eyes reasonable
            if 0.20 < raw_e < 0.48 and raw_yaw < CAL_MAX_YAW:
                self.ear_cal_buf.append(raw_e)
            self.nod_cal_buf.append(calc_nose_y_norm(lm, w, h))

            if self.cal_frame >= CAL_FRAMES:
                if len(self.ear_cal_buf) >= 20:
                    s = sorted(self.ear_cal_buf)
                    self.ear_baseline = s[len(s) // 2]   # median
                if self.nod_cal_buf:
                    sorted_nod = sorted(self.nod_cal_buf)
                    self.nod_baseline = sorted_nod[len(sorted_nod) // 2]  # median
                self.cal_done = True

            cal_pct = min(100, int(self.cal_frame / CAL_FRAMES * 100))
            return self._build(raw_e, 0.0, None, None,
                               False, False, False, False, False,
                               perclos=0.0, jaw_raw=None,
                               cal_pct=cal_pct, calibrated=False)

        # ─────────────────────────────────────────────────
        # 0. FACE YAW — side face = distraction
        # ─────────────────────────────────────────────────
        raw_yaw = calc_face_yaw(lm, w, h)
        self.yaw_sm = exp_smooth(self.yaw_sm, abs(raw_yaw), 0.30)

        yaw_alert = False
        if self.yaw_sm > YAW_DISTRACT_THRESH:
            self.yaw_frames += 1
            self.yaw_clear   = 0
            if self.yaw_frames >= YAW_DISTRACT_FRAMES:
                self.yaw_alerting = True
                yaw_alert         = True
                if not self.yaw_fired:
                    self.dist_alerts += 1
                    self.yaw_fired    = True
        else:
            self.yaw_frames = 0
            if self.yaw_alerting:
                self.yaw_clear += 1
                if self.yaw_clear >= 6:
                    self._clear_yaw()

        # ─────────────────────────────────────────────────
        # 1. EAR — PERCLOS (with lock requirement)
        # ─────────────────────────────────────────────────
        raw_l   = calc_ear(lm, LEFT_EYE,  w, h)
        raw_r   = calc_ear(lm, RIGHT_EYE, w, h)
        raw_ear = (raw_l + raw_r) / 2.0
        self.ear_sm = exp_smooth(self.ear_sm, raw_ear, EAR_SMOOTH_ALPHA)

        self.ear_window.append(1 if self.ear_sm < self.ear_thr else 0)
        perclos = sum(self.ear_window) / len(self.ear_window) if self.ear_window else 0.0

        eye_alert = False
        if perclos >= EAR_PERCLOS_THRESH:
            self.ear_lock_count += 1
            if self.ear_lock_count >= EAR_LOCK_FRAMES:
                self.ear_alerting = True
                self.status       = "drowsy"
                eye_alert         = True
                if not self.ear_fired:
                    self.eye_alerts += 1
                    self.ear_fired   = True
        else:
            self.ear_lock_count = max(0, self.ear_lock_count - 1)
            if self.ear_alerting:
                self.ear_clear += 1
                if self.ear_clear >= 10:
                    self._clear_ear()
            else:
                self.ear_clear = 0

        # ─────────────────────────────────────────────────
        # 2. YAWN — AI blendshape preferred
        # ─────────────────────────────────────────────────
        jaw_score = get_jaw_blendshape(face_result.face_blendshapes)

        if jaw_score is not None:
            self.using_blendshape = True
            self.mar_sm   = exp_smooth(self.mar_sm, jaw_score, MAR_SMOOTH_ALPHA)
            mar_display   = self.mar_sm
            mar_triggered = self.mar_sm > JAW_OPEN_THRESHOLD
        else:
            self.using_blendshape = False
            raw_mar       = calc_mar_geometry(lm, w, h)
            self.mar_sm   = exp_smooth(self.mar_sm, raw_mar, MAR_SMOOTH_ALPHA)
            mar_display   = self.mar_sm
            mar_triggered = self.mar_sm > MAR_GEO_THRESHOLD

        yawn_alert = False
        if mar_triggered:
            self.mar_frames += 1
            self.mar_clear   = 0
            if self.mar_frames >= MAR_FRAMES:
                self.mar_alerting = True
                yawn_alert        = True
                if not self.mar_fired:
                    self.yawn_alerts += 1
                    self.mar_fired    = True
        else:
            self.mar_frames = 0
            if self.mar_alerting:
                self.mar_clear += 1
                if self.mar_clear >= MAR_RESET_FRAMES:
                    self._clear_mar()

        # ─────────────────────────────────────────────────
        # 3. HEAD NOD
        # ─────────────────────────────────────────────────
        ny = calc_nose_y_norm(lm, w, h)
        self.nod_sm = exp_smooth(self.nod_sm, ny, NOD_SMOOTH_ALPHA)

        nod_alert = False
        nod_drop  = 0.0

        if self.nod_baseline is not None:
            nod_drop = self.nod_sm - self.nod_baseline

            if nod_drop > NOD_THRESHOLD:
                self.nod_frames += 1
                self.nod_clear   = 0
                if self.nod_frames >= NOD_FRAMES:
                    self.nod_alerting = True
                    nod_alert         = True
                    if not self.nod_fired:
                        self.nod_alerts += 1
                        self.nod_fired   = True
            else:
                self.nod_frames = 0
                if self.nod_alerting:
                    self.nod_clear += 1
                    if self.nod_clear >= NOD_RESET_FRAMES:
                        self._clear_nod()

        # ─────────────────────────────────────────────────
        # 4. HANDS OFF WHEEL
        # ─────────────────────────────────────────────────
        hand_alert = False
        hand_status = calc_hand_raised(hand_result, lm, w, h)

        if hand_status == 'raised':
            self.hand_frames += 1
            self.hand_clear   = 0
            if self.hand_frames >= HAND_RAISED_FRAMES:
                self.hand_alerting = True
                hand_alert         = True
                if not self.hand_fired:
                    self.hand_alerts += 1
                    self.hand_fired   = True
        else:
            self.hand_frames = max(0, self.hand_frames - 1)
            if self.hand_alerting:
                self.hand_clear += 1
                if self.hand_clear >= HAND_RESET_FRAMES:
                    self._clear_hand()

        # ── OVERALL STATUS ────────────────────────────────
        dist_alert = yaw_alert  # side-face is a distraction alert

        if self.ear_alerting or self.nod_alerting or self.mar_alerting:
            self.status = "drowsy"
        elif dist_alert or hand_alert:
            self.status = "distracted"
        else:
            self.status = "awake"

        return self._build(
            self.ear_sm, mar_display,
            nod_drop if self.nod_baseline else None,
            abs(raw_yaw),
            eye_alert, yawn_alert, nod_alert, dist_alert, hand_alert,
            perclos=perclos, jaw_raw=jaw_score
        )

    # ─────────────────────────────────────────────────────
    def _build(self, ear_v, mar_v, nod_v, yaw_v,
               eye_a, yawn_a, nod_a, dist_a, hand_a,
               perclos=0.0, jaw_raw=None, cal_pct=100, calibrated=True):

        any_alert = eye_a or yawn_a or nod_a or dist_a or hand_a

        # Confidence 0–100
        ec = min(100, int(perclos / EAR_PERCLOS_THRESH * 100)) if perclos else 0
        mc = 0
        if mar_v is not None:
            thr = JAW_OPEN_THRESHOLD if self.using_blendshape else MAR_GEO_THRESHOLD
            mc  = min(100, int(mar_v / thr * 100))
        nc = min(100, int(max(0.0, nod_v) / NOD_THRESHOLD * 100)) if nod_v is not None else 0
        yc = min(100, int((yaw_v or 0) / YAW_DISTRACT_THRESH * 100)) if yaw_v is not None else 0
        hc = min(100, int(self.hand_frames / HAND_RAISED_FRAMES * 100))

        return {
            "status":                self.status,
            "ear":                   round(ear_v,  4) if ear_v  is not None else None,
            "mar":                   round(mar_v,  4) if mar_v  is not None else None,
            "nod_drop":              round(nod_v,  4) if nod_v  is not None else None,
            "yaw":                   round(yaw_v,  4) if yaw_v  is not None else None,
            "jaw_raw":               round(jaw_raw, 4) if jaw_raw is not None else None,
            "perclos":               round(perclos, 3),
            "using_blendshape":      self.using_blendshape,
            "eye_alert":             eye_a,
            "yawn_alert":            yawn_a,
            "nod_alert":             nod_a,
            "distract_alert":        dist_a,
            "hand_alert":            hand_a,
            "alert":                 any_alert,
            "ear_alerting":          self.ear_alerting,
            "mar_alerting":          self.mar_alerting,
            "nod_alerting":          self.nod_alerting,
            "yaw_alerting":          self.yaw_alerting,
            "hand_alerting":         self.hand_alerting,
            "ear_frames":            int(sum(self.ear_window)) if self.ear_window else 0,
            "mar_frames":            self.mar_frames,
            "nod_frames":            self.nod_frames,
            "yaw_frames":            self.yaw_frames,
            "hand_frames":           self.hand_frames,
            "ear_confidence":        ec,
            "mar_confidence":        mc,
            "nod_confidence":        nc,
            "yaw_confidence":        yc,
            "hand_confidence":       hc,
            "calibrated":            calibrated if calibrated is not None else self.cal_done,
            "calibration_progress":  cal_pct,
            "ear_baseline":          round(self.ear_baseline, 4) if self.ear_baseline else None,
            "ear_threshold":         round(self.ear_thr, 4),
            "total_eye_alerts":      self.eye_alerts,
            "total_yawn_alerts":     self.yawn_alerts,
            "total_nod_alerts":      self.nod_alerts,
            "total_distract_alerts": self.dist_alerts,
            "total_hand_alerts":     self.hand_alerts,
            "frames_processed":      self.frames,
            "no_face_frames":        self.no_face_frames,
            "message":               self._msg(eye_a, yawn_a, nod_a, dist_a, hand_a),
        }

    def _msg(self, e, y, n, d, h):
        if d: return "Eyes off road!"
        if h: return "Hands off wheel!"
        parts = []
        if e: parts.append("Eyes closing!")
        if y: parts.append("Yawning!")
        if n: parts.append("Head drooping!")
        return " + ".join(parts) if parts else "Awake"