import os
import base64
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mediapipe as mp
from gesture_logic import GestureCounter

# --- CORS desde variable de entorno (o * en pruebas) ---
origins_env = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()] or ["*"]

app = FastAPI(title="Gestures API", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rutas de salud ---
@app.get("/")
def root():
    return {"status": "ok", "service": "gestures-api"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

gc = GestureCounter()

# --- Esquemas ---
class ProcessIn(BaseModel):
    session_id: str
    image_b64: str  # dataURL

class Point(BaseModel):
    x: float
    y: float

class ProcessOut(BaseModel):
    session_id: str
    counts: dict
    states: dict
    landmarks: List[Point]

# --- Util ---
def decode_dataurl(dataurl: str):
    header, b64 = dataurl.split(",", 1)
    img_bytes = base64.b64decode(b64)
    npbuf = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    return bgr

# --- Endpoint principal ---
@app.post("/process", response_model=ProcessOut)
def process(in_data: ProcessIn):
    bgr = decode_dataurl(in_data.image_b64)
    h, w = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    landmarks_out: List[Point] = []
    states = {"eye": "—", "mouth": "—", "brow": "—"}
    counts = {"blinks": 0, "mouth_opens": 0, "brow_raises": 0}

    if result.multi_face_landmarks:
        fl = result.multi_face_landmarks[0].landmark
        # Puntos en píxeles
        pts_px = np.array([[p.x * w, p.y * h] for p in fl], dtype=np.float32)

        xs, ys = pts_px[:, 0], pts_px[:, 1]
        x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        face_size = float(np.hypot(x2 - x1, y2 - y1))

        S, states = gc.process(in_data.session_id, pts_px, face_size)
        counts = {
            "blinks": S.blinks,
            "mouth_opens": S.mouth_opens,
            "brow_raises": S.brow_raises,
        }

        # Puntos normalizados para overlay
        for i in range(0, len(fl), 8):
            landmarks_out.append(Point(x=float(fl[i].x), y=float(fl[i].y)))

    return ProcessOut(
        session_id=in_data.session_id,
        counts=counts,
        states=states,
        landmarks=landmarks_out,
    )
