from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

# ===== Landmarks (MediaPipe FaceMesh) =====
# Ojos (EAR)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Boca (apertura vertical)
UPPER_LIP = 13
LOWER_LIP = 14

# Cejas vs párpado superior (usamos ambos lados)
# Izquierda: ceja alta ~105, párpado sup ~159
# Derecha:   ceja alta ~334, párpado sup ~386
BROW_TOP_L = 105
EYELID_TOP_L = 159
BROW_TOP_R = 334
EYELID_TOP_R = 386


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_ear(pts: np.ndarray, idx):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (euclid(p2, p6) + euclid(p3, p5)) / (2.0 * euclid(p1, p4) + 1e-9)


@dataclass
class SessionState:
    # Contadores
    blinks: int = 0
    mouth_opens: int = 0
    brow_raises: int = 0
    # Estados instantáneos
    eye_closed: bool = False
    mouth_open: bool = False
    brow_high: bool = False
    # ---- Calibración / suavizado para cejas ----
    frames_seen: int = 0
    brow_base_l: float = 0.0
    brow_base_r: float = 0.0
    brow_ma_l: float = 0.0  # media exponencial (EMA) izq
    brow_ma_r: float = 0.0  # EMA der
    calibrated: bool = False


class GestureCounter:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_state(self, sid: str) -> SessionState:
        if sid not in self.sessions:
            self.sessions[sid] = SessionState()
        return self.sessions[sid]

    def process(self, sid: str, pts_px: np.ndarray, face_size: float) -> Tuple[SessionState, Dict[str, str]]:
        """
        pts_px: landmarks en PIXELES (importante)
        face_size: escala de rostro en PIXELES (diagonal de la bbox)
        """
        S = self.get_state(sid)

        # =========== OJOS (parpadeo) ===========
        ear_l = eye_ear(pts_px, LEFT_EYE)
        ear_r = eye_ear(pts_px, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0
        EAR_CLOSE = 0.21
        if ear < EAR_CLOSE and not S.eye_closed:
            S.eye_closed = True
        elif ear >= EAR_CLOSE and S.eye_closed:
            S.eye_closed = False
            S.blinks += 1

        # =========== BOCA (apertura) ===========
        mouth_dist = euclid(pts_px[UPPER_LIP], pts_px[LOWER_LIP])
        mouth_ratio = mouth_dist / (face_size + 1e-9)
        MOUTH_OPEN = 0.030
        MOUTH_OPEN_HYST = 0.025
        if not S.mouth_open and mouth_ratio > MOUTH_OPEN:
            S.mouth_open = True
            S.mouth_opens += 1
        elif S.mouth_open and mouth_ratio < MOUTH_OPEN_HYST:
            S.mouth_open = False

        # =========== CEJAS (alza) con baseline + EMA ===========
        # Distancias ceja–párpado (en píxeles) normalizadas por face_size para robustez a escala
        brow_l = euclid(pts_px[BROW_TOP_L], pts_px[EYELID_TOP_L]) / (face_size + 1e-9)
        brow_r = euclid(pts_px[BROW_TOP_R], pts_px[EYELID_TOP_R]) / (face_size + 1e-9)

        # Calibración inicial: 12 frames (~0.7s si analizas ~17 fps) en "neutro"
        CALIB_FRAMES = 12
        if not S.calibrated:
            S.frames_seen += 1
            # EMA rápida durante calibración para evitar valores raros
            alpha_cal = 0.4
            if S.frames_seen == 1:
                S.brow_ma_l = brow_l
                S.brow_ma_r = brow_r
            else:
                S.brow_ma_l = alpha_cal * brow_l + (1 - alpha_cal) * S.brow_ma_l
                S.brow_ma_r = alpha_cal * brow_r + (1 - alpha_cal) * S.brow_ma_r

            if S.frames_seen >= CALIB_FRAMES:
                S.brow_base_l = S.brow_ma_l
                S.brow_base_r = S.brow_ma_r
                S.calibrated = True
        else:
            # EMA más lenta en operación normal
            alpha = 0.25
            S.brow_ma_l = alpha * brow_l + (1 - alpha) * S.brow_ma_l
            S.brow_ma_r = alpha * brow_r + (1 - alpha) * S.brow_ma_r

            # Umbral relativo: +18% sobre baseline (con histéresis al 12%)
            RAISE_PCT = 1.18
            RAISE_PCT_HYST = 1.12

            # Lado izquierdo
            left_high = S.brow_ma_l > (max(S.brow_base_l, 1e-6) * RAISE_PCT)
            left_low  = S.brow_ma_l < (max(S.brow_base_l, 1e-6) * RAISE_PCT_HYST)
            # Lado derecho
            right_high = S.brow_ma_r > (max(S.brow_base_r, 1e-6) * RAISE_PCT)
            right_low  = S.brow_ma_r < (max(S.brow_base_r, 1e-6) * RAISE_PCT_HYST)

            # Disparar si cualquiera sube
            if not S.brow_high and (left_high or right_high):
                S.brow_high = True
                S.brow_raises += 1
            # Volver a normal si ambos bajan
            elif S.brow_high and (left_low and right_low):
                S.brow_high = False

            # Opcional: ajuste lento del baseline (seguir al usuario)
            # Útil si la cabeza se acomoda distinto con el tiempo
            BASELINE_LR = 0.002  # muy lento
            S.brow_base_l = (1 - BASELINE_LR) * S.brow_base_l + BASELINE_LR * S.brow_ma_l
            S.brow_base_r = (1 - BASELINE_LR) * S.brow_base_r + BASELINE_LR * S.brow_ma_r

        states = {
            'eye': 'cerrado' if S.eye_closed else 'abierto',
            'mouth': 'abierta' if S.mouth_open else 'cerrada',
            'brow': 'alta' if S.brow_high else ('calibrando' if not S.calibrated else 'normal')
        }
        return S, states
