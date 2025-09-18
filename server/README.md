# Servidor (FastAPI)

## Requisitos
- Python 3.10+

```bash
cd server
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Notas y calibración
- **OpenCV + MediaPipe**: detección de puntos. Ajusta umbrales en `gesture_logic.py`.
- Umbrales: `ear_thresh_close`, `mouth_thresh`, `brow_thresh`.
- El conteo es por sesión (cada pestaña genera un `session_id`).

## Flujo
Frontend → (fetch /process con dataURL) → Backend analiza → responde con conteos + estados + landmarks parciales.
