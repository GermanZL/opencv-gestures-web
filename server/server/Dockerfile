# Use Python 3.10 to ensure mediapipe wheels are available
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (opencv + mediapipe may need these at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY server /app

# Start the service (Render provides $PORT)
CMD ["/bin/sh", "-lc", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
