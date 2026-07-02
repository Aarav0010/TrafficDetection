# ============================================================
# NVIDIA GPU Docker Image for Traffic Light Detection
# ============================================================
# Supports: NVIDIA Jetson (Nano/Xavier/Orin) & Desktop GPUs
# ============================================================

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app.py .
COPY yolov8n.pt .
COPY best.onnx .

# Copy video file if it exists (optional — app falls back to dashcam)
COPY edited_ultimate_video.mp4* ./

# Environment variables (can be overridden at runtime)
ENV VIDEO_FILE=edited_ultimate_video.mp4
ENV CAMERA_INDEX=0

# Expose the web server port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the app
CMD ["python3", "app.py"]
