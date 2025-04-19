# Use TensorFlow GPU base image for better performance
FROM tensorflow/tensorflow:2.12.0-gpu

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create directories for artifacts
RUN mkdir -p artifacts/models/serving artifacts/monitoring artifacts/metrics

# Expose port
ENV PORT=5000
EXPOSE $PORT

# Set environment variables
ENV MODEL_BATCH_SIZE=32
ENV MODEL_NUM_WORKERS=4
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start Gunicorn server
CMD gunicorn --bind 0.0.0.0:$PORT \
    --workers=4 \
    --threads=4 \
    --timeout=120 \
    --worker-class=gthread \
    --worker-tmp-dir=/dev/shm \
    app:app