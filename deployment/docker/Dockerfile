FROM python:3.10-slim

# ======================
# System dependencies
# ======================
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ======================
# Working directory
# ======================
WORKDIR /app

# ======================
# Install dependencies
# ======================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ======================
# Copy source code
# ======================
COPY src/ src/
COPY apps/ apps/
COPY setup.py .
COPY openapi.yaml .

RUN pip install -e .

# ======================
# Runtime configuration
# ======================
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ======================
# Expose API port
# ======================
EXPOSE 5000

# ======================
# Start FastAPI server
# ======================
CMD ["uvicorn", "apps.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "5000"]
