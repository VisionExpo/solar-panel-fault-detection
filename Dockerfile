FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY apps/ apps/
COPY setup.py .
COPY openapi.yaml .

# 🔑 THIS IS THE FIX
ENV PYTHONPATH=/app/src

# Install package
RUN pip install -e .

EXPOSE 5000

CMD ["uvicorn", "apps.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "5000"]
