FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model /app/model
COPY label_mapping.json /app/label_mapping.json
COPY inference.py /app/inference.py

# Copy application files
COPY app.py /app/app.py
COPY app_fastapi.py /app/app_fastapi.py

# Expose ports for the web server
EXPOSE 7860
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/model
ENV LABEL_MAPPING_PATH=/app/label_mapping.json

# Run the application (default to Gradio app)
CMD ["python", "app.py"]
