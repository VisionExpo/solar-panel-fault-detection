FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Create necessary directories
RUN mkdir -p artifacts/models

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application
CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 120 --workers 4 app:app