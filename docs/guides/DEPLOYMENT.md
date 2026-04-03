# 🚀 Deployment Guide

Complete deployment instructions for production environments.

## Deployment Options

### 1. Render Cloud (Recommended)

#### Setup

1. **Fork Repository**
   ```bash
   # Fork on GitHub
   ```

2. **Connect to Render**
   - Go to https://dashboard.render.com
   - Click "New Web Service"
   - Connect GitHub account
   - Select repository

3. **Configure Service**
   - Name: `solar-panel-detection-api`
   - Environment: `Python 3`
   - Build Command: (Read from render.yaml)
   - Start Command: (Read from render.yaml)

4. **Set Environment Variables**
   ```
   PYTHON_VERSION=3.10
   LOG_LEVEL=INFO
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Build starts automatically
   - URL provided after ~5-10 minutes

#### Verification

```bash
# Health check
curl https://<your-render-app>.onrender.com/health

# Test prediction
curl -X POST https://<your-render-app>.onrender.com/predict \
  -F "file=@test_image.jpg"
```

---

### 2. Docker Deployment

#### Build Image

```bash
# Build locally
docker build -t solar-panel-detector:latest .

# Build for production
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  -t solar-panel-detector:v1.0 .
```

#### Run Container

**Development**:
```bash
docker run -p 5000:5000 solar-panel-detector:latest
```

**Production**:
```bash
docker run \
  -d \
  --name solar-detector \
  -p 5000:5000 \
  --restart unless-stopped \
  --health-cmd='curl -f http://localhost:5000/health || exit 1' \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  solar-panel-detector:v1.0
```

#### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

### 3. Kubernetes Deployment

#### Create deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: solar-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: solar-detector
  template:
    metadata:
      labels:
        app: solar-detector
    spec:
      containers:
      - name: solar-detector
        image: your-registry/solar-detector:v1.0
        ports:
        - containerPort: 5000
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### Deploy

```bash
kubectl apply -f deployment.yaml
```

---

### 4. AWS (Amazon Web Services)

#### Using EC2

1. **Launch Instance**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance: t3.medium (minimum)
   - Security: Allow port 5000

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3.10 python3-pip git
   ```

3. **Clone & Deploy**
   ```bash
   git clone https://github.com/VisionExpo/solar-panel-fault-detection.git
   cd solar-panel-fault-detection
   pip install -r requirements-prod.txt
   pip install -e .
   python -m uvicorn apps.api.fastapi_app:app --host 0.0.0.0
   ```

#### Using AWS Lambda (Serverless)

1. **Create Function**
   - Runtime: Python 3.10
   - Handler: `apps.api.fastapi_app.app`

2. **Install Dependencies**
   - Package local dependencies with code
   - Add to Lambda layer

3. **Deploy**
   ```bash
   sam package
   sam deploy
   ```

---

### 5. Google Cloud Platform (GCP)

#### Cloud Run (Serverless)

```bash
# Build and deploy
gcloud run deploy solar-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --timeout 600 \
  --allow-unauthenticated
```

#### Cloud App Engine

```bash
# Deploy
gcloud app deploy
```

---

### 6. Azure

#### Container Instances

```bash
# Build image
docker build -t solar-detector .

# Push to registry
docker tag solar-detector myregistry.azurecr.io/solar-detector:latest
docker push myregistry.azurecr.io/solar-detector:latest

# Deploy
az container create \
  --resource-group mygroup \
  --name solar-detector \
  --image myregistry.azurecr.io/solar-detector:latest \
  --port 5000 \
  --cpu 1 \
  --memory 1
```

#### App Service

```bash
# Create app
az webapp create --resource-group mygroup --plan myplan --name my-app

# Deploy
az webapp deployment source config --resource-group mygroup \
  --name my-app --repo-url https://github.com/VisionExpo/solar-detector \
  --branch main --manual-integration
```

---

## Performance Optimization

### API Scaling

**Horizontal Scaling**:
```bash
# Render: Set to "Standard" plan for auto-scaling
# Docker: Use load balancer (Nginx, HAProxy)
# Kubernetes: Horizontal Pod Autoscaler
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_model():
    return tf.keras.models.load_model(model_path)
```

### Monitoring

Set up health checks and monitoring:
- Response time alerts
- Error rate monitoring
- Resource usage tracking

---

## Security Best Practices

### 1. Input Validation
- ✅ File type validation (JPG, PNG)
- ✅ File size limits
- ✅ Dimension checks

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile):
    ...
```

### 3. HTTPS/SSL

- Use SSL certificates
- Force HTTPS redirect
- Set secure cookies

### 4. API Authentication

```python
from fastapi import HTTPException, Header

@app.post("/predict")
async def predict(
    file: UploadFile,
    x_token: str = Header(...)
):
    if x_token != valid_token:
        raise HTTPException(status_code=403)
    ...
```

---

## Monitoring & Logging

### Application Logs

```python
import logging

logger = logging.getLogger(__name__)
logger.info("API started")
logger.error("Prediction failed")
```

### Monitoring Services

- **Datadog**: APM and monitoring
- **New Relic**: Performance monitoring
- **ELK Stack**: Log aggregation
- **Prometheus**: Metrics collection

---

## Troubleshooting

### API Not Responding

```bash
# Check health
curl http://localhost:5000/health

# Check logs
docker logs solar-detector

# Restart container
docker restart solar-detector
```

### Model Download Fails

```bash
# Manual download
python -m solar_fault_detector.utils.download_model

# Check permissions
ls -la artifacts/models/
```

### Out of Memory

```bash
# Increase instance size
# Reduce batch size
# Clear logs periodically
```

---

## Cost Optimization

| Platform | Estimated Cost | Notes |
|----------|---|---|
| Render (Free) | $0 | 512MB RAM, auto-suspend |
| Render (Standard) | $7/month | Persistent uptime |
| AWS Lambda | $0.20/million | Pay per request |
| GCP Cloud Run | $0.40/month | Free tier available |
| Azure Container | $0.00125/hour | Pay per container-hour |

---

## Rollback Procedure

### Render

1. Go to Dashboard
2. Select deployment
3. Click "Previous Deployments"
4. Select previous version
5. Click "Deploy"

### Docker

```bash
# Tag specific version
docker tag solar-detector:latest solar-detector:v1.0

# Rollback to previous
docker run -d solar-detector:v0.9
```

