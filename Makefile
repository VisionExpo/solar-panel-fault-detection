# ======================
# Project Makefile
# ======================

.PHONY: help install clean lint test train evaluate infer api ui docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make clean          Remove cache and build artifacts"
	@echo "  make test           Run all tests"
	@echo "  make train          Run training pipeline"
	@echo "  make evaluate       Run evaluation pipeline"
	@echo "  make infer          Run inference pipeline"
	@echo "  make api            Run FastAPI server locally"
	@echo "  make ui             Run Streamlit app"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start services via docker-compose"
	@echo "  make docker-down    Stop docker-compose services"

# ======================
# Environment
# ======================
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache build dist *.egg-info

# ======================
# Quality
# ======================
test:
	pytest -v

lint:
	@echo "Linting placeholder (add ruff/flake8 later)"

# ======================
# Pipelines
# ======================
train:
	python pipelines/train.py

evaluate:
	python pipelines/evaluate.py

infer:
	python pipelines/inference.py

# ======================
# Apps
# ======================
api:
	uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 5000 --reload

ui:
	streamlit run apps/api/streamlit_app.py

# ======================
# Docker
# ======================
docker-build:
	docker build -f deployment/docker/Dockerfile -t solar-fault-detector .

docker-up:
	docker-compose -f deployment/docker/docker-compose.yml up --build

docker-down:
	docker-compose -f deployment/docker/docker-compose.yml down
