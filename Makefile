.PHONY: setup clean train evaluate test docker-build docker-run

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .

clean:
	rm -rf __pycache__
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf mlruns
	rm -rf wandb
	rm -rf logs/*.log

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

test:
	pytest tests/ -v --cov=src

docker-build:
	docker build -t solar-panel-detector .

docker-run:
	docker run -p 5000:10000 solar-panel-detector