.PHONY: setup test test-ui test-api test-all coverage run-api run-ui run-all clean

# Python settings
PYTHON = python
VENV = venv
PIP = $(VENV)/Scripts/pip
PYTEST = $(VENV)/Scripts/pytest
COVERAGE = $(VENV)/Scripts/coverage
STREAMLIT = $(VENV)/Scripts/streamlit
FLASK = $(VENV)/Scripts/flask

# Project settings
FLASK_APP = app.py
STREAMLIT_APP = streamlit_app.py
TEST_PATH = tests
COVERAGE_PATH = src

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

test-ui:
	$(PYTEST) $(TEST_PATH)/test_streamlit_app.py -v --cov=$(COVERAGE_PATH) --cov-report=html

test-api:
	$(PYTEST) $(TEST_PATH)/test_data_and_api.py -v --cov=$(COVERAGE_PATH) --cov-report=html

test:
	$(PYTEST) $(TEST_PATH) -v --cov=$(COVERAGE_PATH) --cov-report=html

coverage:
	$(COVERAGE) report
	$(COVERAGE) html
	@echo "HTML coverage report generated in htmlcov/"

run-api:
	$(PYTHON) $(FLASK_APP)

run-ui:
	$(STREAMLIT) run $(STREAMLIT_APP)

run-all:
	$(PYTHON) start_apps.py

clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .streamlit/secrets.toml
	rm -rf artifacts/metrics/*
	rm -rf artifacts/monitoring/*