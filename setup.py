from setuptools import setup, find_packages
from pathlib import Path
import os

BASE_DIR = Path(__file__).parent

# Determine which requirements file to use
# Use production requirements in Docker/production environments
requirements_file = os.getenv('REQUIREMENTS_FILE', 'requirements.txt')

# Fallback to requirements-prod.txt if requirements.txt doesn't exist
req_path = BASE_DIR / requirements_file
if not req_path.exists():
    req_path = BASE_DIR / "requirements-prod.txt"

# If neither exists, use minimal requirements
if req_path.exists():
    with open(req_path, encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]
else:
    # Minimal fallback requirements for basic functionality
    requirements = [
        "tensorflow>=2.13,<3.0",
        "numpy>=1.23",
        "Pillow>=9.5",
        "fastapi>=0.100",
        "uvicorn>=0.23",
        "pydantic>=2.0",
    ]

setup(
    name="solar-fault-detector",
    version="1.0.0",
    description="End-to-end solar panel fault detection system with training, inference, and experiment tracking",
    author="Vishal Gorule",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.9",
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
