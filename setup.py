from setuptools import setup, find_packages
from pathlib import Path

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f.readlines()
        if line.strip() and not line.startswith("#")
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
