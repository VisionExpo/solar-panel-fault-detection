from setuptools import setup, find_packages

setup(
    name="solar_panel_detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.12.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mlflow>=2.7.0",
        "wandb>=0.15.0",
        "python-dotenv>=1.0.0",
        "albumentations>=1.3.1",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.65.0",
        "flask>=2.3.0",
        "gunicorn>=21.2.0",
        "pytest>=7.4.0",
        "optuna>=3.0.0",
        "tensorflow-addons>=0.21.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning system for detecting faults in solar panels",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="deep-learning, computer-vision, solar-panels, fault-detection",
    url="https://github.com/yourusername/solar-panel-fault-detection",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/solar-panel-fault-detection/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)