# Documentation Index

Complete documentation for the Solar Panel Fault Detection System.

## 📚 Table of Contents

### Getting Started
- **[Setup & Installation](guides/SETUP.md)** - Environment setup, dependencies, and installation
- **[Quick Start](guides/QUICKSTART.md)** - Get up and running in 5 minutes

### Architecture & Design
- **[System Architecture](architecture/ARCHITECTURE.md)** - High-level system design and components
- **[Data Pipeline](architecture/DATA_PIPELINE.md)** - Data flow and preprocessing
- **[Model Architecture](architecture/MODEL_ARCHITECTURE.md)** - Neural network design and layers

### API Documentation
- **[REST API Guide](api/REST_API.md)** - API endpoints, usage, and examples
- **[Python API](api/PYTHON_API.md)** - Python module imports and class references

### Guides & Tutorials
- **[Training Guide](guides/TRAINING.md)** - How to train models locally
- **[Deployment Guide](guides/DEPLOYMENT.md)** - Deploy to Render, Docker, cloud
- **[Configuration Guide](guides/CONFIGURATION.md)** - Customize settings and hyperparameters
- **[Troubleshooting](guides/TROUBLESHOOTING.md)** - Common issues and solutions

### Examples
- **[Local Inference](examples/LOCAL_INFERENCE.md)** - Python example code
- **[API Usage](examples/API_USAGE.md)** - curl and Python HTTP examples
- **[Batch Processing](examples/BATCH_PROCESSING.md)** - Process multiple images

### Development
- **[Contributing Guidelines](../CONTRIBUTING.md)** - How to contribute
- **[Code Standards](../CODE_STANDARDS.md)** - Code quality and style guidelines

---

## 🎯 Quick Links

| Task | Link |
|------|------|
| Install & run locally | [Setup Guide](guides/SETUP.md) |
| Understand the system | [Architecture](architecture/ARCHITECTURE.md) |
| Use the API | [REST API](api/REST_API.md) |
| Deploy to production | [Deployment Guide](guides/DEPLOYMENT.md) |
| Debug issues | [Troubleshooting](guides/TROUBLESHOOTING.md) |

---

## 📂 Project Structure

```
solar-panel-fault-detection/
├── docs/                          # This folder
│   ├── api/                       # API documentation
│   ├── guides/                    # User and developer guides
│   ├── architecture/              # Technical architecture docs
│   └── examples/                  # Code examples and tutorials
├── src/solar_fault_detector/      # Main package
│   ├── config/                    # Configuration management
│   ├── models/                    # Model architectures
│   ├── training/                  # Training pipelines
│   ├── inference/                 # Inference engines
│   ├── monitoring/                # Experiment tracking
│   ├── utils/                     # Utilities and helpers
│   └── data/                      # Data loading and preprocessing
├── apps/                          # Applications
│   ├── api/                       # FastAPI application
│   └── cli/                       # Command-line tools
├── tests/                         # Unit and integration tests
└── deployment/                    # Deployment configurations
```

---

## 🚀 Features

- ✅ Real-time fault detection from solar panel images
- ✅ REST API for easy integration
- ✅ Multiple model architectures (CNN, Ensemble)
- ✅ Experiment tracking with Weights & Biases
- ✅ Docker containerization
- ✅ Production-ready deployment
- ✅ Comprehensive monitoring and logging

---

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Check [Troubleshooting Guide](guides/TROUBLESHOOTING.md)
- **Contributing**: See [Contributing Guidelines](../CONTRIBUTING.md)

