import os
import shutil
import json
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare model for deployment on Render')
    parser.add_argument('--model_path', type=str, default='deployment/model',
                        help='Path to the model directory')
    parser.add_argument('--output_dir', type=str, default='render_deployment',
                        help='Output directory for Render deployment files')
    return parser.parse_args()

def create_render_yaml(output_dir):
    """Create render.yaml file for Render deployment."""
    render_yaml = """
services:
  # Web service
  - type: web
    name: solar-panel-fault-detector
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 7860
    healthCheckPath: /
    """
    
    with open(os.path.join(output_dir, 'render.yaml'), 'w') as f:
        f.write(render_yaml)
    
    print(f"Created render.yaml in {output_dir}")

def create_procfile(output_dir):
    """Create Procfile for Render deployment."""
    with open(os.path.join(output_dir, 'Procfile'), 'w') as f:
        f.write("web: python app.py")
    
    print(f"Created Procfile in {output_dir}")

def create_runtime_txt(output_dir):
    """Create runtime.txt file for Render deployment."""
    with open(os.path.join(output_dir, 'runtime.txt'), 'w') as f:
        f.write("python-3.9.0")
    
    print(f"Created runtime.txt in {output_dir}")

def copy_deployment_files(model_path, output_dir):
    """Copy deployment files to output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model directory in output directory
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    
    # Copy model files
    if os.path.exists(model_path):
        print(f"Copying model from {model_path} to {output_dir}/model...")
        
        # If model_path is a directory, copy all files
        if os.path.isdir(model_path):
            for item in os.listdir(model_path):
                s = os.path.join(model_path, item)
                d = os.path.join(output_dir, 'model', item)
                
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        else:
            # If model_path is a file, copy it directly
            shutil.copy2(model_path, os.path.join(output_dir, 'model'))
    else:
        print(f"Warning: Model path {model_path} does not exist")
    
    # Copy deployment files
    deployment_dir = 'deployment'
    if os.path.exists(deployment_dir):
        print(f"Copying deployment files from {deployment_dir} to {output_dir}...")
        
        # Copy specific files
        files_to_copy = [
            'app.py',
            'app_fastapi.py',
            'inference.py',
            'label_mapping.json',
            'requirements.txt'
        ]
        
        for file in files_to_copy:
            src = os.path.join(deployment_dir, file)
            dst = os.path.join(output_dir, file)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"Warning: {src} does not exist")
    else:
        print(f"Warning: Deployment directory {deployment_dir} does not exist")
    
    # Copy visualization files
    vis_dir = 'visualization/static'
    if os.path.exists(vis_dir):
        print(f"Copying visualization files from {vis_dir} to {output_dir}/static...")
        
        # Create static directory in output directory
        os.makedirs(os.path.join(output_dir, 'static'), exist_ok=True)
        
        # Copy all files
        for item in os.listdir(vis_dir):
            s = os.path.join(vis_dir, item)
            d = os.path.join(output_dir, 'static', item)
            
            if os.path.isfile(s):
                shutil.copy2(s, d)
                print(f"Copied {s} to {d}")
    else:
        print(f"Warning: Visualization directory {vis_dir} does not exist")

def update_app_paths(output_dir):
    """Update paths in app.py to work with Render deployment."""
    app_path = os.path.join(output_dir, 'app.py')
    
    if os.path.exists(app_path):
        print(f"Updating paths in {app_path}...")
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Update model path
        content = content.replace(
            'MODEL_PATH = os.environ.get("MODEL_PATH", "deployment/model")',
            'MODEL_PATH = os.environ.get("MODEL_PATH", "model")'
        )
        
        # Update label mapping path
        content = content.replace(
            'LABEL_MAPPING_PATH = os.environ.get("LABEL_MAPPING_PATH", "deployment/label_mapping.json")',
            'LABEL_MAPPING_PATH = os.environ.get("LABEL_MAPPING_PATH", "label_mapping.json")'
        )
        
        # Update visualization path
        content = content.replace(
            'vis_path = os.path.join(project_root, "visualization", "static")',
            'vis_path = os.path.join(os.path.dirname(__file__), "static")'
        )
        
        # Update import statement
        content = content.replace(
            'from inference import SolarPanelFaultDetector',
            'from inference import SolarPanelFaultDetector'
        )
        
        with open(app_path, 'w') as f:
            f.write(content)
        
        print(f"Updated paths in {app_path}")
    else:
        print(f"Warning: {app_path} does not exist")

def update_fastapi_paths(output_dir):
    """Update paths in app_fastapi.py to work with Render deployment."""
    app_path = os.path.join(output_dir, 'app_fastapi.py')
    
    if os.path.exists(app_path):
        print(f"Updating paths in {app_path}...")
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Update model path
        content = content.replace(
            'MODEL_PATH = os.environ.get("MODEL_PATH", "deployment/model")',
            'MODEL_PATH = os.environ.get("MODEL_PATH", "model")'
        )
        
        # Update label mapping path
        content = content.replace(
            'LABEL_MAPPING_PATH = os.environ.get("LABEL_MAPPING_PATH", "deployment/label_mapping.json")',
            'LABEL_MAPPING_PATH = os.environ.get("LABEL_MAPPING_PATH", "label_mapping.json")'
        )
        
        with open(app_path, 'w') as f:
            f.write(content)
        
        print(f"Updated paths in {app_path}")
    else:
        print(f"Warning: {app_path} does not exist")

def create_readme(output_dir):
    """Create README.md file for Render deployment."""
    readme = """# Solar Panel Fault Detector - Render Deployment

This repository contains the files needed to deploy the Solar Panel Fault Detector on Render.

## Deployment Instructions

1. Create a new Web Service on Render.
2. Connect your GitHub repository.
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment Variables**:
     - `PORT`: 7860
     - `PYTHON_VERSION`: 3.9.0

## API Endpoints

The application provides both a web interface and a REST API:

- Web Interface: `https://your-app-name.onrender.com`
- REST API: `https://your-app-name.onrender.com/api`

## API Documentation

When the application is running, you can access the API documentation at:
`https://your-app-name.onrender.com/docs`

## Files

- `app.py`: Gradio web application
- `app_fastapi.py`: FastAPI server
- `inference.py`: Core inference module
- `model/`: Directory containing the trained model
- `label_mapping.json`: Mapping of class indices to class names
- `static/`: Directory containing visualizations
- `requirements.txt`: Python dependencies
- `render.yaml`: Render deployment configuration
- `Procfile`: Process file for Render
- `runtime.txt`: Python runtime version
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme)
    
    print(f"Created README.md in {output_dir}")

def main():
    """Main function."""
    args = parse_args()
    
    print(f"Preparing model for deployment on Render...")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Copy deployment files
    copy_deployment_files(args.model_path, args.output_dir)
    
    # Update paths in app.py
    update_app_paths(args.output_dir)
    
    # Update paths in app_fastapi.py
    update_fastapi_paths(args.output_dir)
    
    # Create Render deployment files
    create_render_yaml(args.output_dir)
    create_procfile(args.output_dir)
    create_runtime_txt(args.output_dir)
    create_readme(args.output_dir)
    
    print(f"Model prepared for deployment on Render in {args.output_dir}")
    print(f"You can now deploy this directory to Render.")

if __name__ == "__main__":
    main()
