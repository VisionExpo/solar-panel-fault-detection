import os
import shutil
import argparse
import json
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy the trained model')
    parser.add_argument('--model_path', type=str, default='artifacts/models/final_model',
                        help='Path to the trained model')
    parser.add_argument('--deploy_dir', type=str, default='deployment/model',
                        help='Directory to deploy the model to')
    parser.add_argument('--label_mapping', type=str, default='deployment/label_mapping.json',
                        help='Path to the label mapping file')
    return parser.parse_args()

def deploy_model(model_path, deploy_dir, label_mapping_path):
    """
    Deploy the trained model to the deployment directory.
    
    Args:
        model_path: Path to the trained model
        deploy_dir: Directory to deploy the model to
        label_mapping_path: Path to the label mapping file
    """
    # Create deployment directory if it doesn't exist
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Copy model files to deployment directory
    print(f"Copying model from {model_path} to {deploy_dir}...")
    if os.path.exists(model_path):
        # If the model_path is a directory, copy all files
        if os.path.isdir(model_path):
            # Remove existing files in deploy_dir
            if os.path.exists(deploy_dir):
                for item in os.listdir(deploy_dir):
                    item_path = os.path.join(deploy_dir, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            
            # Copy all files from model_path to deploy_dir
            for item in os.listdir(model_path):
                s = os.path.join(model_path, item)
                d = os.path.join(deploy_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print(f"Model copied successfully to {deploy_dir}")
        else:
            # If model_path is a file, copy it directly
            shutil.copy2(model_path, deploy_dir)
            print(f"Model file copied successfully to {deploy_dir}")
    else:
        print(f"Error: Model path {model_path} does not exist")
        return False
    
    # Create label mapping if it doesn't exist
    if not os.path.exists(label_mapping_path):
        print(f"Creating label mapping file at {label_mapping_path}...")
        label_mapping = {
            "Bird-drop": 0,
            "Clean": 1,
            "Dusty": 2,
            "Electrical-damage": 3,
            "Physical-damage": 4,
            "Snow-covered": 5
        }
        
        with open(label_mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=4)
        print(f"Label mapping file created at {label_mapping_path}")
    else:
        print(f"Label mapping file already exists at {label_mapping_path}")
    
    print("Model deployment completed successfully!")
    return True

def main():
    """Main function."""
    args = parse_args()
    deploy_model(args.model_path, args.deploy_dir, args.label_mapping)

if __name__ == "__main__":
    main()
