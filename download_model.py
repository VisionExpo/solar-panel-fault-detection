import os
import sys
import requests
from pathlib import Path
import shutil

def download_model_from_huggingface(repo_id, filename, output_path):
    """
    Download a model file from Hugging Face Model Hub.
    
    Args:
        repo_id (str): The Hugging Face repository ID (username/repo_name)
        filename (str): The filename in the repository
        output_path (str): Path where the file should be saved
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Construct the URL
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    print(f"Downloading model from {url}...")
    
    # Download the file
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"File size: {total_size_mb:.2f} MB")
        
        # Write the file
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size_mb:.1f} MB)")
                sys.stdout.flush()
        
        print("\nDownload complete!")
        
        # Verify the file was downloaded
        if os.path.exists(output_path):
            print(f"✅ Model file downloaded successfully to {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"❌ Failed to download model file to {output_path}")
            return False
    else:
        print(f"❌ Failed to download model file. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def download_model_from_url(url, output_path):
    """
    Download a model file from a direct URL.
    
    Args:
        url (str): The direct download URL
        output_path (str): Path where the file should be saved
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download the file
    print(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"File size: {total_size_mb:.2f} MB")
        
        # Write the file
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size_mb:.1f} MB)")
                sys.stdout.flush()
        
        print("\nDownload complete!")
        
        # Verify the file was downloaded
        if os.path.exists(output_path):
            print(f"✅ Model file downloaded successfully to {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"❌ Failed to download model file to {output_path}")
            return False
    else:
        print(f"❌ Failed to download model file. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    """Main function to download the model."""
    # Define the model file paths
    model_dir = Path("deployment/model/variables")
    render_model_dir = Path("render_deployment/model/variables")
    
    # Create directories if they don't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    render_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the file paths
    model_file = model_dir / "variables.data-00000-of-00001"
    render_model_file = render_model_dir / "variables.data-00000-of-00001"
    
    # Check if model files already exist
    if model_file.exists() and render_model_file.exists():
        print("Model files already exist. Skipping download.")
        return True
    
    # Hugging Face repository information
    repo_id = "VishalGorule09/SolarPanelModel"
    filename = "variables.data-00000-of-00001"
    
    # Download from Hugging Face
    success = download_model_from_huggingface(repo_id, filename, str(model_file))
    
    if not success:
        # Try alternative URL if Hugging Face fails
        url = os.environ.get("MODEL_URL", "")
        if url:
            print(f"Trying alternative URL from environment variable: {url}")
            success = download_model_from_url(url, str(model_file))
        
        if not success:
            print("❌ Failed to download model file from all sources.")
            return False
    
    # Copy the file from one location to another if needed
    if model_file.exists() and not render_model_file.exists():
        print(f"Copying model file from {model_file} to {render_model_file}...")
        shutil.copy2(model_file, render_model_file)
        print(f"✅ Model file copied successfully to {render_model_file}")
    
    if render_model_file.exists() and not model_file.exists():
        print(f"Copying model file from {render_model_file} to {model_file}...")
        shutil.copy2(render_model_file, model_file)
        print(f"✅ Model file copied successfully to {model_file}")
    
    return True

if __name__ == "__main__":
    if main():
        print("Model download completed successfully.")
    else:
        print("Model download failed.")
        sys.exit(1)
