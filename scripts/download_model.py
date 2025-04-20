import os
import sys
import gdown
import requests
from pathlib import Path

def download_model_from_gdrive(file_id, output_path):
    """
    Download a model file from Google Drive.
    
    Args:
        file_id (str): The Google Drive file ID
        output_path (str): Path where the file should be saved
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download the file
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    
    # Verify the file was downloaded
    if os.path.exists(output_path):
        print(f"✅ Model file downloaded successfully to {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    else:
        print(f"❌ Failed to download model file to {output_path}")
        sys.exit(1)

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
        block_size = 1024  # 1 Kibibyte
        
        # Write the file
        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        
        # Verify the file was downloaded
        if os.path.exists(output_path):
            print(f"✅ Model file downloaded successfully to {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        else:
            print(f"❌ Failed to download model file to {output_path}")
            sys.exit(1)
    else:
        print(f"❌ Failed to download model file. Status code: {response.status_code}")
        sys.exit(1)

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
        return
    
    # Replace this with your actual Google Drive file ID or direct download URL
    # Example for Google Drive:
    # file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
    # download_model_from_gdrive(file_id, str(model_file))
    
    # Example for direct URL:
    # url = "YOUR_DIRECT_DOWNLOAD_URL"
    # download_model_from_url(url, str(model_file))
    
    # For demonstration purposes, we'll use a placeholder
    print("⚠️ IMPORTANT: You need to upload your model file to Google Drive or another storage service")
    print("   and update this script with the correct file ID or URL.")
    print("   For Google Drive:")
    print("   1. Upload your model file to Google Drive")
    print("   2. Share the file (make it accessible via link)")
    print("   3. Get the file ID from the share link (the part after 'id=' in the URL)")
    print("   4. Update this script with the file ID")
    
    # Copy the file from one location to another if it exists in only one place
    if model_file.exists() and not render_model_file.exists():
        print(f"Copying model file from {model_file} to {render_model_file}...")
        import shutil
        shutil.copy2(model_file, render_model_file)
        print(f"✅ Model file copied successfully to {render_model_file}")
    
    if render_model_file.exists() and not model_file.exists():
        print(f"Copying model file from {render_model_file} to {model_file}...")
        import shutil
        shutil.copy2(render_model_file, model_file)
        print(f"✅ Model file copied successfully to {model_file}")

if __name__ == "__main__":
    main()
