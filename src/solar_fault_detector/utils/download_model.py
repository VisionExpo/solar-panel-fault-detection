"""
Model download utility for Render deployment.
Downloads model from Hugging Face Model Hub during build.
"""
from pathlib import Path
import os
import sys


def download_model_from_huggingface(
    repo_id: str = "VishalGorule09/SolarPanelModel",
    model_filename: str = "best_model.h5",
    cache_dir: Path = Path("artifacts/models"),
) -> Path:
    """
    Download model from Hugging Face Model Hub.
    
    Args:
        repo_id: Hugging Face model repository ID
        model_filename: Name of the model file to download
        cache_dir: Directory to save the model
        
    Returns:
        Path to the downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface-hub not installed. Install with: pip install huggingface-hub")
        sys.exit(1)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading model from Hugging Face: {repo_id}/{model_filename}")
    
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        cache_dir=str(cache_dir),
        force_filename=model_filename,
    )
    
    model_path = Path(model_path)
    print(f"✅ Model downloaded successfully to: {model_path}")
    return model_path


def ensure_model_exists(model_path: Path, repo_id: str = "VishalGorule09/SolarPanelModel") -> Path:
    """
    Ensure model file exists. Download if missing.
    
    Args:
        model_path: Expected path to model file
        repo_id: Hugging Face repository ID
        
    Returns:
        Path to the model file
        
    Raises:
        FileNotFoundError: If model cannot be downloaded
    """
    if model_path.exists():
        print(f"✅ Model found at: {model_path}")
        return model_path
    
    print(f"⚠️  Model not found at: {model_path}")
    model_dir = model_path.parent
    model_filename = model_path.name
    
    return download_model_from_huggingface(
        repo_id=repo_id,
        model_filename=model_filename,
        cache_dir=model_dir,
    )


if __name__ == "__main__":
    # This script can be run standalone during Docker build
    model_path = ensure_model_exists(
        Path("artifacts/models/best_model.h5")
    )
    print(f"Model ready at: {model_path}")
