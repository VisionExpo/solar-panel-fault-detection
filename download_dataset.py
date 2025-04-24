import os
import sys
import zipfile
import shutil
import kaggle
from pathlib import Path

def download_kaggle_dataset(dataset_name, output_dir):
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_name: Name of the dataset on Kaggle (e.g., 'gitenavnath/solar-augmented-dataset')
        output_dir: Directory to save the dataset
    """
    print(f"Downloading dataset {dataset_name} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"Dataset downloaded successfully to {output_dir}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def organize_dataset(dataset_dir, output_dir):
    """
    Organize the dataset into train, validation, and test sets.
    
    Args:
        dataset_dir: Directory containing the downloaded dataset
        output_dir: Directory to save the organized dataset
    """
    print(f"Organizing dataset from {dataset_dir} to {output_dir}...")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class directories
    classes = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-damage', 'Snow-covered']
    
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # Check if the dataset has a specific structure
    if os.path.exists(os.path.join(dataset_dir, 'train')) and os.path.exists(os.path.join(dataset_dir, 'test')):
        print("Dataset already has train/test split. Using existing structure.")
        
        # Copy train data
        for class_name in classes:
            src_dir = os.path.join(dataset_dir, 'train', class_name)
            if os.path.exists(src_dir):
                for file in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, file)
                    dst_file = os.path.join(train_dir, class_name, file)
                    shutil.copy2(src_file, dst_file)
        
        # Copy test data
        for class_name in classes:
            src_dir = os.path.join(dataset_dir, 'test', class_name)
            if os.path.exists(src_dir):
                for file in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, file)
                    dst_file = os.path.join(test_dir, class_name, file)
                    shutil.copy2(src_file, dst_file)
    else:
        print("Dataset doesn't have a standard structure. Organizing manually.")
        
        # Look for class directories
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.exists(class_dir):
                files = os.listdir(class_dir)
                num_files = len(files)
                
                # Split: 70% train, 15% validation, 15% test
                train_split = int(0.7 * num_files)
                val_split = int(0.85 * num_files)
                
                # Copy files to train, validation, and test directories
                for i, file in enumerate(files):
                    src_file = os.path.join(class_dir, file)
                    
                    if i < train_split:
                        dst_file = os.path.join(train_dir, class_name, file)
                    elif i < val_split:
                        dst_file = os.path.join(val_dir, class_name, file)
                    else:
                        dst_file = os.path.join(test_dir, class_name, file)
                    
                    shutil.copy2(src_file, dst_file)
    
    # Count files in each directory
    train_count = sum(len(os.listdir(os.path.join(train_dir, c))) for c in classes)
    val_count = sum(len(os.listdir(os.path.join(val_dir, c))) for c in classes)
    test_count = sum(len(os.listdir(os.path.join(test_dir, c))) for c in classes)
    
    print(f"Dataset organized successfully:")
    print(f"  - Train: {train_count} images")
    print(f"  - Validation: {val_count} images")
    print(f"  - Test: {test_count} images")
    
    return True

def main():
    # Dataset information
    dataset_name = "gitenavnath/solar-augmented-dataset"
    download_dir = "data/solar_panels/raw"
    organized_dir = "data/solar_panels/organized"
    
    # Download the dataset
    success = download_kaggle_dataset(dataset_name, download_dir)
    if not success:
        print("Failed to download dataset. Please check your Kaggle credentials.")
        return 1
    
    # Organize the dataset
    success = organize_dataset(download_dir, organized_dir)
    if not success:
        print("Failed to organize dataset.")
        return 1
    
    print("Dataset preparation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
