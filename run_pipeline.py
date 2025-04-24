import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """
    Run a command and print its output.
    
    Args:
        command: Command to run
        description: Description of the command
    
    Returns:
        True if the command succeeded, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"STEP: {description}")
    print(f"{'=' * 80}")
    print(f"Running command: {command}")
    print(f"{'-' * 80}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for the process to complete
        process.wait()
        
        # Check if the command succeeded
        if process.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {process.returncode}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run the solar panel fault detection pipeline')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the dataset')
    parser.add_argument('--skip-training', action='store_true', help='Skip training the model')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluating the model')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('src/models', exist_ok=True)
    os.makedirs('visualization/static', exist_ok=True)
    
    # Step 1: Download and prepare the dataset
    if not args.skip_download:
        success = run_command('python download_dataset.py', 'Download and prepare dataset')
        if not success:
            print("Failed to download and prepare dataset. Exiting.")
            return 1
    else:
        print("\nSkipping dataset download as requested.")
    
    # Step 2: Train the model
    if not args.skip_training:
        success = run_command('python train_model.py', 'Train model')
        if not success:
            print("Failed to train model. Exiting.")
            return 1
    else:
        print("\nSkipping model training as requested.")
    
    # Step 3: Evaluate the model
    if not args.skip_evaluation:
        success = run_command('python evaluate_model.py', 'Evaluate model')
        if not success:
            print("Failed to evaluate model.")
            return 1
    else:
        print("\nSkipping model evaluation as requested.")
    
    # Step 4: Copy model files to deployment directory
    success = run_command(
        'python -c "import shutil; import os; os.makedirs(\'deployment/model\', exist_ok=True); shutil.copytree(\'src/models/final_model\', \'deployment/model\', dirs_exist_ok=True); shutil.copy(\'src/models/label_mapping.json\', \'deployment/label_mapping.json\')"',
        'Copy model files to deployment directory'
    )
    if not success:
        print("Failed to copy model files to deployment directory.")
        return 1
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nYou can now run the application with:")
    print("  python deployment/app.py")
    print("  or")
    print("  python deployment/app_fastapi.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
