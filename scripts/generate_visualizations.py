import os
import subprocess
import time

def main():
    """Run all visualization scripts"""
    print("Generating visualizations for EDA and model training...")
    
    # Create visualization directory if it doesn't exist
    os.makedirs('visualization/static', exist_ok=True)
    
    # Run EDA visualizations script
    print("\n=== Running EDA visualizations ===")
    start_time = time.time()
    subprocess.run(['python', 'visualization/eda_visualizations.py'])
    print(f"EDA visualizations completed in {time.time() - start_time:.2f} seconds")
    
    # Run training visualizations script
    print("\n=== Running training visualizations ===")
    start_time = time.time()
    subprocess.run(['python', 'visualization/training_visualizations.py'])
    print(f"Training visualizations completed in {time.time() - start_time:.2f} seconds")
    
    print("\nAll visualizations generated successfully!")
    print("Visualizations saved to visualization/static/")

if __name__ == "__main__":
    main()
