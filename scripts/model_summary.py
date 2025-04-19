import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Load model
        model_path = "artifacts/models/final_model"
        model = tf.keras.models.load_model(model_path)
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Get model architecture details
        print("\nModel Architecture Details:")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        print(f"Number of Layers: {len(model.layers)}")
        
        # Count trainable and non-trainable parameters
        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        print(f"\nTrainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Total Parameters: {total_params:,}")
        
        # Load label mapping
        with open("artifacts/models/label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
        
        print(f"\nClasses: {list(label_mapping.keys())}")
        
        # Create model architecture visualization
        tf.keras.utils.plot_model(
            model,
            to_file="artifacts/models/model_architecture.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96
        )
        
        print("\nModel architecture visualization saved to artifacts/models/model_architecture.png")
        
    except Exception as e:
        print(f"Error in model summary: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
