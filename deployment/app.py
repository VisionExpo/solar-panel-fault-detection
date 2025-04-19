import os
import sys
import numpy as np
import cv2
import json
import gradio as gr
from pathlib import Path
import tensorflow as tf
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the inference module
from inference import SolarPanelFaultDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to model and label mapping
MODEL_PATH = os.environ.get("MODEL_PATH", "model")
LABEL_MAPPING_PATH = os.environ.get("LABEL_MAPPING_PATH", "label_mapping.json")

# Initialize detector
detector = SolarPanelFaultDetector(MODEL_PATH, LABEL_MAPPING_PATH)

# Define class descriptions
class_descriptions = {
    "Bird-drop": "Solar panel with bird droppings on the surface.",
    "Clean": "Solar panel with no visible faults or issues.",
    "Dusty": "Solar panel covered with dust or dirt, reducing efficiency.",
    "Electrical-damage": "Solar panel with electrical damage, such as hotspots or broken connections.",
    "Physical-damage": "Solar panel with physical damage, such as cracks or broken glass.",
    "Snow-covered": "Solar panel covered with snow, preventing sunlight absorption."
}

def predict_image(image):
    """
    Predict the class of an image using the detector.
    
    Args:
        image: Image from Gradio
        
    Returns:
        Dictionary with prediction results and visualization
    """
    if image is None:
        return {
            "error": "No image provided"
        }
    
    try:
        # Convert from BGR to RGB (Gradio uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        predictions = detector.predict(image)
        
        # Get top prediction
        top_class, confidence = detector.classify_image(image)
        
        # Get top 3 predictions
        top_3 = detector.get_top_k_predictions(image, k=3)
        
        # Create result dictionary
        result = {
            "top_class": top_class,
            "confidence": float(confidence),
            "description": class_descriptions.get(top_class, "No description available."),
            "top_3": [
                {"class": class_name, "confidence": float(conf)}
                for class_name, conf in top_3
            ],
            "all_predictions": {
                class_name: float(conf)
                for class_name, conf in predictions.items()
            }
        }
        
        # Create visualization
        vis_image = image.copy()
        
        # Add text with prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{top_class}: {confidence:.2f}"
        cv2.putText(vis_image, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Convert back to BGR for Gradio
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        return vis_image, result
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None, {"error": str(e)}

def create_interface():
    """
    Create the Gradio interface.
    
    Returns:
        Gradio interface
    """
    # Create interface
    with gr.Blocks(title="Solar Panel Fault Detector") as interface:
        gr.Markdown("# Solar Panel Fault Detector")
        gr.Markdown("Upload an image of a solar panel to detect faults.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                submit_button = gr.Button("Detect Faults")
            
            with gr.Column():
                output_image = gr.Image(label="Visualization")
                output_json = gr.JSON(label="Prediction Results")
        
        # Set up examples
        examples_dir = os.path.join(project_root, "data", "examples")
        example_images = []
        if os.path.exists(examples_dir):
            example_images = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=input_image,
            )
        
        # Set up event handlers
        submit_button.click(
            fn=predict_image,
            inputs=input_image,
            outputs=[output_image, output_json]
        )
        
        # Add description
        gr.Markdown("""
        ## About
        
        This application detects faults in solar panels using a deep learning model.
        
        ### Fault Types
        
        - **Bird-drop**: Solar panel with bird droppings on the surface
        - **Clean**: Solar panel with no visible faults or issues
        - **Dusty**: Solar panel covered with dust or dirt
        - **Electrical-damage**: Solar panel with electrical damage
        - **Physical-damage**: Solar panel with physical damage
        - **Snow-covered**: Solar panel covered with snow
        
        ### Model Information
        
        - Architecture: EfficientNetB3
        - Input size: 300x300 pixels
        - Accuracy: ~50%
        - Top-3 Accuracy: ~80%
        """)
    
    return interface

if __name__ == "__main__":
    # Create interface
    interface = create_interface()
    
    # Launch interface
    interface.launch(server_name="0.0.0.0", server_port=7860)
