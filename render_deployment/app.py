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

        # Create tabs for different sections
        with gr.Tabs():
            # Prediction tab
            with gr.TabItem("Fault Detection"):
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

                # If no examples in data/examples, try using some from the Faulty_solar_panel directory
                if not example_images and os.path.exists('Faulty_solar_panel'):
                    for class_dir in os.listdir('Faulty_solar_panel'):
                        class_path = os.path.join('Faulty_solar_panel', class_dir)
                        if os.path.isdir(class_path):
                            files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if files:
                                # Take up to 2 examples from each class
                                example_images.extend(files[:2])
                                if len(example_images) >= 10:
                                    break

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

            # Data Analysis tab
            with gr.TabItem("Data Analysis"):
                gr.Markdown("## Exploratory Data Analysis")
                gr.Markdown("These visualizations show the characteristics of the training dataset.")

                with gr.Row():
                    # Check if visualization files exist
                    vis_path = os.path.join(os.path.dirname(__file__), "static")

                    if os.path.exists(os.path.join(vis_path, "class_distribution.png")):
                        gr.Image(os.path.join(vis_path, "class_distribution.png"), label="Class Distribution")
                    else:
                        gr.Markdown("*Class distribution visualization not available*")

                with gr.Row():
                    if os.path.exists(os.path.join(vis_path, "sample_images.png")):
                        gr.Image(os.path.join(vis_path, "sample_images.png"), label="Sample Images")
                    else:
                        gr.Markdown("*Sample images visualization not available*")

                with gr.Row():
                    with gr.Column():
                        if os.path.exists(os.path.join(vis_path, "dimension_distributions.png")):
                            gr.Image(os.path.join(vis_path, "dimension_distributions.png"), label="Image Dimensions")
                        else:
                            gr.Markdown("*Image dimensions visualization not available*")

                    with gr.Column():
                        if os.path.exists(os.path.join(vis_path, "brightness_distribution.png")):
                            gr.Image(os.path.join(vis_path, "brightness_distribution.png"), label="Brightness Distribution")
                        else:
                            gr.Markdown("*Brightness distribution visualization not available*")

            # Model Performance tab
            with gr.TabItem("Model Performance"):
                gr.Markdown("## Training and Evaluation Results")
                gr.Markdown("These visualizations show the performance of the trained model.")

                with gr.Row():
                    if os.path.exists(os.path.join(vis_path, "training_curves.png")):
                        gr.Image(os.path.join(vis_path, "training_curves.png"), label="Training Curves")
                    else:
                        gr.Markdown("*Training curves visualization not available*")

                with gr.Row():
                    if os.path.exists(os.path.join(vis_path, "learning_rate.png")):
                        gr.Image(os.path.join(vis_path, "learning_rate.png"), label="Learning Rate Schedule")
                    else:
                        gr.Markdown("*Learning rate visualization not available*")

                with gr.Row():
                    with gr.Column():
                        if os.path.exists(os.path.join(vis_path, "confusion_matrix.png")):
                            gr.Image(os.path.join(vis_path, "confusion_matrix.png"), label="Confusion Matrix")
                        else:
                            gr.Markdown("*Confusion matrix visualization not available*")

                    with gr.Column():
                        if os.path.exists(os.path.join(vis_path, "class_metrics.png")):
                            gr.Image(os.path.join(vis_path, "class_metrics.png"), label="Class Metrics")
                        else:
                            gr.Markdown("*Class metrics visualization not available*")

                # Load and display training summary if available
                summary_path = os.path.join(vis_path, "training_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)

                    gr.Markdown(f"""
                    ### Training Summary

                    - **Epochs**: {summary.get('epochs', 'N/A')}
                    - **Best Validation Accuracy**: {summary.get('best_val_acc', 'N/A'):.4f} (Epoch {summary.get('best_val_acc_epoch', 'N/A')})
                    - **Final Validation Accuracy**: {summary.get('final_val_acc', 'N/A'):.4f}
                    - **Best Validation Loss**: {summary.get('best_val_loss', 'N/A'):.4f} (Epoch {summary.get('best_val_loss_epoch', 'N/A')})
                    - **Final Validation Loss**: {summary.get('final_val_loss', 'N/A'):.4f}
                    """)
                else:
                    gr.Markdown("*Training summary not available*")

            # About tab
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About

                This application detects faults in solar panels using a deep learning model. It helps solar farm operators and maintenance teams identify issues that may reduce energy production efficiency.

                ### Fault Types

                - **Bird-drop**: Solar panel with bird droppings on the surface. Bird droppings can create hotspots and reduce efficiency by blocking sunlight.
                - **Clean**: Solar panel with no visible faults or issues. These panels are operating at optimal efficiency.
                - **Dusty**: Solar panel covered with dust or dirt. Dust accumulation can reduce efficiency by up to 30% in some regions.
                - **Electrical-damage**: Solar panel with electrical damage, such as hotspots, PID (Potential Induced Degradation), or broken connections.
                - **Physical-damage**: Solar panel with physical damage, such as cracks, broken glass, or delamination.
                - **Snow-covered**: Solar panel covered with snow. Snow can completely block sunlight and reduce output to zero.

                ### Dataset Information

                - The model was trained on a dataset of solar panel images collected from various sources
                - Images include panels in different environmental conditions and with various fault types
                - The dataset was split into training (70%), validation (15%), and test (15%) sets
                - Data augmentation techniques were applied to increase the diversity of the training data

                ### Model Information

                - **Architecture**: EfficientNetB3 - a convolutional neural network known for its efficiency and accuracy
                - **Input size**: 384x384 pixels - images are resized and normalized before processing
                - **Transfer Learning**: Uses pre-trained weights from ImageNet to leverage features learned on a large dataset
                - **Fine-tuning**: The model was fine-tuned on our solar panel dataset with a reduced learning rate
                - **Training Process**: Trained using categorical cross-entropy loss and Adam optimizer
                - **Regularization**: Dropout and early stopping were used to prevent overfitting

                ### Performance Metrics

                - **Accuracy**: The model achieves good accuracy on the test set
                - **Precision and Recall**: Balanced to ensure reliable fault detection
                - **Confusion Matrix**: Available in the Model Performance tab
                - **Class-specific Metrics**: Different fault types have varying detection accuracy

                ### Deployment Information

                - **Web Interface**: Built with Gradio for interactive use and visualization
                - **API**: FastAPI REST API available for integration with other systems
                - **Batch Processing**: Support for processing multiple images in batch mode
                - **Cloud Deployment**: Ready for deployment on cloud platforms like Render
                - **Containerization**: Docker support for consistent deployment across environments

                ### Use Cases

                - **Routine Inspection**: Regular monitoring of solar farm health
                - **Maintenance Planning**: Identifying panels that need cleaning or repair
                - **Performance Optimization**: Ensuring maximum energy production
                - **Damage Assessment**: Evaluating panels after storms or other events
                - **Quality Control**: Checking newly installed panels for manufacturing defects

                ### Limitations

                - The model may not detect subtle or internal faults that don't have visible signs
                - Performance may vary based on lighting conditions and image quality
                - Some fault types may be confused with others if they have similar visual characteristics
                - The model should be used as an aid to human inspection, not as a complete replacement
                """)

    return interface

if __name__ == "__main__":
    # Create interface
    interface = create_interface()

    # Launch interface
    interface.launch(server_name="0.0.0.0", server_port=7860)
