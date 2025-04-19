import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
from tensorflow.keras.utils import plot_model

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.utils.logger import logger
from src.solar_panel_detector.components.data_preparation import DataPreparation

def visualize_model_architecture(model, output_path):
    """
    Visualize the model architecture and save it to a file.
    
    Args:
        model (tf.keras.Model): The model to visualize
        output_path (str): Path to save the visualization
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Plot model architecture
        plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96
        )
        logger.info(f"Model architecture visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Error visualizing model architecture: {str(e)}")

def visualize_training_history(history_path, output_dir):
    """
    Visualize the training history and save the plots.
    
    Args:
        history_path (str): Path to the training history JSON file
        output_dir (str): Directory to save the visualizations
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load training history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Convert to DataFrame for easier plotting
        history_df = pd.DataFrame(history)
        
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_df['accuracy'], label='Training Accuracy')
        plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history_df['loss'], label='Training Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        logger.info(f"Training history visualization saved to {os.path.join(output_dir, 'training_history.png')}")
        
        # Plot top-3 accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history_df['top_3_accuracy'], label='Training Top-3 Accuracy')
        plt.plot(history_df['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        plt.title('Model Top-3 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Top-3 Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top3_accuracy.png'))
        logger.info(f"Top-3 accuracy visualization saved to {os.path.join(output_dir, 'top3_accuracy.png')}")
        
        # Plot learning rate
        if 'lr' in history_df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(history_df['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
            logger.info(f"Learning rate visualization saved to {os.path.join(output_dir, 'learning_rate.png')}")
    
    except Exception as e:
        logger.error(f"Error visualizing training history: {str(e)}")

def visualize_sample_predictions(model, test_ds, label_mapping, output_dir, num_samples=16):
    """
    Visualize sample predictions from the test dataset.
    
    Args:
        model (tf.keras.Model): The trained model
        test_ds (tf.data.Dataset): Test dataset
        label_mapping (dict): Mapping from class names to indices
        output_dir (str): Directory to save the visualizations
        num_samples (int): Number of samples to visualize
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create reverse mapping (index to label)
        idx_to_label = {v: k for k, v in label_mapping.items()}
        
        # Get a batch of test images
        test_batch = next(iter(test_ds))
        test_images, test_labels = test_batch
        
        # Make predictions
        predictions = model.predict(test_images)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Plot images with predictions
        num_rows = int(np.ceil(num_samples / 4))
        plt.figure(figsize=(15, num_rows * 3.5))
        
        for i in range(min(num_samples, len(test_images))):
            plt.subplot(num_rows, 4, i + 1)
            img = test_images[i].numpy()
            
            # Denormalize image for display
            img = (img * 255).astype(np.uint8)
            
            plt.imshow(img)
            
            true_label = idx_to_label[true_classes[i]]
            pred_label = idx_to_label[pred_classes[i]]
            
            # Color code the title based on correct/incorrect prediction
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
            
            # Add prediction probabilities
            pred_prob = predictions[i][pred_classes[i]]
            plt.xlabel(f"Confidence: {pred_prob:.2f}")
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
        logger.info(f"Sample predictions visualization saved to {os.path.join(output_dir, 'sample_predictions.png')}")
    
    except Exception as e:
        logger.error(f"Error visualizing sample predictions: {str(e)}")

def visualize_confusion_matrix(model, test_ds, label_mapping, output_dir):
    """
    Visualize the confusion matrix for the test dataset.
    
    Args:
        model (tf.keras.Model): The trained model
        test_ds (tf.data.Dataset): Test dataset
        label_mapping (dict): Mapping from class names to indices
        output_dir (str): Directory to save the visualizations
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions and true labels
        y_pred = []
        y_true = []
        
        for images, labels in test_ds:
            predictions = model.predict(images)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(labels, axis=1)
            
            y_pred.extend(pred_classes)
            y_true.extend(true_classes)
        
        # Convert to numpy arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Get class names
        class_names = list(label_mapping.keys())
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        
        # Plot raw counts
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Counts)')
        
        # Plot normalized confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        logger.info(f"Confusion matrix visualization saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    except Exception as e:
        logger.error(f"Error visualizing confusion matrix: {str(e)}")

def visualize_class_activation_maps(model, test_ds, label_mapping, output_dir, num_samples=5):
    """
    Visualize class activation maps for sample images.
    
    Args:
        model (tf.keras.Model): The trained model
        test_ds (tf.data.Dataset): Test dataset
        label_mapping (dict): Mapping from class names to indices
        output_dir (str): Directory to save the visualizations
        num_samples (int): Number of samples to visualize
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Grad-CAM model
        # This is a simplified implementation and may need to be adapted based on the model architecture
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            logger.warning("Could not find a convolutional layer for Grad-CAM visualization")
            return
        
        # Create a model that outputs both the predictions and the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        
        # Create reverse mapping (index to label)
        idx_to_label = {v: k for k, v in label_mapping.items()}
        
        # Get a batch of test images
        test_batch = next(iter(test_ds))
        test_images, test_labels = test_batch
        
        # Plot images with Grad-CAM
        plt.figure(figsize=(15, num_samples * 3))
        
        for i in range(min(num_samples, len(test_images))):
            # Get the image
            img = test_images[i:i+1]
            
            # Get the true label
            true_class = np.argmax(test_labels[i])
            true_label = idx_to_label[true_class]
            
            # Compute Grad-CAM
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img)
                pred_class = tf.argmax(predictions[0])
                pred_label = idx_to_label[pred_class.numpy()]
                loss = predictions[:, pred_class]
            
            # Extract gradients
            grads = tape.gradient(loss, conv_output)
            
            # Pool gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight output feature map with gradients
            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to match image size
            img_shape = img[0].shape[:2]
            heatmap = cv2.resize(heatmap, (img_shape[1], img_shape[0]))
            
            # Convert image to RGB
            img_display = (img[0].numpy() * 255).astype(np.uint8)
            
            # Apply heatmap to image
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_display, 0.6, heatmap, 0.4, 0)
            
            # Plot original image and heatmap
            plt.subplot(num_samples, 2, i*2 + 1)
            plt.imshow(img_display)
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
            plt.axis('off')
            
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.imshow(superimposed_img)
            plt.title("Class Activation Map")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_activation_maps.png'))
        logger.info(f"Class activation maps saved to {os.path.join(output_dir, 'class_activation_maps.png')}")
    
    except Exception as e:
        logger.error(f"Error visualizing class activation maps: {str(e)}")

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Load model
        model_path = "artifacts/models/final_model"
        model = tf.keras.models.load_model(model_path)
        
        # Load label mapping
        label_mapping_path = "artifacts/models/label_mapping.json"
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        # Create output directory
        output_dir = "artifacts/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize model architecture
        visualize_model_architecture(model, os.path.join(output_dir, 'model_architecture.png'))
        
        # Visualize training history
        history_path = "artifacts/models/training_history.json"
        if os.path.exists(history_path):
            visualize_training_history(history_path, output_dir)
        else:
            logger.warning(f"Training history file not found at {history_path}")
        
        # Prepare test dataset
        data_preparation = DataPreparation(config)
        _, _, test_ds = data_preparation.get_datasets()
        
        # Visualize sample predictions
        visualize_sample_predictions(model, test_ds, label_mapping, output_dir)
        
        # Visualize confusion matrix
        visualize_confusion_matrix(model, test_ds, label_mapping, output_dir)
        
        # Visualize class activation maps
        visualize_class_activation_maps(model, test_ds, label_mapping, output_dir)
        
        logger.info("Model visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model visualization: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
