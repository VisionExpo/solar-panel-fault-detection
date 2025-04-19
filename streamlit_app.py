import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io
from PIL import Image
import time
import json

# Set page configuration
st.set_page_config(
    page_title="Solar Panel Fault Detection",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_URL = "http://localhost:5000"  # Change this to your deployed API URL

def load_and_resize_image(image_file):
    """Load and resize image for display"""
    image = Image.open(image_file)
    return image

def predict_single_image(image_file):
    """Make prediction for a single image"""
    files = {'image': image_file}
    response = requests.post(f"{API_URL}/predict", files=files)
    return response.json() if response.status_code == 200 else None

def batch_predict_images(image_files):
    """Make predictions for multiple images"""
    files = [('images', img) for img in image_files]
    response = requests.post(f"{API_URL}/batch_predict", files=files)
    return response.json() if response.status_code == 200 else None

def get_metrics():
    """Get current model metrics"""
    response = requests.get(f"{API_URL}/metrics")
    return response.json() if response.status_code == 200 else None

def create_metrics_dashboard(metrics_data):
    """Create performance metrics dashboard"""
    if not metrics_data:
        return None
    
    # Create metrics visualizations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Inference Time",
            f"{metrics_data['performance_metrics']['average_inference_time_ms']:.2f} ms"
        )
    
    with col2:
        st.metric(
            "CPU Usage",
            f"{metrics_data['resource_usage']['average_cpu_percent']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Memory Usage",
            f"{metrics_data['resource_usage']['average_memory_percent']:.1f}%"
        )
    
    # Create predictions distribution chart
    predictions_data = metrics_data['performance_metrics']['predictions_per_class']
    fig = px.bar(
        x=list(predictions_data.keys()),
        y=list(predictions_data.values()),
        title="Predictions Distribution"
    )
    st.plotly_chart(fig)

def main():
    st.title("Solar Panel Fault Detection System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Performance Metrics"])
    
    if page == "Single Prediction":
        st.header("Single Image Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose an image of a solar panel", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            # Display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = load_and_resize_image(uploaded_file)
                st.image(image, use_column_width=True)
            
            # Make prediction
            with col2:
                st.subheader("Prediction Results")
                with st.spinner('Analyzing image...'):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    prediction = predict_single_image(uploaded_file)
                
                if prediction:
                    # Display main prediction
                    st.success(f"Predicted Class: {prediction['prediction']}")
                    st.info(f"Confidence: {prediction['confidence']:.2%}")
                    st.info(f"Inference Time: {prediction['inference_time_ms']:.2f} ms")
                    
                    # Display top 3 predictions
                    st.subheader("Top 3 Predictions")
                    for pred in prediction['top_3_predictions']:
                        st.write(f"{pred['class']}: {pred['confidence']:.2%}")
                else:
                    st.error("Error making prediction. Please try again.")
    
    elif page == "Batch Prediction":
        st.header("Batch Image Prediction")
        
        uploaded_files = st.file_uploader(
            "Choose multiple solar panel images", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.subheader(f"Processing {len(uploaded_files)} images")
            
            # Display images in grid
            cols = st.columns(3)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = load_and_resize_image(file)
                    st.image(image, use_column_width=True)
            
            # Make predictions
            if st.button("Analyze All Images"):
                with st.spinner('Analyzing images...'):
                    # Reset file pointers
                    for file in uploaded_files:
                        file.seek(0)
                    
                    predictions = batch_predict_images(uploaded_files)
                
                if predictions:
                    st.success(f"Processed {len(predictions['results'])} images")
                    st.info(f"Total inference time: {predictions['inference_time_ms']:.2f} ms")
                    
                    # Create results table
                    results_data = []
                    for idx, result in enumerate(predictions['results'], 1):
                        results_data.append({
                            'Image': f"Image {idx}",
                            'Prediction': result['prediction'],
                            'Confidence': f"{result['confidence']:.2%}"
                        })
                    
                    st.table(pd.DataFrame(results_data))
                else:
                    st.error("Error processing images. Please try again.")
    
    else:  # Performance Metrics
        st.header("Model Performance Metrics")
        
        # Auto-refresh metrics
        with st.spinner('Loading metrics...'):
            metrics = get_metrics()
        
        if metrics:
            create_metrics_dashboard(metrics)
            
            # Add refresh button
            if st.button("Refresh Metrics"):
                with st.spinner('Refreshing...'):
                    metrics = get_metrics()
                    create_metrics_dashboard(metrics)
        else:
            st.error("Error loading metrics. Please try again later.")
        
        # Add model information
        st.subheader("Model Information")
        st.write("""
        This system uses an EfficientNetB0 model optimized for solar panel fault detection.
        It can detect the following types of faults:
        - Bird droppings
        - Dusty panels
        - Electrical damage
        - Physical damage
        - Snow coverage
        - Clean panels (no faults)
        """)

if __name__ == "__main__":
    main()