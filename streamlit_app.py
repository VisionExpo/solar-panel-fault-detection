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
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Solar Panel Fault Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        .stAlert > div {
            padding: 0.5rem 1rem;
            margin-bottom: 1rem;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:5000")

# Cache API calls
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_metrics():
    """Get current model metrics"""
    try:
        response = requests.get(f"{API_URL}/metrics")
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_historical_metrics():
    """Get historical performance metrics"""
    try:
        metrics_file = Path("artifacts/metrics/performance_history.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
    except:
        return None
    return None

def process_image(image):
    """Process image for display and prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize for display while maintaining aspect ratio
    max_size = (800, 800)
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def create_performance_plots(metrics_data, historical_data=None):
    """Create performance visualization plots"""
    if not metrics_data:
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Current Metrics", "Historical Trends", "Resource Usage"])
    
    with tab1:
        # Current performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_inference = metrics_data['performance_metrics']['average_inference_time_ms']
            st.metric("Avg. Inference Time", f"{avg_inference:.1f} ms")
        
        with col2:
            p95_inference = metrics_data['performance_metrics'].get('p95_inference_time_ms', 0)
            st.metric("P95 Inference Time", f"{p95_inference:.1f} ms")
        
        with col3:
            total_predictions = sum(metrics_data['performance_metrics']
                                 ['predictions_per_class'].values())
            st.metric("Total Predictions", total_predictions)
        
        # Predictions distribution
        fig = px.bar(
            x=list(metrics_data['performance_metrics']['predictions_per_class'].keys()),
            y=list(metrics_data['performance_metrics']['predictions_per_class'].values()),
            title="Predictions Distribution by Class",
            labels={'x': 'Class', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if historical_data:
            # Create historical trends
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Accuracy trend
            fig = px.line(
                df, x='timestamp', y='accuracy',
                title="Model Accuracy Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Inference time trend
            fig = px.line(
                df, x='timestamp', y='inference_time_ms',
                title="Average Inference Time Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Resource usage gauges
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resource_usage']['average_cpu_percent'],
                title={'text': "CPU Usage %"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#1f77b4"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resource_usage']['average_memory_percent'],
                title={'text': "Memory Usage %"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#2ca02c"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resource_usage']['average_gpu_percent'],
                title={'text': "GPU Usage %"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#ff7f0e"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Solar Panel Fault Detection System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                          ["Single Prediction", "Batch Prediction", 
                           "Performance Monitoring", "About"])
    
    if page == "Single Prediction":
        st.header("Single Image Analysis")
        
        # File uploader with clear instructions
        st.info("📸 Upload a solar panel image for analysis. Supported formats: JPG, JPEG, PNG")
        uploaded_file = st.file_uploader("Choose an image", 
                                       type=["jpg", "jpeg", "png"],
                                       key="single_upload")
        
        if uploaded_file:
            try:
                # Create two columns for image and results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Uploaded Image")
                    image = Image.open(uploaded_file)
                    processed_image = process_image(image)
                    st.image(processed_image, use_column_width=True)
                
                with col2:
                    st.subheader("Analysis Results")
                    with st.spinner('Analyzing image...'):
                        # Reset file pointer and make prediction
                        uploaded_file.seek(0)
                        files = {'image': uploaded_file}
                        response = requests.post(f"{API_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results in a styled box
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            
                            # Main prediction with confidence
                            st.success(f"📌 Prediction: {result['prediction']}")
                            confidence = result['confidence'] * 100
                            st.progress(confidence / 100)
                            st.info(f"Confidence: {confidence:.1f}%")
                            
                            # Performance metrics
                            st.write(f"⚡ Inference Time: {result['inference_time_ms']:.1f} ms")
                            
                            # Top 3 predictions table
                            st.subheader("Alternative Predictions")
                            predictions_df = pd.DataFrame(result['top_3_predictions'])
                            predictions_df['confidence'] = predictions_df['confidence'] * 100
                            predictions_df.columns = ['Class', 'Confidence (%)']
                            st.table(predictions_df.set_index('Class'))
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("Error analyzing image. Please try again.")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    elif page == "Batch Prediction":
        st.header("Batch Image Analysis")
        
        # Batch upload with clear instructions
        st.info("📸 Upload multiple solar panel images for batch analysis. Maximum 32 images.")
        uploaded_files = st.file_uploader("Choose images", 
                                        type=["jpg", "jpeg", "png"],
                                        accept_multiple_files=True,
                                        key="batch_upload")
        
        if uploaded_files:
            if len(uploaded_files) > 32:
                st.warning("⚠️ Maximum 32 images allowed. Please reduce the selection.")
            else:
                # Display uploaded images in a grid
                cols = st.columns(4)
                for idx, file in enumerate(uploaded_files):
                    with cols[idx % 4]:
                        image = Image.open(file)
                        processed_image = process_image(image)
                        st.image(processed_image, caption=f"Image {idx+1}")
                
                if st.button("🔍 Analyze All Images"):
                    with st.spinner('Analyzing images...'):
                        # Prepare files for batch prediction
                        files = [('images', file) for file in uploaded_files]
                        response = requests.post(f"{API_URL}/batch_predict", files=files)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Display batch results
                            st.success(f"✅ Processed {len(results['results'])} images")
                            st.info(f"Total inference time: {results['inference_time_ms']:.1f} ms")
                            
                            # Create results table
                            results_data = []
                            for idx, result in enumerate(results['results'], 1):
                                results_data.append({
                                    'Image': f"Image {idx}",
                                    'Prediction': result['prediction'],
                                    'Confidence': f"{result['confidence']*100:.1f}%"
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.table(results_df.set_index('Image'))
                            
                            # Download results as CSV
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results CSV",
                                csv,
                                "batch_predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.error("Error processing batch. Please try again.")
    
    elif page == "Performance Monitoring":
        st.header("Model Performance Dashboard")
        
        # Add auto-refresh checkbox
        auto_refresh = st.checkbox("🔄 Auto-refresh (every 60 seconds)")
        
        if auto_refresh:
            st.empty()
            while True:
                # Get current metrics
                metrics = get_metrics()
                historical_data = get_historical_metrics()
                
                if metrics:
                    create_performance_plots(metrics, historical_data)
                else:
                    st.error("Unable to fetch metrics. Please try again later.")
                
                time.sleep(60)  # Wait for 60 seconds
                st.experimental_rerun()
        else:
            # Manual refresh button
            if st.button("🔄 Refresh Metrics"):
                metrics = get_metrics()
                historical_data = get_historical_metrics()
                
                if metrics:
                    create_performance_plots(metrics, historical_data)
                else:
                    st.error("Unable to fetch metrics. Please try again later.")
    
    else:  # About page
        st.header("About the System")
        
        st.markdown("""
        ### Solar Panel Fault Detection System
        
        This system uses deep learning to detect various types of faults in solar panels:
        
        - 🦅 **Bird droppings**
        - ✨ **Clean panels**
        - 🌫️ **Dusty panels**
        - ⚡ **Electrical damage**
        - 💢 **Physical damage**
        - ❄️ **Snow coverage**
        
        #### Technical Details
        
        - **Model**: EfficientNetB0 with custom top layers
        - **Input Size**: 224x224 pixels
        - **Processing**: Real-time inference with GPU acceleration
        - **Performance Monitoring**: Live metrics and resource usage tracking
        
        #### Usage Tips
        
        1. Upload clear images of solar panels
        2. Ensure good lighting conditions
        3. Avoid extreme angles
        4. Use batch processing for multiple images
        
        #### Need Help?
        
        Contact support at: your.email@example.com
        """)

if __name__ == "__main__":
    main()