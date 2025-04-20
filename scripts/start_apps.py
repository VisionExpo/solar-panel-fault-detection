import subprocess
import sys
import time
from pathlib import Path
import webbrowser
import os

def start_flask_app():
    """Start the Flask API server"""
    print("Starting Flask API server...")
    flask_process = subprocess.Popen(
        [sys.executable, "app.py"],
        env=dict(os.environ, FLASK_ENV="development")
    )
    time.sleep(2)  # Wait for Flask to start
    return flask_process

def start_streamlit_app():
    """Start the Streamlit web interface"""
    print("Starting Streamlit interface...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    )
    time.sleep(2)  # Wait for Streamlit to start
    return streamlit_process

def main():
    try:
        # Create necessary directories
        Path("artifacts/monitoring").mkdir(parents=True, exist_ok=True)
        Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
        
        # Start Flask API
        flask_process = start_flask_app()
        
        # Start Streamlit
        streamlit_process = start_streamlit_app()
        
        # Open web interfaces in browser
        webbrowser.open("http://localhost:5000/docs")  # API documentation
        webbrowser.open("http://localhost:8501")  # Streamlit interface
        
        print("\nSolar Panel Fault Detection System is running!")
        print("API Documentation: http://localhost:5000/docs")
        print("Web Interface: http://localhost:8501")
        print("\nPress Ctrl+C to stop all services...")
        
        # Keep the script running
        flask_process.wait()
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down services...")
        flask_process.terminate()
        streamlit_process.terminate()
        print("Services stopped successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()