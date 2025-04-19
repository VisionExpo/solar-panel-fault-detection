import time
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from collections import deque
import threading
from ..utils.logger import logger
import mlflow
import psutil
import GPUtil
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelMonitor:
    def __init__(self, metrics_path: Path = Path("artifacts/monitoring")):
        self.metrics_path = metrics_path
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics storage
        self.inference_times = deque(maxlen=1000)
        self.batch_sizes = deque(maxlen=1000)
        self.prediction_counts = {category: 0 for category in [
            'Bird-drop', 'Clean', 'Dusty', 
            'Electrical-damage', 'Physical-Damage', 'Snow-Covered'
        ]}
        
        # Resource monitoring
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def log_prediction(self, prediction: str, confidence: float, inference_time: float,
                      batch_size: int = 1):
        """Log a single prediction with its metadata"""
        timestamp = datetime.now().isoformat()
        
        # Update metrics
        self.inference_times.append(inference_time)
        self.batch_sizes.append(batch_size)
        self.prediction_counts[prediction] += 1
        
        # Log to MLflow
        mlflow.log_metrics({
            'inference_time': inference_time,
            'confidence': confidence,
            'batch_size': batch_size
        }, step=len(self.inference_times))
        
        # Save detailed log
        log_entry = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'inference_time': inference_time,
            'batch_size': batch_size
        }
        
        log_file = self.metrics_path / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while True:
            # CPU and memory usage
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # GPU usage if available
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usage.append(gpus[0].load * 100)
            else:
                self.gpu_usage.append(0)
            
            time.sleep(1)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_time = datetime.now()
        
        # Calculate metrics
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        p95_inference_time = np.percentile(self.inference_times, 95) if self.inference_times else 0
        avg_batch_size = np.mean(self.batch_sizes) if self.batch_sizes else 0
        
        # Resource usage statistics
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        avg_gpu = np.mean(self.gpu_usage) if self.gpu_usage else 0
        
        report = {
            'timestamp': current_time.isoformat(),
            'performance_metrics': {
                'average_inference_time_ms': avg_inference_time * 1000,
                'p95_inference_time_ms': p95_inference_time * 1000,
                'average_batch_size': avg_batch_size,
                'predictions_per_class': dict(self.prediction_counts)
            },
            'resource_usage': {
                'average_cpu_percent': avg_cpu,
                'average_memory_percent': avg_memory,
                'average_gpu_percent': avg_gpu
            }
        }
        
        # Save report
        report_file = self.metrics_path / f"performance_report_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate visualizations
        self._generate_performance_plots(current_time)
        
        return report
    
    def _generate_performance_plots(self, timestamp: datetime):
        """Generate performance visualization plots"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inference Time Distribution', 'Predictions by Class',
                          'Resource Usage Over Time', 'Batch Size Distribution')
        )
        
        # Inference time histogram
        fig.add_trace(
            go.Histogram(x=list(self.inference_times), name='Inference Time (s)'),
            row=1, col=1
        )
        
        # Predictions by class bar chart
        fig.add_trace(
            go.Bar(x=list(self.prediction_counts.keys()),
                  y=list(self.prediction_counts.values()),
                  name='Predictions Count'),
            row=1, col=2
        )
        
        # Resource usage line plot
        fig.add_trace(
            go.Scatter(y=list(self.cpu_usage), name='CPU %'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=list(self.memory_usage), name='Memory %'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=list(self.gpu_usage), name='GPU %'),
            row=2, col=1
        )
        
        # Batch size histogram
        fig.add_trace(
            go.Histogram(x=list(self.batch_sizes), name='Batch Size'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True,
                         title_text=f"Performance Dashboard - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save plot
        plot_file = self.metrics_path / f"performance_dashboard_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(plot_file))
        
        # Log to MLflow
        mlflow.log_artifact(str(plot_file))