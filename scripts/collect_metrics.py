import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.utils.logger import logger
import mlflow

def collect_run_metrics(experiment_name: str) -> pd.DataFrame:
    """Collect metrics from all MLflow runs"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    metrics_data = []
    
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        metrics.update(params)
        metrics['run_id'] = run.info.run_id
        metrics['start_time'] = run.info.start_time
        metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data)

def analyze_training_trends(df: pd.DataFrame, save_dir: Path):
    """Analyze and visualize training metrics trends"""
    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
    
    # Create performance over time plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Model Accuracy Over Time',
            'Training Loss Over Time',
            'Inference Time Trends',
            'Resource Usage'
        )
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=df['start_time'], y=df['val_accuracy'],
                  name='Validation Accuracy'),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=df['start_time'], y=df['val_loss'],
                  name='Validation Loss'),
        row=1, col=2
    )
    
    # Inference time plot
    if 'mean_inference_time_ms' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['start_time'], y=df['mean_inference_time_ms'],
                      name='Mean Inference Time (ms)'),
            row=2, col=1
        )
    
    # Resource usage plot
    if 'gpu_utilization' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['start_time'], y=df['gpu_utilization'],
                      name='GPU Utilization %'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True,
                     title_text='Model Performance Trends')
    
    # Save plot
    fig.write_html(str(save_dir / 'training_trends.html'))
    
    # Generate summary statistics
    latest_metrics = df.sort_values('start_time').iloc[-1]
    summary = {
        'latest_val_accuracy': latest_metrics.get('val_accuracy'),
        'latest_val_loss': latest_metrics.get('val_loss'),
        'latest_inference_time': latest_metrics.get('mean_inference_time_ms'),
        'best_val_accuracy': df['val_accuracy'].max(),
        'best_inference_time': df.get('mean_inference_time_ms', pd.Series()).min(),
        'total_runs': len(df),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

def analyze_class_performance(df: pd.DataFrame, save_dir: Path):
    """Analyze per-class performance metrics"""
    class_metrics = []
    
    for col in df.columns:
        if col.endswith('_f1'):
            class_name = col.replace('_f1', '')
            if class_name in ['Bird-drop', 'Clean', 'Dusty', 
                            'Electrical-damage', 'Physical-Damage', 'Snow-Covered']:
                class_metrics.append({
                    'class': class_name,
                    'f1_score': df[col].iloc[-1],
                    'accuracy': df.get(f'{class_name}_accuracy', pd.Series()).iloc[-1]
                })
    
    if class_metrics:
        class_df = pd.DataFrame(class_metrics)
        
        # Create class performance visualization
        fig = go.Figure(data=[
            go.Bar(name='F1 Score', x=class_df['class'], y=class_df['f1_score']),
            go.Bar(name='Accuracy', x=class_df['class'], y=class_df['accuracy'])
        ])
        
        fig.update_layout(
            title='Per-Class Performance Metrics',
            barmode='group'
        )
        
        fig.write_html(str(save_dir / 'class_performance.html'))
        
        class_df.to_csv(save_dir / 'class_metrics.csv', index=False)

def main():
    try:
        # Create metrics directory
        metrics_dir = Path('artifacts/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect metrics from different experiments
        experiments = ['solar_panel_fault_detection', 
                      'solar_panel_fault_detection_optimization']
        
        all_metrics = []
        for experiment in experiments:
            try:
                df = collect_run_metrics(experiment)
                all_metrics.append(df)
                
                # Save raw metrics
                df.to_csv(metrics_dir / f'{experiment}_metrics.csv', index=False)
            except Exception as e:
                logger.warning(f"Error collecting metrics for {experiment}: {str(e)}")
        
        if all_metrics:
            # Combine all metrics
            combined_df = pd.concat(all_metrics, ignore_index=True)
            
            # Analyze trends
            summary = analyze_training_trends(combined_df, metrics_dir)
            
            # Analyze class performance
            analyze_class_performance(combined_df, metrics_dir)
            
            logger.info(f"Metrics collection complete. Summary: {summary}")
        else:
            logger.warning("No metrics collected")
            
    except Exception as e:
        logger.error(f"Error in metrics collection: {str(e)}")
        raise e

if __name__ == "__main__":
    main()