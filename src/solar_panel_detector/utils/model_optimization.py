import tensorflow as tf
from pathlib import Path
import numpy as np
from typing import Tuple
from ..utils.logger import logger

class ModelOptimizer:
    @staticmethod
    def optimize_for_inference(model_path: Path, optimized_path: Path) -> None:
        """Optimize model for inference using TensorRT"""
        try:
            # Load the model
            model = tf.keras.models.load_model(str(model_path))
            
            # Convert to TensorRT
            converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=str(model_path),
                conversion_params=tf.experimental.tensorrt.ConversionParams(
                    precision_mode='FP16',
                    maximum_cached_engines=1000
                )
            )
            
            # Convert and save
            converter.convert()
            converter.save(str(optimized_path))
            logger.info(f"Model optimized and saved to {optimized_path}")
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise e
    
    @staticmethod
    def quantize_model(model_path: Path, quantized_path: Path,
                      calibration_data: tf.data.Dataset) -> None:
        """Quantize model to int8 for faster inference"""
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            
            # Set representative dataset for quantization
            def representative_dataset():
                for data, _ in calibration_data.take(100):
                    yield [data]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            quantized_model = converter.convert()
            
            # Save quantized model
            quantized_path.parent.mkdir(parents=True, exist_ok=True)
            quantized_path.write_bytes(quantized_model)
            
            logger.info(f"Model quantized and saved to {quantized_path}")
            
        except Exception as e:
            logger.error(f"Error quantizing model: {str(e)}")
            raise e
    
    @staticmethod
    def optimize_graph(model_path: Path, optimized_path: Path) -> None:
        """Optimize computation graph by fusing operations"""
        try:
            saved_model_path = str(model_path)
            optimized_model_path = str(optimized_path)
            
            # Load model
            model = tf.saved_model.load(saved_model_path)
            
            # Get concrete function
            concrete_func = model.signatures[
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            
            # Optimize graph
            optimized_graph = tf.compat.v1.graph_util.optimize_for_inference(
                concrete_func.graph.as_graph_def(),
                ['serving_default_input'],  # Input node names
                ['StatefulPartitionedCall'],  # Output node names
                tf.float32.as_datatype_enum
            )
            
            # Convert optimized graph to SavedModel
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
                optimized_model_path)
            
            with tf.compat.v1.Session() as sess:
                tf.import_graph_def(optimized_graph, name='')
                builder.add_meta_graph_and_variables(
                    sess,
                    [tf.saved_model.SERVING],
                    strip_default_attrs=True
                )
            
            builder.save()
            logger.info(f"Graph optimized and saved to {optimized_path}")
            
        except Exception as e:
            logger.error(f"Error optimizing graph: {str(e)}")
            raise e
    
    @staticmethod
    def benchmark_model(model_path: Path, test_data: tf.data.Dataset) -> dict:
        """Benchmark model performance"""
        try:
            model = tf.keras.models.load_model(str(model_path))
            batch_size = 32
            num_warmup_runs = 10
            num_benchmark_runs = 50
            
            # Warmup runs
            for images, _ in test_data.take(num_warmup_runs).batch(batch_size):
                _ = model.predict(images)
            
            # Benchmark runs
            inference_times = []
            for images, _ in test_data.take(num_benchmark_runs).batch(batch_size):
                start_time = tf.timestamp()
                _ = model.predict(images)
                end_time = tf.timestamp()
                inference_times.append(float(end_time - start_time))
            
            # Calculate statistics
            mean_time = np.mean(inference_times) * 1000  # Convert to ms
            p95_time = np.percentile(inference_times, 95) * 1000
            p99_time = np.percentile(inference_times, 99) * 1000
            
            results = {
                'mean_inference_time_ms': mean_time,
                'p95_inference_time_ms': p95_time,
                'p99_inference_time_ms': p99_time,
                'throughput_imgs_per_sec': batch_size / np.mean(inference_times)
            }
            
            logger.info(f"Benchmark results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {str(e)}")
            raise e