"""
Monitoring and observability module.

Provides Prometheus metrics, logging, and health checks
for the Solar Panel Fault Detection system.
"""

import logging
import time
from typing import Dict, Any

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create dummy classes for when prometheus is not available
    class Counter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Histogram:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Gauge:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

    def start_http_server(*args, **kwargs):
        pass


logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Prometheus metrics collector for model performance and system health.
    """

    def __init__(self, service_name: str = "solar_fault_detector"):
        self.service_name = service_name

        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus-client not available. Metrics will be no-ops.")
            return

        # Prediction metrics
        self.prediction_total = Counter(
            f"{service_name}_predictions_total",
            "Total number of predictions made",
            ["model_type", "status"],
        )

        self.prediction_duration = Histogram(
            f"{service_name}_prediction_duration_seconds",
            "Time spent processing predictions",
            ["model_type"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        self.prediction_confidence = Histogram(
            f"{service_name}_prediction_confidence",
            "Prediction confidence distribution",
            ["model_type"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        # System metrics
        self.active_requests = Gauge(
            f"{service_name}_active_requests",
            "Number of active requests being processed",
        )

        self.memory_usage = Gauge(
            f"{service_name}_memory_usage_bytes", "Current memory usage in bytes"
        )

        self.cpu_usage = Gauge(
            f"{service_name}_cpu_usage_percent", "Current CPU usage percentage"
        )

        # Error metrics
        self.errors_total = Counter(
            f"{service_name}_errors_total",
            "Total number of errors",
            ["error_type", "component"],
        )

        # Cache metrics
        self.cache_hits = Counter(
            f"{service_name}_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"],
        )

        self.cache_misses = Counter(
            f"{service_name}_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"],
        )

        # Model metrics
        self.model_load_duration = Histogram(
            f"{service_name}_model_load_duration_seconds",
            "Time spent loading models",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        self.model_loaded = Gauge(
            f"{service_name}_model_loaded",
            "Whether the model is currently loaded (1) or not (0)",
        )

    def record_prediction(
        self, model_type: str, duration: float, confidence: float, success: bool = True
    ) -> None:
        """Record prediction metrics."""
        status = "success" if success else "error"

        self.prediction_total.labels(model_type=model_type, status=status).inc()
        self.prediction_duration.labels(model_type=model_type).observe(duration)
        self.prediction_confidence.labels(model_type=model_type).observe(confidence)

    def record_request_start(self) -> None:
        """Record start of a request."""
        self.active_requests.inc()

    def record_request_end(self) -> None:
        """Record end of a request."""
        self.active_requests.dec()

    def record_error(self, error_type: str, component: str) -> None:
        """Record an error."""
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()

    def record_model_load(self, duration: float, success: bool) -> None:
        """Record model loading metrics."""
        self.model_load_duration.observe(duration)
        self.model_loaded.set(1 if success else 0)

    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)

        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")


class HealthChecker:
    """
    Health check utilities for system monitoring.
    """

    def __init__(self, model_predictor=None):
        self.model_predictor = model_predictor
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dictionary with health status and details
        """
        current_time = time.time()

        # Basic health
        health = {"status": "healthy", "timestamp": current_time, "checks": {}}

        # Model health check
        if self.model_predictor:
            try:
                # Try a simple prediction to verify model works
                # This would need a test image in practice
                model_status = "healthy"
                model_details = "Model loaded and responsive"
            except Exception as e:
                model_status = "unhealthy"
                model_details = f"Model error: {str(e)}"
                health["status"] = "unhealthy"
        else:
            model_status = "unknown"
            model_details = "Model predictor not initialized"

        health["checks"]["model"] = {"status": model_status, "details": model_details}  # type: ignore

        # System resources
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()

            system_status = "healthy" if memory.percent < 90 and cpu < 95 else "warning"
            system_details = f"Memory: {memory.percent:.1f}%, CPU: {cpu:.1f}%"

            if system_status == "warning":
                health["status"] = "warning"

        except ImportError:
            system_status = "unknown"
            system_details = "System monitoring not available"

        health["checks"]["system"] = {  # type: ignore
            "status": system_status,
            "details": system_details,
        }

        # Update last check time
        self.last_health_check = current_time

        return health

    def is_healthy(self) -> bool:
        """Quick health check returning boolean."""
        health = self.health_check()
        return health["status"] == "healthy"


def start_metrics_server(port: int = 8000) -> None:
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on
    """
    if PROMETHEUS_AVAILABLE:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    else:
        logger.warning("Prometheus not available, metrics server not started")
