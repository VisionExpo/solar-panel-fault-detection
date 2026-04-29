"""
Real-time monitoring and anomaly detection for production deployments.

Provides latency tracking, data drift detection, and anomaly alerts.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import time
from solar_fault_detector.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class AlertThresholds:
    """Thresholds for triggering monitoring alerts."""

    latency_p95_ms: float = 500.0
    error_rate_percent: float = 5.0
    confidence_drop_percent: float = 10.0
    memory_usage_percent: float = 90.0


class RealTimeMonitor:
    """
    Real-time monitoring system for production deployments.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        thresholds: Optional[AlertThresholds] = None,
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize real-time monitor.

        Args:
            metrics_collector: Configured Prometheus metrics collector
            thresholds: Alert thresholds
            alert_callback: Optional callback for custom alert routing (e.g. Slack/Email)
        """
        self.metrics_collector = metrics_collector
        self.thresholds = thresholds or AlertThresholds()
        self.alert_callback = alert_callback

        # Real-time state
        self._recent_latencies: List[float] = []
        self._recent_errors: int = 0
        self._total_requests: int = 0
        self._baseline_confidence: float = 0.85
        self._recent_confidences: List[float] = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Background monitoring thread
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
        logger.info("Real-time monitor started")

    def stop(self) -> None:
        """Stop background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            logger.info("Real-time monitor stopped")

    def record_inference(
        self, latency_ms: float, confidence: float, is_error: bool = False
    ) -> None:
        """Record a single inference event."""
        with self._lock:
            self._total_requests += 1

            if is_error:
                self._recent_errors += 1
                return

            self._recent_latencies.append(latency_ms)
            self._recent_confidences.append(confidence)

            # Keep only recent history (e.g., last 1000 requests)
            if len(self._recent_latencies) > 1000:
                self._recent_latencies.pop(0)
            if len(self._recent_confidences) > 1000:
                self._recent_confidences.pop(0)

    def _monitoring_loop(self) -> None:
        """Background loop evaluating thresholds and triggering alerts."""
        while self._running:
            self._evaluate_thresholds()
            self._update_system_metrics()
            time.sleep(60)  # Check every minute

    def _evaluate_thresholds(self) -> None:
        """Evaluate current metrics against thresholds."""
        with self._lock:
            if not self._total_requests:
                return

            # 1. Latency Check (P95)
            if len(self._recent_latencies) >= 10:
                import numpy as np

                p95_latency = float(np.percentile(self._recent_latencies, 95))
                if p95_latency > self.thresholds.latency_p95_ms:
                    self._trigger_alert(
                        "High Latency",
                        f"P95 latency ({p95_latency:.1f}ms) "
                        f"exceeds threshold ({self.thresholds.latency_p95_ms}ms)",
                    )

            # 2. Error Rate Check
            error_rate = (self._recent_errors / max(1, self._total_requests)) * 100
            if error_rate > self.thresholds.error_rate_percent:
                self._trigger_alert(
                    "High Error Rate",
                    f"Error rate ({error_rate:.1f}%) "
                    f"exceeds threshold ({self.thresholds.error_rate_percent}%)",
                )

            # 3. Confidence Drop (Data Drift Indicator)
            if len(self._recent_confidences) >= 50:
                import numpy as np

                avg_confidence = float(np.mean(self._recent_confidences))
                drop_percent = (
                    (self._baseline_confidence - avg_confidence)
                    / self._baseline_confidence
                ) * 100

                if drop_percent > self.thresholds.confidence_drop_percent:
                    self._trigger_alert(
                        "Potential Data Drift",
                        f"Average confidence dropped by {drop_percent:.1f}% "
                        f"(Current: {avg_confidence:.2f}, "
                        f"Baseline: {self._baseline_confidence:.2f})",
                    )

            # Reset short-term counters
            self._recent_errors = 0
            self._total_requests = 0

    def _update_system_metrics(self) -> None:
        """Update system metrics and check memory thresholds."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.percent > self.thresholds.memory_usage_percent:
                self._trigger_alert(
                    "High Memory Usage",
                    f"Memory usage ({memory.percent}%) "
                    f"exceeds threshold ({self.thresholds.memory_usage_percent}%)",
                )

            # Update Prometheus
            self.metrics_collector.update_system_metrics()

        except ImportError:
            pass

    def _trigger_alert(self, alert_type: str, message: str) -> None:
        """Trigger an alert via logging and optional callback."""
        logger.warning(f"🚨 ALERT [{alert_type}]: {message}")

        if self.alert_callback:
            try:
                self.alert_callback(alert_type, message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get current real-time statistics."""
        with self._lock:
            import numpy as np

            stats = {
                "total_requests": self._total_requests,
                "recent_errors": self._recent_errors,
                "current_time": datetime.now().isoformat(),
            }

            if self._recent_latencies:
                stats.update(
                    {
                        "avg_latency_ms": float(np.mean(self._recent_latencies)),
                        "p95_latency_ms": float(
                            np.percentile(self._recent_latencies, 95)
                        ),
                    }
                )

            if self._recent_confidences:
                stats.update(
                    {"avg_confidence": float(np.mean(self._recent_confidences))}
                )

            return stats
