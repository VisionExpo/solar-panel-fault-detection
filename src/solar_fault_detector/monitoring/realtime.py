"""
Real-time monitoring and alerting system.

Provides real-time model performance monitoring, anomaly detection,
and alerting capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    metric: str
    condition: str  # e.g., ">", "<", "==", "!="
    threshold: float
    duration_minutes: int = 5
    cooldown_minutes: int = 60


@dataclass
class Alert:
    """Alert instance."""

    rule_name: str
    message: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    value: float
    threshold: float


class RealTimeMonitor:
    """
    Real-time monitoring system for model performance.

    Features:
    - Real-time metric collection
    - Anomaly detection
    - Alerting system
    - Performance dashboards
    """

    def __init__(self, metrics_collector=None):
        """
        Initialize real-time monitor.

        Args:
            metrics_collector: MetricsCollector instance for Prometheus integration
        """
        self.metrics_collector = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.metric_history: Dict[str, List] = {}
        self.alert_callbacks: List[Callable] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Anomaly detection
        self.anomaly_detector = AnomalyDetector()

    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.

        Args:
            rule: AlertRule configuration
        """
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def add_alert_callback(self, callback: Callable) -> None:
        """
        Add callback function for alerts.

        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)

    def record_metric(
        self, metric_name: str, value: float, labels: Dict = None
    ) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        timestamp = datetime.now()

        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []

        self.metric_history[metric_name].append(
            {"value": value, "timestamp": timestamp, "labels": labels or {}}
        )

        # Keep only last 1000 values per metric
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]

        # Check alert rules
        self._check_alerts(metric_name, value)

        # Update Prometheus metrics if available
        if self.metrics_collector:
            if metric_name == "prediction_latency":
                self.metrics_collector.prediction_duration.labels(
                    model_type="realtime"
                ).observe(value)
            elif metric_name == "prediction_confidence":
                self.metrics_collector.prediction_confidence.labels(
                    model_type="realtime"
                ).observe(value)

    def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check if any alert rules are triggered."""
        for rule in self.alert_rules:
            if rule.metric != metric_name:
                continue

            # Check if condition is met
            triggered = self._evaluate_condition(value, rule.condition, rule.threshold)

            if triggered:
                # Check if alert is already active (cooldown)
                active_alert = next(
                    (a for a in self.active_alerts if a.rule_name == rule.name), None
                )

                if active_alert:
                    # Check cooldown
                    time_since_alert = datetime.now() - active_alert.timestamp
                    if time_since_alert < timedelta(minutes=rule.cooldown_minutes):
                        continue

                # Check duration requirement
                if self._check_duration_rule(rule, value):
                    self._trigger_alert(rule, value)

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 1e-6
        elif condition == "!=":
            return abs(value - threshold) >= 1e-6
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False

    def _check_duration_rule(self, rule: AlertRule, current_value: float) -> bool:
        """Check if alert condition has been met for the required duration."""
        if rule.duration_minutes == 0:
            return True

        metric_data = self.metric_history.get(rule.metric, [])
        if not metric_data:
            return False

        # Check recent values within duration window
        cutoff_time = datetime.now() - timedelta(minutes=rule.duration_minutes)
        recent_values = [
            entry["value"] for entry in metric_data if entry["timestamp"] > cutoff_time
        ]

        # All recent values must meet the condition
        return all(
            self._evaluate_condition(v, rule.condition, rule.threshold)
            for v in recent_values
        )

    def _trigger_alert(self, rule: AlertRule, value: float) -> None:
        """Trigger an alert."""
        severity = self._determine_severity(rule, value)

        alert = Alert(
            rule_name=rule.name,
            message=f"Alert triggered: {rule.metric} {rule.condition} {rule.threshold} (current: {value})",
            severity=severity,
            timestamp=datetime.now(),
            value=value,
            threshold=rule.threshold,
        )

        self.active_alerts.append(alert)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"ALERT: {alert.message} (severity: {severity})")

    def _determine_severity(self, rule: AlertRule, value: float) -> str:
        """Determine alert severity based on rule and value."""
        deviation = abs(value - rule.threshold) / rule.threshold

        if deviation > 1.0:
            return "critical"
        elif deviation > 0.5:
            return "high"
        elif deviation > 0.2:
            return "medium"
        else:
            return "low"

    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start real-time monitoring in background thread.

        Args:
            interval_seconds: Monitoring interval
        """
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("Real-time monitoring started")

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Real-time monitoring stopped")

    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._perform_monitoring_checks()
            except Exception as e:
                logger.error(f"Monitoring check failed: {e}")

            time.sleep(interval_seconds)

    def _perform_monitoring_checks(self) -> None:
        """Perform periodic monitoring checks."""
        # Update system metrics
        if self.metrics_collector:
            self.metrics_collector.update_system_metrics()

        # Check for anomalies
        for metric_name, history in self.metric_history.items():
            if len(history) < 10:  # Need minimum history
                continue

            recent_values = [entry["value"] for entry in history[-10:]]
            if self.anomaly_detector.detect_anomaly(recent_values):
                self.record_metric(f"{metric_name}_anomaly", 1.0)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring information
        """
        return {
            "is_monitoring": self.is_monitoring,
            "active_alerts": len(self.active_alerts),
            "alert_rules": len(self.alert_rules),
            "monitored_metrics": list(self.metric_history.keys()),
            "total_metric_points": sum(
                len(history) for history in self.metric_history.values()
            ),
        }

    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get recent alerts.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.active_alerts if alert.timestamp > cutoff_time]


class AnomalyDetector:
    """
    Simple anomaly detection using statistical methods.
    """

    def __init__(self, threshold_sigma: float = 3.0):
        """
        Initialize anomaly detector.

        Args:
            threshold_sigma: Standard deviation threshold for anomalies
        """
        self.threshold_sigma = threshold_sigma

    def detect_anomaly(self, values: List[float]) -> bool:
        """
        Detect if the latest value is anomalous.

        Args:
            values: List of recent values

        Returns:
            True if anomalous
        """
        if len(values) < 5:
            return False

        mean = np.mean(values[:-1])  # Exclude latest value
        std = np.std(values[:-1])

        if std == 0:
            return False

        latest_value = values[-1]
        z_score = abs(latest_value - mean) / std

        return z_score > self.threshold_sigma
