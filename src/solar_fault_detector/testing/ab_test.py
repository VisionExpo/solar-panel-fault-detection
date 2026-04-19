"""
A/B Testing framework for model evaluation and comparison.

Supports running experiments with different models, configurations,
and measuring performance metrics.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment."""

    name: str
    variants: List[str]  # e.g., ['model_v1', 'model_v2']
    traffic_split: Dict[str, float]  # e.g., {'model_v1': 0.5, 'model_v2': 0.5}
    metrics: List[str]  # Metrics to track
    duration_days: int = 7
    min_sample_size: int = 1000


@dataclass
class ExperimentResult:
    """Results from A/B test experiment."""

    experiment_name: str
    variant: str
    metrics: Dict[str, float]
    sample_size: int
    timestamp: float


class ABTester:
    """
    A/B Testing framework for comparing model performance.

    Supports:
    - Traffic splitting between model variants
    - Statistical significance testing
    - Performance metric tracking
    - Experiment result analysis
    """

    def __init__(self, results_dir: str = "experiments"):
        """
        Initialize A/B tester.

        Args:
            results_dir: Directory to store experiment results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.results: List[ExperimentResult] = []

    def create_experiment(self, config: ExperimentConfig) -> None:
        """
        Create a new A/B test experiment.

        Args:
            config: Experiment configuration
        """
        # Validate traffic split
        total_split = sum(config.traffic_split.values())
        if not np.isclose(total_split, 1.0):
            raise ValueError(f"Traffic split must sum to 1.0, got {total_split}")

        if set(config.variants) != set(config.traffic_split.keys()):
            raise ValueError("Variants and traffic_split keys must match")

        self.active_experiments[config.name] = config
        logger.info(f"Created experiment: {config.name}")

    def assign_variant(self, experiment_name: str, user_id: Optional[str] = None) -> str:
        """
        Assign a variant to a user/request based on traffic split.

        Args:
            experiment_name: Name of the experiment
            user_id: Optional user identifier for consistent assignment

        Returns:
            Assigned variant name
        """
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        config = self.active_experiments[experiment_name]

        if user_id:
            # Use user_id for consistent assignment (deterministic)
            hash_value = hash(user_id + experiment_name) % 1000
            cumulative = 0.0
            for variant, split in config.traffic_split.items():
                cumulative += split
                if hash_value / 1000 <= cumulative:
                    return variant
        else:
            # Random assignment
            rand_val = np.random.random()
            cumulative = 0.0
            for variant, split in config.traffic_split.items():
                cumulative += split
                if rand_val <= cumulative:
                    return variant

        # Fallback (should not reach here)
        return config.variants[0]

    def record_result(
        self,
        experiment_name: str,
        variant: str,
        metrics: Dict[str, float],
        user_id: Optional[str] = None,
    ) -> None:
        """
        Record experiment result.

        Args:
            experiment_name: Name of the experiment
            variant: Variant that was used
            metrics: Dictionary of metric values
            user_id: Optional user identifier
        """
        result = ExperimentResult(
            experiment_name=experiment_name,
            variant=variant,
            metrics=metrics,
            sample_size=1,
            timestamp=time.time(),
        )

        self.results.append(result)

        # Save to file periodically
        if len(self.results) % 100 == 0:
            self._save_results()

        logger.debug(f"Recorded result for {experiment_name}:{variant}")

    def get_experiment_results(self, experiment_name: str) -> pd.DataFrame:
        """
        Get results for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            DataFrame with experiment results
        """
        experiment_results = [
            r for r in self.results if r.experiment_name == experiment_name
        ]

        if not experiment_results:
            return pd.DataFrame()

        # Convert to DataFrame
        data = []
        for result in experiment_results:
            row = {
                "variant": result.variant,
                "timestamp": result.timestamp,
                **result.metrics,
            }
            data.append(row)

        return pd.DataFrame(data)

    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Analyze experiment results and determine statistical significance.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Analysis results dictionary
        """
        df = self.get_experiment_results(experiment_name)

        if df.empty:
            return {"error": "No results found for experiment"}

        config = self.active_experiments.get(experiment_name)
        if not config:
            return {"error": "Experiment configuration not found"}

        analysis = {
            "experiment_name": experiment_name,
            "total_samples": len(df),
            "variants": {},
        }

        # Analyze each variant
        for variant in config.variants:
            variant_data = df[df["variant"] == variant]

            if len(variant_data) == 0:
                continue

            variant_analysis = {"sample_size": len(variant_data), "metrics": {}}  # type: ignore

            # Calculate metrics for each tracked metric
            for metric in config.metrics:
                if metric in variant_data.columns:
                    values = variant_data[metric]
                    variant_analysis["metrics"][metric] = {  # type: ignore
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }

            analysis["variants"][variant] = variant_analysis  # type: ignore

        # Statistical significance testing (simplified)
        if len(config.variants) == 2 and len(config.metrics) > 0:
            analysis["significance_tests"] = self._test_significance(df, config)  # type: ignore

        return analysis

    def _test_significance(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests between variants."""
        variant_a, variant_b = config.variants
        tests: Dict[str, Dict[str, Any]] = {}

        for metric in config.metrics:
            if metric not in df.columns:
                continue

            data_a = df[df["variant"] == variant_a][metric]
            data_b = df[df["variant"] == variant_b][metric]

            if len(data_a) < 10 or len(data_b) < 10:
                tests[metric] = {
                    "error": "Insufficient sample size for significance test"
                }
                continue

            # Simple t-test (in practice, you'd use scipy.stats)
            try:
                from scipy import stats

                t_stat, p_value = stats.ttest_ind(data_a, data_b)

                tests[metric] = {  # type: ignore
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "effect_size": abs(data_a.mean() - data_b.mean()) / data_a.std(),
                }
            except ImportError:
                # Fallback without scipy
                mean_diff = abs(data_a.mean() - data_b.mean())
                tests[metric] = {  # type: ignore
                    "mean_difference": float(mean_diff),
                    "note": "Install scipy for proper significance testing",
                }

        return tests

    def _save_results(self) -> None:
        """Save results to disk."""
        results_file = self.results_dir / "experiment_results.jsonl"

        with open(results_file, "a") as f:
            for result in self.results[-100:]:  # Save last 100 results
                json.dump(
                    {
                        "experiment_name": result.experiment_name,
                        "variant": result.variant,
                        "metrics": result.metrics,
                        "sample_size": result.sample_size,
                        "timestamp": result.timestamp,
                    },
                    f,
                )
                f.write("\n")

    def load_results(self) -> None:
        """Load results from disk."""
        results_file = self.results_dir / "experiment_results.jsonl"

        if not results_file.exists():
            return

        self.results = []
        with open(results_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                result = ExperimentResult(**data)
                self.results.append(result)


class ModelComparator:
    """
    Compare multiple models on the same test dataset.
    """

    def __init__(self, models: Dict[str, Any], test_data: np.ndarray):
        """
        Initialize model comparator.

        Args:
            models: Dictionary of model_name -> model_instance
            test_data: Test dataset
        """
        self.models = models
        self.test_data = test_data

    def compare_models(self, metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare models on test data.

        Args:
            metrics: List of metrics to compute

        Returns:
            Dictionary of model_name -> metrics_dict
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score"]

        results = {}

        for model_name, model in self.models.items():
            try:
                # Generate predictions
                predictions = model(self.test_data, training=False).numpy()
                _ = np.argmax(predictions, axis=1)

                # For demonstration, assume binary classification
                # In practice, you'd need true labels
                results[model_name] = {
                    "mean_confidence": float(np.max(predictions, axis=1).mean()),
                    "prediction_std": float(np.max(predictions, axis=1).std()),
                    "inference_time": 0.0,  # Would need to measure
                }

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {"error": str(e)}  # type: ignore

        return results
