"""
Model explainability and interpretability module.

Provides LIME and SHAP-based explanations for model predictions,
helping understand why the model made certain decisions.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np

try:
    import lime
    import lime.lime_image

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Model explainability wrapper using LIME and SHAP.

    Provides local and global explanations for image classification predictions.
    """

    def __init__(self, model, config, preprocessor=None):
        """
        Initialize explainer.

        Args:
            model: Trained Keras model
            config: ModelConfig with image specifications
            preprocessor: ImagePreprocessor instance
        """
        self.model = model
        self.config = config
        self.preprocessor = preprocessor

        # Initialize explainers
        self.lime_explainer = None
        self.shap_explainer = None

        if LIME_AVAILABLE:
            try:
                self.lime_explainer = lime.lime_image.LimeImageExplainer()
                logger.info("LIME explainer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LIME explainer: {e}")

        if SHAP_AVAILABLE:
            try:
                # For image models, we'll use a simpler approach
                # SHAP can be computationally expensive for images
                self.shap_explainer = shap.Explainer(self.model)
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")

    def explain_prediction_lime(
        self,
        image: np.ndarray,
        prediction_class: int = None,
        num_samples: int = 1000,
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Explain prediction using LIME (Local Interpretable Model-agnostic Explanations).

        Args:
            image: Preprocessed image array
            prediction_class: Class to explain (default: predicted class)
            num_samples: Number of samples for LIME
            num_features: Number of top features to return

        Returns:
            Dictionary with explanation results
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {"method": "lime", "available": False, "error": "LIME not available"}

        try:
            # LIME expects images in different format
            # Convert from (H, W, C) to (H, W, C) with values 0-255
            image_for_lime = (image * 255).astype(np.uint8)

            # Create prediction function for LIME
            def predict_fn(images):
                # LIME passes images as (N, H, W, C)
                processed_images = []
                for img in images:
                    # Convert back to float and normalize
                    img_float = img.astype(np.float32) / 255.0
                    processed_images.append(img_float)
                batch = np.array(processed_images)
                return self.model(batch, training=False).numpy()

            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                image_for_lime,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=num_samples,
            )

            # Get the class to explain
            if prediction_class is None:
                prediction_class = explanation.top_labels[0]

            # Get top features
            temp, mask = explanation.get_image_and_mask(
                prediction_class,
                positive_only=True,
                num_features=num_features,
                hide_rest=True,
            )

            return {
                "method": "lime",
                "available": True,
                "explained_class": prediction_class,
                "top_features": num_features,
                "explanation_mask": mask.tolist(),
                "explained_image": temp.tolist(),
                "feature_importance": explanation.local_exp[prediction_class],
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"method": "lime", "available": False, "error": str(e)}

    def explain_prediction_shap(
        self,
        image: np.ndarray,
        background_images: Optional[np.ndarray] = None,
        max_evals: int = 100,
    ) -> Dict[str, Any]:
        """
        Explain prediction using SHAP (SHapley Additive exPlanations).

        Args:
            image: Preprocessed image array
            background_images: Background images for SHAP (optional)
            max_evals: Maximum evaluations for SHAP

        Returns:
            Dictionary with SHAP explanation results
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"method": "shap", "available": False, "error": "SHAP not available"}

        try:
            # For image models, SHAP can be expensive
            # We'll use a simplified approach
            if background_images is None:
                # Create simple background (mean of some images)
                background_images = np.mean(image, axis=0, keepdims=True)
                background_images = np.tile(background_images, (10, 1, 1, 1))

            # Explain prediction
            shap_values = self.shap_explainer(background_images, max_evals=max_evals)

            return {
                "method": "shap",
                "available": True,
                "shap_values": (
                    shap_values.values.tolist()
                    if hasattr(shap_values, "values")
                    else []
                ),
                "base_values": (
                    shap_values.base_values.tolist()
                    if hasattr(shap_values, "base_values")
                    else []
                ),
                "data": (
                    shap_values.data.tolist() if hasattr(shap_values, "data") else []
                ),
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"method": "shap", "available": False, "error": str(e)}

    def get_feature_importance(
        self, test_images: np.ndarray, method: str = "lime", sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get global feature importance across multiple images.

        Args:
            test_images: Batch of test images
            method: Explanation method ('lime' or 'shap')
            sample_size: Number of images to sample

        Returns:
            Global feature importance results
        """
        if len(test_images) > sample_size:
            indices = np.random.choice(len(test_images), sample_size, replace=False)
            test_images = test_images[indices]

        explanations = []

        for i, image in enumerate(test_images):
            if method == "lime":
                exp = self.explain_prediction_lime(image)
            elif method == "shap":
                exp = self.explain_prediction_shap(image)
            else:
                continue

            if exp.get("available", False):
                explanations.append(exp)

        if not explanations:
            return {
                "method": method,
                "global_importance": {},
                "error": "No successful explanations",
            }

        # Aggregate explanations (simplified)
        # In practice, you'd want more sophisticated aggregation
        return {
            "method": method,
            "sample_size": len(explanations),
            "global_importance": "Feature importance analysis completed",
            "explanations_count": len(explanations),
        }

    def generate_explanation_report(
        self, image: np.ndarray, prediction_result: Dict
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report for a prediction.

        Args:
            image: Preprocessed image array
            prediction_result: Original prediction result

        Returns:
            Complete explanation report
        """
        report = {"prediction": prediction_result, "explanations": {}}

        # LIME explanation
        lime_exp = self.explain_prediction_lime(image)
        report["explanations"]["lime"] = lime_exp

        # SHAP explanation
        shap_exp = self.explain_prediction_shap(image)
        report["explanations"]["shap"] = shap_exp

        # Add summary
        report["summary"] = {
            "explained_class": prediction_result.get("predicted_class"),
            "confidence": prediction_result.get("confidence"),
            "explanation_methods": [
                method
                for method in ["lime", "shap"]
                if report["explanations"][method].get("available", False)
            ],
        }

        return report
