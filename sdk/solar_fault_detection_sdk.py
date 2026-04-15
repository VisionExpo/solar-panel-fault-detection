"""
Solar Panel Fault Detection API Client SDK

A Python SDK for interacting with the Solar Panel Fault Detection REST API.
Provides easy-to-use methods for prediction, batch processing, and monitoring.
"""

import requests
import json
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class SolarFaultDetectionClient:
    """
    Client for Solar Panel Fault Detection API.

    Provides methods to:
    - Make single predictions
    - Process batches of images
    - Monitor API health
    - Handle authentication
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        retries: int = 3,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retries: Number of retries for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "SolarFaultDetection-SDK/1.0",
            }
        )

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        logger.info(f"Initialized client for {base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.retries):
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    logger.error(f"Request failed after {self.retries} attempts: {e}")
                    raise
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                time.sleep(2**attempt)  # Exponential backoff

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status dictionary
        """
        response = self._make_request("GET", "/health")
        return response.json()

    def predict(
        self, image_path: Union[str, Path], return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction on a single image.

        Args:
            image_path: Path to image file
            return_probabilities: Whether to return all class probabilities

        Returns:
            Prediction results dictionary
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = self._make_request("POST", "/predict", files=files)

        result = response.json()

        if not return_probabilities and "probabilities" in result:
            del result["probabilities"]

        return result

    def predict_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            # For now, process individually (API might support batch in future)
            for path in batch_paths:
                try:
                    result = self.predict(path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to predict {path}: {e}")
                    results.append({"image": str(path), "error": str(e)})

        return results

    def predict_url(
        self, image_url: str, return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction on an image from URL.

        Args:
            image_url: URL of the image
            return_probabilities: Whether to return all class probabilities

        Returns:
            Prediction results dictionary
        """
        data = {"url": image_url, "return_probabilities": return_probabilities}

        response = self._make_request("POST", "/predict-url", json=data)
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the deployed model.

        Returns:
            Model information dictionary
        """
        response = self._make_request("GET", "/model-info")
        return response.json()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get API performance metrics.

        Returns:
            Metrics dictionary
        """
        response = self._make_request("GET", "/metrics")
        return response.json()


class AsyncSolarFaultDetectionClient(SolarFaultDetectionClient):
    """
    Asynchronous version of the API client using aiohttp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_session = None

    async def _ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if self._async_session is None:
            try:
                import aiohttp

                self._async_session = aiohttp.ClientSession(
                    headers=self.session.headers.copy(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                )
            except ImportError:
                raise ImportError("aiohttp required for async client")

    async def predict_async(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Make prediction asynchronously.

        Args:
            image_path: Path to image file

        Returns:
            Prediction results dictionary
        """
        await self._ensure_session()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=image_path.name)

            async with self._async_session.post(
                f"{self.base_url}/predict", data=data
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def close(self):
        """Close the async session."""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None


# Convenience functions
def create_client(
    base_url: str = "http://localhost:5000", **kwargs
) -> SolarFaultDetectionClient:
    """
    Create a new API client instance.

    Args:
        base_url: Base URL of the API
        **kwargs: Additional client arguments

    Returns:
        Configured client instance
    """
    return SolarFaultDetectionClient(base_url=base_url, **kwargs)


def quick_predict(
    image_path: Union[str, Path], base_url: str = "http://localhost:5000"
) -> Dict[str, Any]:
    """
    Quick prediction without creating a persistent client.

    Args:
        image_path: Path to image file
        base_url: API base URL

    Returns:
        Prediction results
    """
    client = SolarFaultDetectionClient(base_url=base_url)
    return client.predict(image_path)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    client = SolarFaultDetectionClient()

    # Health check
    try:
        health = client.health_check()
        print(f"API Health: {health}")
    except Exception as e:
        print(f"Health check failed: {e}")

    # Quick prediction example
    # result = quick_predict("path/to/image.jpg")
    # print(f"Prediction: {result}")
