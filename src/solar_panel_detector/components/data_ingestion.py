import os
import requests
from pathlib import Path
import logging
from ..utils.logger import logger
from ..config.configuration import Config
from bs4 import BeautifulSoup
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
import hashlib
from PIL import Image
import io
import random
import numpy as np

class DataIngestion:
    def __init__(self, config: Config):
        self.config = config
        self.categories = [
            'Bird-drop', 'Clean', 'Dusty',
            'Electrical-damage', 'Physical-Damage', 'Snow-Covered'
        ]
        # Rotate between different user agents to avoid blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        # Target counts per category (more realistic numbers)
        self.target_counts = {
            'Bird-drop': 500,
            'Clean': 1000,  # More clean examples as baseline
            'Dusty': 500,
            'Electrical-damage': 300,
            'Physical-Damage': 300,
            'Snow-Covered': 300
        }

    def get_random_user_agent(self):
        return random.choice(self.user_agents)

    def count_category_data(self) -> Dict[str, int]:
        """Count existing images for each category"""
        counts = {}
        for category in self.categories:
            category_path = Path(self.config.data.data_dir) / category
            if category_path.exists():
                image_files = list(category_path.glob('*.[jJ][pP][gG]')) + \
                            list(category_path.glob('*.[jJ][pP][eE][gG]'))
                counts[category] = len(image_files)
            else:
                counts[category] = 0
        return counts

    def download_image(self, url: str, save_path: Path) -> bool:
        """Download and validate a single image"""
        try:
            headers = {'User-Agent': self.get_random_user_agent()}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Validate image
                try:
                    img = Image.open(io.BytesIO(response.content))
                    img = img.convert('RGB')  # Convert to RGB if needed

                    # Enhanced image validation
                    if img.size[0] < 100 or img.size[1] < 100:
                        return False
                    if img.size[0] > 4000 or img.size[1] > 4000:  # Skip very large images
                        return False

                    # Resize if too large while maintaining aspect ratio
                    if img.size[0] > 1024 or img.size[1] > 1024:
                        img.thumbnail((1024, 1024), Image.LANCZOS)

                    # Basic quality check (skip mostly black/white images)
                    img_array = np.array(img)
                    if np.mean(img_array) < 20 or np.mean(img_array) > 235:
                        return False

                    # Save image
                    img.save(save_path, quality=85, optimize=True)  # Optimize file size
                    return True
                except:
                    return False
            return False
        except:
            return False

    def search_images(self, query: str, max_results: int = 100) -> List[str]:
        """Search for images using multiple search engines"""
        image_urls = set()
        headers = {'User-Agent': self.get_random_user_agent()}

        # Try Google Custom Search if API keys are available
        google_api_key = os.getenv('GOOGLE_API_KEY')
        google_cse_id = os.getenv('GOOGLE_CSE_ID')

        if google_api_key and google_cse_id:
            try:
                response = requests.get(
                    'https://www.googleapis.com/customsearch/v1',
                    params={
                        'key': google_api_key,
                        'cx': google_cse_id,
                        'searchType': 'image',
                        'q': query,
                        'num': 10  # Max allowed by API
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'items' in data:
                        for item in data['items']:
                            image_urls.add(item['link'])
            except Exception as e:
                logger.error(f"Error using Google Custom Search: {str(e)}")

        # Try Bing
        try:
            bing_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&qft=+filterui:photo-photo&first=1&count=100"
            response = requests.get(bing_url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.select('img.mimg'):
                    if 'src' in img.attrs:
                        image_urls.add(img['src'])

                # Try to extract more image URLs from the JSON data in the page
                for script in soup.find_all('script'):
                    if 'iurl' in str(script):
                        import re
                        urls = re.findall(r'"iurl":"([^"]+)"', str(script))
                        image_urls.update(urls)
        except Exception as e:
            logger.error(f"Error using Bing search: {str(e)}")

        # Try DuckDuckGo
        try:
            ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}&iax=images&ia=images"
            response = requests.get(ddg_url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.select('img[data-src]'):
                    if 'data-src' in img.attrs:
                        image_urls.add(img['data-src'])
        except Exception as e:
            logger.error(f"Error using DuckDuckGo search: {str(e)}")

        # Clean and validate URLs
        valid_urls = []
        for url in image_urls:
            if url.startswith('//'):
                url = 'https:' + url
            if url.startswith('http'):
                valid_urls.append(url)

        return list(set(valid_urls))[:max_results]

    def download_category_images(self, category: str):
        """Download images for a specific category"""
        logger.info(f"Downloading images for category: {category}")

        # Create category directory
        category_path = Path(self.config.data.data_dir) / category
        category_path.mkdir(parents=True, exist_ok=True)

        # Count existing images
        existing_count = len(list(category_path.glob('*.[jJ][pP][gG]'))) + \
                        len(list(category_path.glob('*.[jJ][pP][eE][gG]')))

        target_count = self.target_counts[category]
        if existing_count >= target_count:
            logger.info(f"Category {category} already has {existing_count} images (target: {target_count})")
            return

        # Calculate how many more images we need
        needed_count = target_count - existing_count

        # Generate search queries based on category
        search_queries = self._get_search_queries(category)

        downloaded_count = 0
        failed_urls = set()

        with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers to avoid rate limiting
            for query in search_queries:
                if downloaded_count >= needed_count:
                    break

                logger.info(f"Searching with query: {query}")
                image_urls = self.search_images(query, max_results=needed_count)

                if not image_urls:
                    logger.warning(f"No images found for query: {query}")
                    continue

                for url in image_urls:
                    if downloaded_count >= needed_count:
                        break

                    if url in failed_urls:
                        continue

                    # Generate filename from URL
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    image_path = category_path / f"{category}_{url_hash}.jpg"

                    if not image_path.exists():
                        future = executor.submit(self.download_image, url, image_path)
                        if future.result():
                            downloaded_count += 1
                            logger.info(f"Downloaded {downloaded_count}/{needed_count} for {category}")
                        else:
                            failed_urls.add(url)

                    # Random delay between 1-3 seconds
                    time.sleep(random.uniform(1, 3))

        logger.info(f"Completed downloading {downloaded_count} images for {category}")
        return downloaded_count

    def _get_search_queries(self, category: str) -> List[str]:
        """Generate search queries based on category"""
        base_queries = {
            'Bird-drop': [
                'solar panel bird droppings',
                'solar panel bird waste damage',
                'bird droppings on solar panels',
                'solar panel bird mess',
                'solar panel bird contamination',
                'pv panel bird droppings',
                'photovoltaic panel bird waste'
            ],
            'Clean': [
                'clean solar panel',
                'new solar panel installation',
                'pristine solar panel',
                'well maintained solar panel',
                'solar panel perfect condition',
                'clean photovoltaic panel',
                'solar panel after cleaning'
            ],
            'Dusty': [
                'dusty solar panel',
                'dirty solar panel',
                'solar panel dust accumulation',
                'solar panel sand dust',
                'solar panel desert dust',
                'solar panel dirt build up',
                'solar panel dust coating'
            ],
            'Electrical-damage': [
                'solar panel electrical damage',
                'solar panel hotspot damage',
                'damaged solar cell electrical',
                'solar panel burn mark',
                'solar panel electrical fault',
                'solar panel circuit damage',
                'solar panel electrical burn'
            ],
            'Physical-Damage': [
                'broken solar panel',
                'cracked solar panel',
                'solar panel physical damage',
                'solar panel hail damage',
                'solar panel storm damage',
                'solar panel crack',
                'damaged solar panel glass'
            ],
            'Snow-Covered': [
                'snow covered solar panel',
                'solar panel snow',
                'solar panel winter snow',
                'snow on solar panels',
                'solar panel ice snow',
                'solar array snow covered',
                'snowy solar panels'
            ]
        }

        return base_queries.get(category, [])