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

class DataIngestion:
    def __init__(self, config: Config):
        self.config = config
        self.categories = [
            'Bird-drop', 'Clean', 'Dusty', 
            'Electrical-damage', 'Physical-Damage', 'Snow-Covered'
        ]
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
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
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                # Validate image
                try:
                    img = Image.open(io.BytesIO(response.content))
                    img = img.convert('RGB')  # Convert to RGB if needed
                    
                    # Basic image validation
                    if img.size[0] < 100 or img.size[1] < 100:
                        return False
                    
                    # Save image
                    img.save(save_path)
                    return True
                except:
                    return False
            return False
        except:
            return False
            
    def search_images(self, query: str, max_results: int = 100) -> List[str]:
        """Search for images using multiple search engines"""
        image_urls = set()
        
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
                        'q': query
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'items' in data:
                        for item in data['items']:
                            if len(image_urls) >= max_results:
                                break
                            image_urls.add(item['link'])
            except Exception as e:
                logger.error(f"Error using Google Custom Search: {str(e)}")
        
        # If we still need more images, try DuckDuckGo
        if len(image_urls) < max_results:
            try:
                # DuckDuckGo image search
                ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}&iax=images&ia=images"
                response = requests.get(ddg_url, headers=self.headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    image_elements = soup.select('img[data-src]')
                    
                    for img in image_elements:
                        if len(image_urls) >= max_results:
                            break
                        if 'src' in img.attrs:
                            image_urls.add(img['src'])
                        elif 'data-src' in img.attrs:
                            image_urls.add(img['data-src'])
            except Exception as e:
                logger.error(f"Error using DuckDuckGo search: {str(e)}")
        
        # If still need more images, try Bing
        if len(image_urls) < max_results:
            try:
                bing_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&qft=+filterui:photo-photo"
                response = requests.get(bing_url, headers=self.headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    image_elements = soup.select('.mimg')
                    
                    for img in image_elements:
                        if len(image_urls) >= max_results:
                            break
                        if 'src' in img.attrs:
                            image_urls.add(img['src'])
            except Exception as e:
                logger.error(f"Error using Bing search: {str(e)}")
        
        return list(image_urls)[:max_results]
    
    def download_category_images(self, category: str, target_count: int = 5000):
        """Download images for a specific category"""
        logger.info(f"Downloading images for category: {category}")
        
        # Create category directory
        category_path = Path(self.config.data.data_dir) / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        # Count existing images
        existing_count = len(list(category_path.glob('*.[jJ][pP][gG]'))) + \
                        len(list(category_path.glob('*.[jJ][pP][eE][gG]')))
        
        if existing_count >= target_count:
            logger.info(f"Category {category} already has {existing_count} images")
            return
        
        # Calculate how many more images we need
        needed_count = target_count - existing_count
        
        # Generate search queries
        search_queries = [
            f"solar panel {category.lower()}",
            f"solar panel {category.lower()} fault",
            f"photovoltaic panel {category.lower()}",
            f"solar module {category.lower()} damage"
        ]
        
        downloaded_count = 0
        failed_urls = set()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            for query in search_queries:
                if downloaded_count >= needed_count:
                    break
                    
                image_urls = self.search_images(query, max_results=needed_count)
                
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
                    
                    # Add delay to avoid hitting rate limits
                    time.sleep(0.1)
        
        logger.info(f"Completed downloading {downloaded_count} images for {category}")
        return downloaded_count