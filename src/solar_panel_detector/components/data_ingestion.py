import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from ..utils.logger import logger
from ..config.configuration import Config
from pathlib import Path
import shutil
from typing import List, Dict
import concurrent.futures

class DataIngestion:
    def __init__(self, config: Config):
        self.config = config
        self.categories = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(exist_ok=True)

    def count_category_data(self) -> Dict[str, int]:
        """Count images in each category"""
        counts = {}
        for category in self.categories:
            category_path = self.config.data.data_dir / category
            if category_path.exists():
                counts[category] = len(list(category_path.glob('*.[jJ][pP][gG]')))
                counts[category] += len(list(category_path.glob('*.[jJ][pP][eE][gG]')))
                logger.info(f"Category {category}: {counts[category]} images")
        return counts

    def _download_image(self, url: str, save_path: Path) -> bool:
        """Download a single image from URL"""
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
        return False

    def scrape_images(self, query: str, max_images: int = 5000) -> List[str]:
        """Scrape images from web search engines"""
        try:
            from selenium.webdriver.chrome.options import Options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            
            driver = webdriver.Chrome(options=chrome_options)
            image_urls = []
            
            search_url = f"https://www.google.com/search?q={query}&tbm=isch"
            driver.get(search_url)
            
            last_height = driver.execute_script("return document.body.scrollHeight")
            while len(image_urls) < max_images:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                
                images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
                current_urls = [img.get_attribute('src') for img in images if img.get_attribute('src')]
                image_urls.extend(current_urls)
                
                if len(image_urls) >= max_images:
                    image_urls = image_urls[:max_images]
                    break
                    
            driver.quit()
            return image_urls
            
        except Exception as e:
            logger.error(f"Error in web scraping: {str(e)}")
            return []

    def download_category_images(self, category: str):
        """Download images for a specific category"""
        save_dir = self.download_dir / category
        save_dir.mkdir(exist_ok=True)
        
        current_count = self.count_category_data().get(category, 0)
        if current_count >= 5000:
            logger.info(f"Category {category} already has sufficient images")
            return
        
        needed_images = 5000 - current_count
        query = f"solar panel {category.lower()} fault"
        urls = self.scrape_images(query, needed_images)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                executor.submit(
                    self._download_image, 
                    url, 
                    save_dir / f"{category}_{i}.jpg"
                ): url for i, url in enumerate(urls)
            }
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        logger.info(f"Successfully downloaded image from {url}")
                    else:
                        logger.warning(f"Failed to download image from {url}")
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")