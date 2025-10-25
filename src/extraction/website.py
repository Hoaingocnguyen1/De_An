"""
src/extraction/website.py
A high-performance, parallel website text content extractor.
Processes multiple URLs concurrently.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import logging
import time

from .utils import WebsiteUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebsiteExtractor:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    async def extract_from_urls(self, urls: List[str]) -> Dict[str, Optional[str]]:
        start_time = time.time()
        logger.info(f"Starting extraction for {len(urls)} URLs with {self.max_workers} workers...")

        loop = asyncio.get_running_loop()
        # ThreadPoolExecutor rất phù hợp cho các tác vụ I/O-bound (chờ mạng)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tạo một tác vụ để chạy hàm `extract_main_text` cho mỗi URL
            tasks = [
                loop.run_in_executor(executor, WebsiteUtils.extract_main_text, url)
                for url in urls
            ]
            # Chờ tất cả các tác vụ hoàn thành
            results = await asyncio.gather(*tasks)

        # Ánh xạ kết quả trở lại với URL ban đầu
        final_output = {url: result for url, result in zip(urls, results)}
        
        duration = time.time() - start_time
        successful_count = sum(1 for result in results if result is not None)
        logger.info(f"Completed extraction of {successful_count}/{len(urls)} URLs in {duration:.2f} seconds.")
        
        return final_output