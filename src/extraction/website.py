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
        """
        Khởi tạo bộ điều phối trích xuất website.

        Args:
            max_workers: Số lượng URL tối đa để xử lý song song.
                         (Nên đặt cao hơn cho tác vụ I/O-bound như request mạng).
        """
        self.max_workers = max_workers

    async def extract_from_urls(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """
        Trích xuất nội dung văn bản chính từ một danh sách các URL một cách song song.

        Args:
            urls: Danh sách các URL cần xử lý.

        Returns:
            Một dictionary ánh xạ mỗi URL với nội dung văn bản đã trích xuất (hoặc None nếu thất bại).
        """
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


# # --- VÍ DỤ SỬ DỤNG ---
# async def main():
#     # Danh sách các URL mẫu để trích xuất
#     urls_to_process = [
#         "https://vnexpress.net/elon-musk-muon-bien-x-thanh-trung-tam-tai-chinh-4700938.html",
#         "https://www.theverge.com/2023/12/21/24011097/ Humane-ai-pin-first-impressions-hands-on-demo",
#         "https://blog.gopenai.com/new-embedding-models-and-api-updates", # OpenAI Blog
#         "https://non-existent-url-12345.com", # URL lỗi để kiểm tra xử lý lỗi
#         "https://github.com/adbar/trafilatura" # Trang Github (ít văn bản chính)
#     ]

#     # Khởi tạo và chạy trình trích xuất
#     extractor = WebsiteExtractor(max_workers=10)
#     extracted_data = await extractor.extract_from_urls(urls_to_process)

#     # In kết quả
#     for url, text in extracted_data.items():
#         print("\n" + "="*80)
#         print(f"URL: {url}")
#         print("="*80)
#         if text:
#             # In 300 ký tự đầu tiên của nội dung
#             print(f"Extracted Text (first 300 chars):\n\n{text[:300]}...")
#         else:
#             print("--> Failed to extract content.")
#         print("="*80)


# if __name__ == "__main__":
#     asyncio.run(main())