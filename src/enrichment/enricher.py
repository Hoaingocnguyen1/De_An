# src/enrichment/enricher.py
import asyncio
from typing import List, Dict, Any
import logging
from PIL import Image
import pandas as pd

from .clients.base_client import BaseLLMClient
from .schema import EnrichmentOutput

logger = logging.getLogger(__name__)

class ContentEnricher:
    def __init__(self, client: BaseLLMClient, max_concurrent: int = 10):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _enrich_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_type = item.get("type")
        data = item.get("data")
        caption = item.get("caption", "")

        async with self.semaphore:
            prompt = ""
            image = None
            
            if item_type == "text":
                prompt = f"Analyze the following text:\n\n\"{data}\""
            elif item_type == "table":
                table_str = pd.DataFrame(data).to_string(index=False, max_rows=20)
                prompt = f"Analyze the following table with caption \"{caption}\":\n\n{table_str}"
            elif item_type == "image":
                prompt = f"Analyze this image with caption \"{caption}\"."
                image = data # `data` bây giờ là một đối tượng PIL.Image
            else:
                item.update({"status": "failed", "error": "Unsupported type"})
                return item

            # Gọi phương thức tạo completion có cấu trúc
            enrichment_result = await self.client.create_structured_completion(prompt, image=image)

            if enrichment_result:
                # Chuyển đổi đối tượng Pydantic thành dict để lưu
                item.update({"enrichment": enrichment_result.model_dump(), "status": "success"})
            else:
                item.update({"enrichment": None, "status": "failed", "error": "LLM failed to produce valid output."})
            
            return item

    async def enrich_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items: return []
        tasks = [self._enrich_item(item) for item in items]
        return await asyncio.gather(*tasks)
    

# # main.py
# import asyncio
# import os
# from PIL import Image
# from dotenv import load_dotenv

# from src.enrichment.enricher import ContentEnricher
# from src.enrichment.clients.gemini_client import GeminiClient
# from src.enrichment.clients.qwen_client import QwenClient

# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# async def main():
#     # --- CHỌN CLIENT BẠN MUỐN SỬ DỤNG ---
#     # client_choice = "gemini"
#     client_choice = "qwen"
    
#     if client_choice == "gemini":
#         if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not found!")
#         client = GeminiClient(api_key=GEMINI_API_KEY)
#     elif client_choice == "qwen":
#         if not DASHSCOPE_API_KEY: raise ValueError("DASHSCOPE_API_KEY not found!")
#         client = QwenClient(api_key=DASHSCOPE_API_KEY)
#     else:
#         raise ValueError("Invalid client choice")

#     enricher = ContentEnricher(client=client, max_concurrent=5)
    
#     # ... (Phần chuẩn bị dữ liệu `items_to_process` giữ nguyên như trước) ...
#     text_sample = "..."
#     table_sample_data = {...}
#     image_sample = Image.new('RGB', (200, 150), color = 'blue')
#     items_to_process = [
#         {"id": "text_01", "type": "text", "data": text_sample},
#         {"id": "table_01", "type": "table", "data": table_sample_data, "caption": "Model Comparison"},
#         {"id": "image_01", "type": "image", "data": image_sample, "caption": "Performance Chart"}
#     ]

#     results = await enricher.enrich_batch(items_to_process)

#     # In kết quả
#     for result in results:
#         print("\n" + "="*50)
#         print(f"ID: {result['id']} | Type: {result['type']} | Status: {result['status']}")
#         if result['status'] == 'success':
#             print(f"Enrichment Data from {client_choice.upper()}:")
#             print(result['enrichment']) # Kết quả đã là một dictionary đẹp
#         else:
#             print(f"Error: {result.get('error')}")
#         print("="*50)

# if __name__ == "__main__":
#     asyncio.run(main())