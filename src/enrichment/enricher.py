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
            logger.error(f"self.client = {self.client}, type = {type(self.client)}")
            # Gọi phương thức tạo completion có cấu trúc
            enrichment_result = await self.client.create_structured_completion(prompt=prompt, image=image)
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
    