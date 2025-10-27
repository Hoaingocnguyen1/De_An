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
    """
    Content enricher that uses LLM to extract summaries and keywords
    Handles text, tables, and images with proper error handling
    """
    
    def __init__(self, client: BaseLLMClient, max_concurrent: int = 10):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"ContentEnricher initialized with {client.__class__.__name__}")

    async def _enrich_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single item (text, table, or image)
        Returns the item with 'enrichment' and 'status' fields added
        """
        item_type = item.get("type")
        data = item.get("data")
        caption = item.get("caption", "")

        async with self.semaphore:
            try:
                # Build appropriate prompt based on type
                if item_type == "text":
                    prompt = f"""Analyze this text and extract:
1. A concise summary (2-3 sentences)
2. Key keywords/concepts (3-5 keywords)

Text:
{data[:1000]}"""
                    image = None
                    
                elif item_type == "table":
                    try:
                        table_str = pd.DataFrame(data).to_string(index=False, max_rows=20)
                    except Exception:
                        table_str = str(data)[:500]
                    
                    prompt = f"""Analyze this table and extract:
1. A concise summary of key findings
2. Important keywords/concepts

Table with caption "{caption}":
{table_str}"""
                    image = None
                    
                elif item_type == "image":
                    prompt = f"""Analyze this image and extract:
1. Detailed description of what it shows
2. Key insights or findings
3. Important keywords/concepts

Caption: {caption}"""
                    image = data  # PIL Image object
                    
                else:
                    item.update({
                        "status": "failed", 
                        "error": f"Unsupported type: {item_type}"
                    })
                    return item
                
                # Call structured completion
                enrichment_result = await self.client.create_structured_completion(
                    prompt=prompt,
                    response_model=EnrichmentOutput,
                    image=image,
                    max_retries=2
                )
                
                if enrichment_result:
                    # Convert Pydantic model to dict
                    item.update({
                        "enrichment": enrichment_result.model_dump(),
                        "status": "success"
                    })
                    
                    # Log success with truncated summary
                    summary_preview = enrichment_result.summary[:50]
                    logger.debug(f"✓ Enriched {item_type}: {summary_preview}...")
                else:
                    # Enrichment failed - provide minimal fallback
                    logger.warning(f"Enrichment failed for {item_type}, using fallback")
                    item.update({
                        "enrichment": {
                            "summary": str(data)[:200] if item_type == "text" else caption,
                            "keywords": [],
                            "research_metadata": None
                        },
                        "status": "failed_with_fallback",
                        "error": "LLM failed to produce valid output"
                    })
                
            except Exception as e:
                logger.error(f"Enrichment error for {item_type}: {e}", exc_info=False)
                
                # Provide fallback enrichment
                fallback_summary = (
                    str(data)[:200] if item_type == "text" 
                    else caption if caption 
                    else "Content unavailable"
                )
                
                item.update({
                    "enrichment": {
                        "summary": fallback_summary,
                        "keywords": [],
                        "research_metadata": None
                    },
                    "status": "failed",
                    "error": str(e)
                })
            
            return item

    async def enrich_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple items concurrently
        
        Args:
            items: List of dicts with 'type' and 'data' keys
        
        Returns:
            List of enriched items with 'enrichment' and 'status' fields
        """
        if not items:
            return []
        
        logger.info(f"Enriching batch of {len(items)} items...")
        
        # Create tasks for concurrent processing
        tasks = [self._enrich_item(item) for item in items]
        
        # Gather results, catching exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Item {idx} enrichment crashed: {result}")
                
                # Provide fallback for crashed items
                processed_results.append({
                    **items[idx],
                    "enrichment": {
                        "summary": "Enrichment failed",
                        "keywords": [],
                        "research_metadata": None
                    },
                    "status": "crashed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        # Log success rate
        success_count = sum(
            1 for r in processed_results 
            if r.get("status") == "success"
        )
        logger.info(f"✓ Enrichment batch completed: {success_count}/{len(items)} successful")
        
        return processed_results