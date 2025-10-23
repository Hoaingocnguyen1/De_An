"""
src/enrichment/clients/base_client.py
Optimized abstract base class with better error handling and retry logic
"""
import base64
import io
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, ValidationError
from PIL import Image

from ..schema import EnrichmentOutput

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """
    Enhanced base class for LLM clients with:
    - Exponential backoff retry
    - Better error handling
    - Response caching
    - Rate limiting
    """
    
    def __init__(self):
        self._cache = {}
        self._rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent calls
    
    @abstractmethod
    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Must be implemented by subclasses
        Should return raw JSON string
        """
        pass
    
    async def _call_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> Optional[str]:
        """
        Call LLM with exponential backoff retry
        """
        for attempt in range(max_retries):
            try:
                async with self._rate_limiter:
                    return await self._call_llm(messages)
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed: {e}")
                    return None
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        return None
    
    def _clean_json_response(self, raw_response: str) -> str:
        """
        Clean LLM response to extract valid JSON
        Handles markdown code blocks and extra text
        """
        if not raw_response:
            return ""
        
        # Remove markdown code blocks
        text = raw_response.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        
        # Try to find JSON object
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        
        return text
    
    async def create_structured_completion(
        self,
        prompt: str,
        response_model: Type[BaseModel] = EnrichmentOutput,
        image: Optional[Image.Image] = None,
        max_retries: int = 3,
        use_cache: bool = True
    ) -> Optional[BaseModel]:
        """
        Enhanced structured completion with caching and better error handling
        """
        # Create cache key
        cache_key = None
        if use_cache:
            cache_key = hash((prompt, str(response_model), id(image)))
            if cache_key in self._cache:
                logger.debug("Cache hit for structured completion")
                return self._cache[cache_key]
        
        # Build format instructions
        schema = response_model.model_json_schema()
        
        full_prompt = f"""{prompt}

Please provide your response in valid JSON format following this schema.
Do not include explanatory text before or after the JSON.

Schema:
{json.dumps(schema, indent=2)}

Response (JSON only):"""
        
        # Build message with image if provided
        content = [{"type": "text", "text": full_prompt}]
        
        if image:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_url = f"data:image/png;base64,{img_base64}"
            content.insert(0, {"type": "image_url", "image_url": {"url": image_url}})
        
        messages = [{"role": "user", "content": content}]
        
        # Try to get valid response with retries
        for attempt in range(max_retries):
            try:
                raw_response = await self._call_with_retry(messages)
                
                if not raw_response:
                    continue
                
                # Clean and parse JSON
                cleaned = self._clean_json_response(raw_response)
                
                if not cleaned:
                    logger.warning(f"Empty response after cleaning (attempt {attempt + 1})")
                    continue
                
                # Parse with Pydantic
                result = response_model.model_validate_json(cleaned)
                
                # Cache successful result
                if use_cache and cache_key:
                    self._cache[cache_key] = result
                
                return result
            
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}): {e}")
                logger.debug(f"Raw response: {raw_response[:200]}...")
                
            except ValidationError as e:
                logger.warning(f"Validation error (attempt {attempt + 1}): {e}")
                logger.debug(f"Cleaned JSON: {cleaned[:200]}...")
            
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
        
        logger.error(f"Failed to get valid structured output after {max_retries} attempts")
        return None
    
    async def synthesize_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Synthesize answer from context
        Default implementation - can be overridden
        """
        if context:
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = question
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        try:
            response = await self._call_with_retry(messages)
            return response.strip() if response else "I apologize, but I couldn't generate an answer."
        
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"An error occurred while synthesizing the answer: {str(e)}"
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
        logger.info("Cache cleared")