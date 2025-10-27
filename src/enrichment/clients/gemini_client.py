"""
src/enrichment/clients/gemini_client.py
Gemini client using OpenAI SDK for better structured outputs
"""
from openai import AsyncOpenAI
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional, Type
import logging
import json

from pydantic import BaseModel, ValidationError

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Gemini client using OpenAI SDK for structured outputs.
    Uses Google's OpenAI-compatible API endpoint.
    """
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7
    ):
        super().__init__()  # Initialize parent with cache
        
        # Initialize OpenAI client with Google AI endpoint
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        self.model_name = model_name
        self.temperature = temperature
        
        logger.info(f"✓ GeminiClient initialized with OpenAI SDK")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Temperature: {temperature}")

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Call Gemini via OpenAI SDK
        Handles both text and multimodal content
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None

    async def synthesize_answer(
        self, 
        question: str, 
        context: str
    ) -> str:
        """
        Synthesize answer from context using chat completion
        """
        try:
            # Build messages
            if context:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Provide clear, accurate answers based on the given context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                logger.warning("Empty response from Gemini")
                return "I apologize, but I couldn't generate a complete answer."
                
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"

    async def create_structured_completion(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        image: Optional[Image.Image] = None,
        max_retries: int = 2,
        use_cache: bool = True
    ) -> Optional[BaseModel]:
        """
        Create structured completion using OpenAI's response_format
        This enforces strict JSON schema compliance
        """
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = hash((prompt, str(response_model), id(image)))
            if cache_key in self._cache:
                logger.debug("✓ Cache hit for structured completion")
                return self._cache[cache_key]
        
        # Build message content
        if image:
            # Multimodal message
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image(image)
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            # Text-only
            content = prompt
        
        messages = [
            {
                "role": "system",
                "content": "You are a precise data extraction assistant. Always respond with valid JSON matching the required schema. Never include explanations outside the JSON structure."
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Get JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                #  Use response_format for structured output
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistency
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "schema": schema,
                            "strict": True  # ✅ CRITICAL: Enforce strict schema
                        }
                    }
                )
                
                if not response or not response.choices:
                    logger.warning(f"Empty response (attempt {attempt + 1})")
                    continue
                
                raw_json = response.choices[0].message.content
                
                if not raw_json:
                    logger.warning(f"Empty content (attempt {attempt + 1})")
                    continue
                
                # ✅ Log for debugging
                logger.debug(f"Raw response: {raw_json[:200]}...")
                
                # Parse and validate with Pydantic
                try:
                    result = response_model.model_validate_json(raw_json)
                except ValidationError as ve:
                    logger.warning(f"Validation error (attempt {attempt + 1}): {ve}")
                    
                    # Try parsing as dict first
                    try:
                        data = json.loads(raw_json)
                        result = response_model.model_validate(data)
                    except Exception as e:
                        logger.error(f"Failed to parse/validate: {e}")
                        if attempt == max_retries - 1:
                            raise
                        continue
                
                #  Validate required fields
                if not self._validate_required_fields(result, response_model):
                    logger.warning(f"Missing required fields (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        continue
                    # On last attempt, return partial result
                    logger.warning("Returning partial result with missing fields")
                
                # Cache and return
                if use_cache and cache_key:
                    self._cache[cache_key] = result
                
                logger.info(f"✓ Structured output validated: {response_model.__name__}")
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f" All {max_retries} attempts failed for {response_model.__name__}")
                    return None
                
                # Add delay before retry
                import asyncio
                await asyncio.sleep(1 * (attempt + 1))
        
        return None
    
    def _validate_required_fields(
        self, 
        result: BaseModel, 
        model_class: Type[BaseModel]
    ) -> bool:
        """
        Validate that all required fields are present and meaningful
        """
        schema = model_class.model_json_schema()
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            value = getattr(result, field, None)
            
            # Check if field exists
            if value is None:
                logger.warning(f"Required field '{field}' is None")
                return False
            
            # For string fields, check not empty
            if isinstance(value, str) and len(value.strip()) == 0:
                logger.warning(f"Required string field '{field}' is empty")
                return False
            
            # For list fields, allow empty for certain fields
            if isinstance(value, list) and len(value) == 0:
                # These fields can be empty
                allowed_empty = [
                    "keywords", 
                    "numerical_data", 
                    "labels_detected",
                    "citations_mentioned",
                    "datasets_mentioned",
                    "contributions"
                ]
                if field not in allowed_empty:
                    logger.warning(f"Required list field '{field}' is empty")
                    return False
        
        return True

    async def enrich_content(
        self,
        content: str,
        content_type: str = "text",
        image: Optional[Image.Image] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Legacy method for backward compatibility
        Uses create_structured_completion internally
        """
        from ..schema import EnrichmentOutput
        
        # Build appropriate prompt
        if content_type == "table":
            prompt = f"""Analyze this table and extract:
1. A concise summary of key findings
2. Important keywords/concepts

Table:
{content[:1000]}"""
        
        elif content_type == "figure":
            prompt = f"""Analyze this figure and extract:
1. Detailed description of what it shows
2. Key insights
3. Important keywords

Caption: {content}"""
        
        else:  # text
            prompt = f"""Analyze this text and extract:
1. Concise summary
2. Key keywords/concepts

Text:
{content[:1000]}"""
        
        result = await self.create_structured_completion(
            prompt=prompt,
            response_model=EnrichmentOutput,
            image=image
        )
        
        return result.model_dump() if result else None