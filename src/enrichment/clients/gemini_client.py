import google.generativeai as genai
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional
import logging
from google.generativeai.types import GenerationConfig
import asyncio
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, ValidationError
import json

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7
    ):
        # Call the parent class's initializer
        # Remove unsupported/extra JSON Schema keywords that Gemini rejects (e.g., "title", "description", "$id", "$schema").
        # Keep a conservative allowlist of commonly-used keywords Gemini accepts.
        ALLOWED_KEYS = {
            "type",
            "properties",
            "items",
            "additionalProperties",
            "enum",
            "required",
            "anyOf",
            "oneOf",
            "allOf",
            "minimum",
            "maximum",
            "minItems",
            "maxItems",
            "pattern",
            "format",
            "const",
            "$ref",
        }

        def _sanitize_schema(obj: Any) -> None:
            if isinstance(obj, dict):
                # Remove keys not in allowlist
                for k in list(obj.keys()):
                    if k not in ALLOWED_KEYS:
                        obj.pop(k, None)

                # Recurse
                for v in obj.values():
                    _sanitize_schema(v)

            elif isinstance(obj, list):
                for item in obj:
                    _sanitize_schema(item)

        # Sanitize schema before further patching
        try:
            _sanitize_schema(response_schema)
        except Exception:
            # If sanitization fails, continue with best-effort schema
            pass

        def _patch_empty_objects(obj: Any) -> None:
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize model with generation config
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        logger.info(f"GeminiClient initialized with model: {model_name}")

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Call Gemini API with OpenAI-style messages
        Converts format and handles multimodal content
        """
        try:
            # Convert OpenAI format to Gemini format
            gemini_contents = []
            user_message_content = messages[-1]['content']

            for part in user_message_content:
                if part['type'] == 'text':
                    gemini_contents.append(part['text'])
                elif part['type'] == 'image_url':
                    # Decode base64 and create PIL Image
                    base64_str = part['image_url']['url'].split(',')[1]
                    image_data = base64.b64decode(base64_str)
                    image = Image.open(io.BytesIO(image_data))
                    gemini_contents.append(image)
            
            # Generate content asynchronously
            response = await self.model.generate_content_async(gemini_contents)
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None

    async def synthesize_answer(
        self, 
        question: str, 
        context: str
    ) -> str:
        """
        Synthesize a comprehensive answer from retrieved contexts
        Optimized for RAG synthesis with clear, accurate responses
        """
        try:
            # If context is provided separately, combine with question
            if context:
                full_prompt = f"""Context:\n{context}\n\nQuestion: {question}"""
            else:
                full_prompt = question
            
            # Generate answer
            response = await self.model.generate_content_async(full_prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini synthesis")
                return "I apologize, but I couldn't generate a complete answer. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return f"An error occurred while synthesizing the answer: {str(e)}"

    async def enrich_content(
        self,
        content: str,
        content_type: str = "text",
        image: Optional[Image.Image] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich content with summary and keywords
        Can be used for both text and multimodal content
        
        Args:
            content: Text content to enrich
            content_type: Type of content (text, table, figure)
            image: Optional image for multimodal enrichment
        
        Returns:
            Dict with 'summary' and 'keywords'
        """
        try:
            # Build enrichment prompt
            if content_type == "table":
                prompt = f"""Analyze this table and provide:
1. A concise summary of the key findings
2. A list of important keywords/concepts

Table:
{content}

Provide your response in JSON format:
{{"summary": "...", "keywords": ["...", "..."]}}"""
            
            elif content_type == "figure":
                prompt = f"""Analyze this figure and provide:
1. A detailed description of what the figure shows
2. Key insights or findings
3. Important keywords/concepts

Caption: {content}

Provide your response in JSON format:
{{"summary": "...", "keywords": ["...", "..."]}}"""
            
            else:  # text
                prompt = f"""Analyze this text and provide:
1. A concise summary
2. Key keywords/concepts

Text:
{content}

Provide your response in JSON format:
{{"summary": "...", "keywords": ["...", "..."]}}"""
            
            # Build content for API call
            api_content = [prompt]
            if image:
                api_content.insert(0, image)
            
            response = await self.model.generate_content_async(api_content)
            
            if response and response.text:
                # Try to parse JSON from response
                import json
                # Remove markdown code blocks if present
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                
                result = json.loads(text)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Content enrichment failed: {e}")
            return None
        
    async def create_structured_completion(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        image: Optional[Image.Image] = None,
        max_retries: int = 1,
        use_cache: bool = True
    ) -> Optional[BaseModel]:
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = hash((prompt, str(response_model), id(image)))
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Chỉ gửi prompt gốc, để Gemini tự infer từ response_model
        full_prompt = f"""{prompt}

    IMPORTANT: Return ONLY valid JSON matching the expected structure. No explanations, no markdown."""
        
        # Prepare content
        content = [full_prompt]
        if image:
            content.insert(0, image)
        
        # Configure for JSON output - KEY: Use response_schema instead of mime_type
        generation_config = GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=response_model  # Gemini sẽ enforce schema này
        )
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
                
                response = await model.generate_content_async(content)
                
                if not response or not response.text:
                    logger.warning(f"Empty response (attempt {attempt + 1})")
                    continue
                
                # Parse directly - Gemini đã enforce schema rồi
                result = response_model.model_validate_json(response.text)
                
                # Cache and return
                if use_cache and cache_key:
                    self._cache[cache_key] = result
                
                logger.info(f"✓ Structured output validated: {response_model.__name__}")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                logger.debug(f"Raw response: {response.text[:200]}")
                
            except ValidationError as e:
                logger.warning(f"Validation error (attempt {attempt + 1}): {e}")
                logger.debug(f"Response: {response.text[:200]}")
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
        
        logger.error(f"Failed after {max_retries} attempts")
        return None