"""
src/enrichment/clients/gemini_client.py
Enhanced Gemini client with synthesis capabilities for Gemini 2.0 Flash
"""
import google.generativeai as genai
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional
import logging

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Gemini client optimized for both enrichment and answer synthesis
    Uses Gemini 2.0 Flash for fast, high-quality responses
    """
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7
    ):
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
        
        Args:
            question: User's question (can include context in prompt)
            context: Retrieved context (can be empty if already in question)
        
        Returns:
            Synthesized answer string
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