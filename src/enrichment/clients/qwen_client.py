# src/enrich/qwen_client.py
import dashscope
from typing import Dict, Any, List, Optional

from .base_client import BaseLLMClient

class QwenClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "qwen-vl-max"):
        dashscope.api_key = api_key
        self.model_name = model_name

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        # Dịch từ định dạng OpenAI sang định dạng của DashScope
        qwen_contents = []
        user_message_content = messages[-1]['content']
        
        for part in user_message_content:
            if part['type'] == 'text':
                qwen_contents.append({'text': part['text']})
            elif part['type'] == 'image_url':
                # DashScope có thể dùng trực tiếp data URL
                qwen_contents.append({'image': part['image_url']['url']})
        
        qwen_messages = [{'role': 'user', 'content': qwen_contents}]
        
        response = await dashscope.MultimodalConversation.async_call(
            model=self.model_name,
            messages=qwen_messages
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError(f"Qwen API Error: {response.message}")