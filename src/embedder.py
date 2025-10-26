"""
src/embedder.py
FIXED: Voyage AI embedding với API chính xác
"""
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, api_key: str, model_name: str = "voyage-3"):
        self.client = voyageai.Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"TextEmbedder initialized with Voyage AI model: {self.model_name}")

    def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """Embed batch of texts - FIXED API call"""
        if not texts:
            return []
        try:
            result = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type
            )
            return result.embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings from Voyage AI: {e}")
            return [[] for _ in texts]

    def embed(self, text: str, input_type: str = "document") -> List[float]:
        """Embed single text"""
        results = self.embed_batch([text], input_type=input_type)
        return results[0] if results else []

    def chunk_and_embed(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Chunk text and embed each chunk"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            return []
        
        embeddings = self.embed_batch(chunks)
        return [
            {"text_chunk": chunk, "vector": embedding}
            for chunk, embedding in zip(chunks, embeddings) if embedding
        ]


class MultimodalEmbedder:
    """Voyage Multimodal 3 embedder for figures and tables"""

    def __init__(self, api_key: str, model_name: str = "voyage-multimodal-3"):
        self.client = voyageai.Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"MultimodalEmbedder initialized with model: {self.model_name}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def embed_image(self, image: Image.Image) -> List[float]:
        """Embed a single image - FIXED API call"""
        try:
            base64_image = self._image_to_base64(image)
            result = self.client.multimodal_embed(
                inputs=[[base64_image]],
                model=self.model_name
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Failed to embed image: {e}")
            return []

    def embed_image_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """Embed multiple images in batch"""
        try:
            base64_images = [[self._image_to_base64(img)] for img in images]
            result = self.client.multimodal_embed(
                inputs=base64_images,
                model=self.model_name
            )
            return result.embeddings
        except Exception as e:
            logger.error(f"Failed to embed image batch: {e}")
            return [[] for _ in images]

    def embed_table_with_image(
        self,
        table_image: Optional[Image.Image] = None,
        table_text: Optional[str] = None
    ) -> List[float]:
        """
        Embed table using both visual and textual representation
        """
        try:
            inputs = []
            if table_image and table_text:
                base64_image = self._image_to_base64(table_image)
                inputs = [[base64_image, table_text]]
            elif table_image:
                base64_image = self._image_to_base64(table_image)
                inputs = [[base64_image]]
            elif table_text:
                # For text-only tables, use text embedder instead
                logger.warning("Text-only table, switching to text embedding")
                return []
            else:
                logger.warning("No table content provided")
                return []

            result = self.client.multimodal_embed(
                inputs=inputs,
                model=self.model_name
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Failed to embed table: {e}")
            return []


class ContentEmbedder:
    """Unified content embedder using both text and multimodal models"""

    def __init__(
        self,
        text_embedder: TextEmbedder,
        multimodal_embedder: MultimodalEmbedder
    ):
        self.text_embedder = text_embedder
        self.multimodal_embedder = multimodal_embedder
        logger.info("ContentEmbedder initialized with text and multimodal embedders")

    def embed_figure(
        self,
        image: Image.Image,
        caption: str = "",
        summary: str = ""
    ) -> Dict[str, Any]:
        """Create embedding for a figure using multimodal model"""
        text_description = f"Figure Caption: {caption}\n\nDetailed Analysis: {summary}"
        
        # Try multimodal embedding first
        vector = self.multimodal_embedder.embed_table_with_image(
            table_image=image,
            table_text=text_description
        )
        
        # Fallback to text embedding if multimodal fails
        if not vector:
            logger.warning("Multimodal embedding failed, falling back to text embedding")
            vector = self.text_embedder.embed(text_description)
        embedding_type = "multimodal" if vector and len(vector) > 0 and self.multimodal_embedder else "text"
        model_used = self.multimodal_embedder.model_name if embedding_type == "multimodal" else self.text_embedder.model_name

        return {
            "vector": vector,
            "source_text": text_description,
            "embedding_type": embedding_type,
            "model": model_used
        }

    def embed_table(
        self,
        table_data: Dict,
        caption: str = "",
        summary: str = "",
        table_image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """Create embedding for a table - FIXED to always use text embedding"""
        try:
            table_str = pd.DataFrame(table_data).to_markdown(index=False, tablefmt="pipe")
        except Exception as e:
            logger.error(f"Could not convert table to markdown: {e}")
            table_str = str(table_data)

        combined_text = (
            f"Table Caption: {caption}\n\n"
            f"Summary: {summary}\n\n"
            f"Table Content:\n{table_str}"
        )

        # FIXED: Always use text embedding for tables (more reliable)
        vector = self.text_embedder.embed(combined_text)
        embedding_type = "text"
        
        # Optional: Try multimodal if we have image
        if table_image and not vector:
            try:
                vector = self.multimodal_embedder.embed_table_with_image(
                    table_image=table_image,
                    table_text=combined_text
                )
                embedding_type = "multimodal"
            except Exception as e:
                logger.warning(f"Multimodal table embedding failed: {e}")

        return {
            "vector": vector,
            "source_text": combined_text,
            "embedding_type": embedding_type,
            "model": self.text_embedder.model_name if embedding_type == "text" else self.multimodal_embedder.model_name
        }

    def embed_text_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Embed text chunks using text model"""
        return self.text_embedder.chunk_and_embed(text)