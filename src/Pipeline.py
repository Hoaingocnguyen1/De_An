"""
src/Pipeline.py
Optimized pipeline with fixes for multimodal embedding integration
"""

import asyncio
import io
from typing import List, Dict, Any
from datetime import datetime
import logging
from bson import ObjectId
from PIL import Image

from .storage.mongodb_handler import MongoDBHandler
from .storage.schema import KnowledgeUnit
from .extraction.pdf import PDFExtractor
from .enrichment.enricher import ContentEnricher
from .enrichment.clients.base_client import BaseLLMClient
from .embedding.Embedder import ContentEmbedder

logger = logging.getLogger(__name__)


class OptimizedPipeline:
    """Optimized ingestion pipeline with multimodal support"""
    
    def __init__(
        self,
        mongo_handler: MongoDBHandler,
        llm_client: BaseLLMClient,
        content_embedder: ContentEmbedder,
        max_workers: int = 4,
        batch_size: int = 16
    ):
        self.db = mongo_handler
        self.batch_size = batch_size
        
        # Optimized and modular components
        self.pdf_extractor = PDFExtractor(use_layoutparser=False)
        self.enricher = ContentEnricher(client=llm_client, max_concurrent=10)
        self.embedder = content_embedder
        
        # Semaphore for controlling concurrent document processing
        self.executor = asyncio.Semaphore(max_workers)
        
        logger.info("✓ Optimized pipeline initialized")
        logger.info(f"  - LLM Client: {llm_client.__class__.__name__}")
        logger.info(f"  - Text Embedder: {content_embedder.text_embedder.model_name}")
        logger.info(f"  - Multimodal Embedder: {content_embedder.multimodal_embedder.model_name}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Max workers: {max_workers}")

    async def process_document_async(
        self,
        file_path: str,
        source_type: str,
        source_uri: str = None
    ) -> ObjectId:
        """Process a document with all optimizations"""
        async with self.executor:
            source_uri = source_uri or file_path
            source_id = self.db.create_source({
                "source_type": source_type,
                "source_uri": source_uri,
                "original_filename": file_path.split('/')[-1],
                "status": "processing",
                "processing_start": datetime.utcnow()
            })
            
            logger.info(f"⚡ Processing {source_type}: {file_path} (Source ID: {source_id})")
            start_time = datetime.utcnow()
            
            try:
                if source_type == "pdf":
                    kus = await self._process_pdf_optimized(file_path, source_id, source_uri)
                else:
                    raise NotImplementedError(f"Source type '{source_type}' is not supported yet.")
                
                if kus:
                    ku_ids = self.db.insert_knowledge_units(kus)
                    self.db.update_source_status(source_id, "completed", total_kus=len(ku_ids))
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    logger.info(f"✓ Completed {file_path} in {elapsed:.1f}s - {len(ku_ids)} KUs created.")
                    return source_id
                else:
                    self.db.update_source_status(source_id, "completed_with_no_content")
                    logger.warning(f"No content extracted from {file_path}.")
                    return source_id
            
            except Exception as e:
                logger.error(f"Processing failed for {file_path}: {e}", exc_info=True)
                self.db.update_source_status(source_id, "failed", str(e))
                raise

    async def _process_pdf_optimized(
        self,
        pdf_path: str,
        source_id: ObjectId,
        source_uri: str
    ) -> List[Dict]:
        """Optimized PDF processing workflow"""
        
        # Step 1: Extract in parallel
        logger.info("  [1/4] Extracting content in parallel...")
        raw_objects = self.pdf_extractor.extract_parallel(pdf_path)
        logger.info(f"  ✓ Extracted {len(raw_objects)} raw objects")

        # Step 2: Batch enrichment
        logger.info("  [2/4] Enriching content in batches...")
        items_to_enrich = []
        
        for obj in raw_objects:
            item = {
                "type": obj['type'],
                "caption": obj.get('caption', ''),
                "id": obj.get('figure_id') or obj.get('table_id', '')
            }
            
            if obj['type'] == 'text_chunk':
                item['data'] = obj['text']
            elif obj['type'] == 'table':
                item['data'] = obj.get('data', {})
            elif obj['type'] == 'figure':
                try:
                    # Convert image bytes to PIL Image
                    img_data = obj.get('image_data')
                    if isinstance(img_data, bytes):
                        item['data'] = Image.open(io.BytesIO(img_data))
                    else:
                        logger.warning(f"Skipping figure {item['id']}: invalid image data")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to load image: {e}")
                    continue
            
            items_to_enrich.append(item)
        
        # Process in batches
        enriched_results = []
        for i in range(0, len(items_to_enrich), self.batch_size):
            batch = items_to_enrich[i:i + self.batch_size]
            enriched_batch = await self.enricher.enrich_batch(batch)
            enriched_results.extend(enriched_batch)
        
        # Map enrichment back to raw objects
        enrichment_map = {item.get('id'): item.get('enrichment', {}) for item in enriched_results}
        for obj in raw_objects:
            obj_id = obj.get('figure_id') or obj.get('table_id', '')
            obj['enrichment'] = enrichment_map.get(obj_id, {})
        
        logger.info(f"  ✓ Enriched {len(enriched_results)} objects")

        # Step 3: Batch embedding
        logger.info("  [3/4] Generating embeddings in batches...")
        kus_ready = []
        text_chunks_for_embedding = []

        for obj in raw_objects:
            enrichment = obj.get('enrichment', {})
            summary = enrichment.get('summary', '')
            caption = obj.get('caption', '')
            obj_type = obj['type']

            if obj_type == 'text_chunk':
                # Text will be chunked and embedded
                text_chunks_for_embedding.append({
                    'text': obj['text'],
                    'page': obj.get('page'),
                    'bbox': obj.get('bbox')
                })
                
            elif obj_type == 'table':
                # Embed table using multimodal embedder
                embedding_data = self.embedder.embed_table(
                    table_data=obj.get('data', {}),
                    caption=caption,
                    summary=summary,
                    table_image=None  # Could render table as image if needed
                )
                
                kus_ready.append({
                    'type': obj_type,
                    'page': obj.get('page'),
                    'bbox': obj.get('bbox'),
                    'data': obj.get('data'),
                    'caption': caption,
                    'enrichment': enrichment,
                    'embedding_vector': embedding_data['vector'],
                    'source_text': embedding_data['source_text'],
                    'embedding_type': embedding_data.get('embedding_type', 'text')
                })
                
            elif obj_type == 'figure':
                # Embed figure using multimodal embedder
                try:
                    img_data = obj.get('image_data')
                    if isinstance(img_data, bytes):
                        image = Image.open(io.BytesIO(img_data))
                    else:
                        logger.warning("Invalid image data for figure")
                        continue
                    
                    embedding_data = self.embedder.embed_figure(
                        image=image,
                        caption=caption,
                        summary=summary
                    )
                    
                    kus_ready.append({
                        'type': obj_type,
                        'page': obj.get('page'),
                        'bbox': obj.get('bbox'),
                        'figure_id': obj.get('figure_id'),
                        'width': obj.get('width'),
                        'height': obj.get('height'),
                        'caption': caption,
                        'enrichment': enrichment,
                        'embedding_vector': embedding_data['vector'],
                        'source_text': embedding_data['source_text'],
                        'embedding_type': embedding_data.get('embedding_type', 'multimodal')
                    })
                except Exception as e:
                    logger.error(f"Failed to embed figure: {e}")
                    continue

        # Process text chunks
        for text_chunk_data in text_chunks_for_embedding:
            text = text_chunk_data['text']
            chunks = self.embedder.embed_text_chunk(text)
            
            for chunk in chunks:
                kus_ready.append({
                    'type': 'text_chunk',
                    'page': text_chunk_data.get('page'),
                    'bbox': text_chunk_data.get('bbox'),
                    'text': chunk['text_chunk'],
                    'embedding_vector': chunk['vector'],
                    'source_text': chunk['text_chunk'],
                    'embedding_type': 'text',
                    'enrichment': {}  # Text chunks typically don't need enrichment
                })
        
        logger.info(f"  ✓ Generated embeddings for {len(kus_ready)} knowledge units")

        # Step 4: Create Knowledge Units
        logger.info("  [4/4] Finalizing Knowledge Units...")
        final_kus = self._create_knowledge_units(
            kus_ready, source_id, source_type, source_uri
        )
        
        return final_kus

    def _create_knowledge_units(
        self,
        objects: List[Dict],
        source_id: ObjectId,
        source_type: str,
        source_uri: str
    ) -> List[Dict]:
        """Convert processed objects into MongoDB-ready Knowledge Units"""
        kus = []
        
        for idx, obj in enumerate(objects):
            obj_type = obj.get('type')
            enrichment = obj.get('enrichment', {})
            
            # Build KU according to schema
            ku = {
                "source_id": source_id,
                "source_type": source_type,
                "source_uri": source_uri,
                "ku_id": f"{source_type}_{source_id}_{obj_type}_{idx}",
                "ku_type": obj_type,
                "created_at": datetime.utcnow(),
                
                # Raw content
                "raw_content": {
                    "text": obj.get('text') or obj.get('source_text'),
                    "asset_uri": None,  # Could upload to GCS/S3
                },
                
                # Context
                "context": {
                    "page_number": obj.get('page'),
                    "bounding_box": obj.get('bbox'),
                    "start_time_seconds": None,
                    "end_time_seconds": None,
                    "direct_url": None
                },
                
                # Enriched content
                "enriched_content": {
                    "summary": enrichment.get('summary'),
                    "keywords": enrichment.get('keywords', []),
                    "analysis_model": "qwen-vl-max"
                } if enrichment else None,
                
                # Embeddings
                "embeddings": {
                    "vector": obj.get('embedding_vector', []),
                    "model": self.embedder.text_embedder.model_name 
                            if obj.get('embedding_type') == 'text' 
                            else self.embedder.multimodal_embedder.model_name
                }
            }
            
            # Add type-specific metadata
            if obj_type == 'table':
                ku['raw_content']['table_data'] = obj.get('data')
                ku['context']['caption'] = obj.get('caption')
            elif obj_type == 'figure':
                ku['context']['figure_id'] = obj.get('figure_id')
                ku['context']['caption'] = obj.get('caption')
                ku['context']['dimensions'] = {
                    'width': obj.get('width'),
                    'height': obj.get('height')
                }
            
            kus.append(ku)
        
        return kus

    def process_document(
        self,
        file_path: str,
        source_type: str,
        source_uri: str = None
    ) -> ObjectId:
        """Synchronous wrapper for async processing"""
        return asyncio.run(
            self.process_document_async(file_path, source_type, source_uri)
        )