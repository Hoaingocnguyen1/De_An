"""
src/Pipeline.py
Optimized pipeline with fixes for multimodal embedding integration
"""

import asyncio
import io
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from bson import ObjectId
from PIL import Image

from .storage.MongoDBHandler import MongoDBHandler
from .storage.schema import KnowledgeUnit
from .extraction.pdf import PDFExtractor
from .enrichment.enricher import ContentEnricher
from .enrichment.clients.base_client import BaseLLMClient
from .embedder import ContentEmbedder

logger = logging.getLogger(__name__)


class OptimizedPipeline:
    """Optimized ingestion pipeline supporting PDF, YouTube, Video, Website"""
    
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
        
        # Optimized components
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
        source_uri: str = None,
        video_extractor = None,  # For youtube/video
        website_extractor = None  # For website
    ) -> ObjectId:
        """Process a document with all optimizations - supports all source types"""
        async with self.executor:
            source_uri = source_uri or file_path
            source_id = self.db.create_source({
                "source_type": source_type,
                "source_uri": source_uri,
                "original_filename": source_uri.split('/')[-1] if source_type == 'website' else file_path.split('/')[-1],
                "status": "processing",
                "processing_start": datetime.now(timezone.utc)
            })
            
            logger.info(f"⚡ Processing {source_type}: {file_path} (Source ID: {source_id})")
            start_time = datetime.now(timezone.utc)
            
            try:
                # Route to appropriate processor
                if source_type == "pdf":
                    kus = await self._process_pdf_optimized(file_path, source_id, source_uri)
                
                elif source_type == "youtube":
                    if not video_extractor:
                        raise ValueError("video_extractor required for YouTube processing")
                    kus = await self._process_youtube(source_uri, source_id, source_uri, video_extractor)
                
                elif source_type == "video":
                    if not video_extractor:
                        raise ValueError("video_extractor required for video processing")
                    kus = await self._process_video(file_path, source_id, source_uri, video_extractor)
                
                elif source_type == "website":
                    if not website_extractor:
                        raise ValueError("website_extractor required for website processing")
                    kus = await self._process_website(source_uri, source_id, source_uri, website_extractor)
                
                else:
                    raise NotImplementedError(f"Source type '{source_type}' is not supported yet.")
                
                # Save to MongoDB
                if kus:
                    ku_ids = self.db.insert_knowledge_units(kus)
                    self.db.update_source_status(source_id, "completed", total_kus=len(ku_ids))
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
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
            kus_ready, source_id, "pdf", source_uri 
        )
        
        return final_kus
    
    async def _process_youtube(
        self,
        youtube_url: str,
        source_id: ObjectId,
        source_uri: str,
        video_extractor  # Pass extractor from outside
    ) -> List[Dict]:
        """Optimized YouTube processing workflow"""
        
        # Step 1: Extract transcript
        logger.info("  [1/3] Extracting transcript...")
        transcripts = await video_extractor.extract_from_sources([youtube_url])
        transcript_data = transcripts.get(youtube_url, [])
        
        if not transcript_data:
            raise ValueError("No transcript extracted from YouTube video")
        
        logger.info(f"  ✓ Extracted {len(transcript_data)} transcript segments")
        
        # Step 2: Batch embedding
        logger.info("  [2/3] Generating embeddings in batches...")
        texts_to_embed = [seg.get('text', '').strip() for seg in transcript_data if seg.get('text', '').strip()]
        
        if not texts_to_embed:
            raise ValueError("No valid text found in transcript")
        
        # Embed all texts in batch
        vectors = self.embedder.text_embedder.embed_batch(texts_to_embed, input_type="document")
        
        logger.info(f"  ✓ Generated embeddings for {len(vectors)} segments")
        
        # Step 3: Create Knowledge Units
        logger.info("  [3/3] Finalizing Knowledge Units...")
        kus = []
        
        for idx, (segment, vector) in enumerate(zip(transcript_data, vectors)):
            text = segment.get('text', '').strip()
            if not text or not vector:
                continue
            
            ku = {
                "source_id": source_id,
                "source_type": "youtube",
                "source_uri": source_uri,
                "ku_id": f"youtube_{source_id}_segment_{idx}",
                "ku_type": "text_chunk",
                "created_at": datetime.utcnow(),
                
                # Raw content
                "raw_content": {
                    "text": text,
                    "asset_uri": None,
                },
                
                # Context
                "context": {
                    "page_number": None,
                    "bounding_box": None,
                    "start_time_seconds": segment.get('start'),
                    "end_time_seconds": segment.get('end'),
                    "direct_url": f"{youtube_url}&t={int(segment.get('start', 0))}s"
                },
                
                # Enriched content (can be added later if needed)
                "enriched_content": None,
                
                # Embeddings
                "embeddings": {
                    "vector": vector,
                    "model": self.embedder.text_embedder.model_name
                }
            }
            kus.append(ku)
        
        logger.info(f"  ✓ Created {len(kus)} knowledge units")
        return kus


    async def _process_website(
        self,
        url: str,
        source_id: ObjectId,
        source_uri: str,
        website_extractor  # Pass extractor from outside
    ) -> List[Dict]:
        """Optimized website processing workflow"""
        
        # Step 1: Extract content
        logger.info("  [1/3] Extracting website content...")
        extracted = await website_extractor.extract_from_urls([url])
        content = extracted.get(url)
        
        if not content:
            raise ValueError("No content extracted from website")
        
        logger.info(f"  ✓ Extracted {len(content)} characters")
        
        # Step 2: Chunk and embed
        logger.info("  [2/3] Chunking and embedding text...")
        text_chunks = self.embedder.text_embedder.chunk_and_embed(
            content,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        if not text_chunks:
            raise ValueError("No text chunks generated from website content")
        
        logger.info(f"  ✓ Generated {len(text_chunks)} text chunks with embeddings")
        
        # Step 3: Create Knowledge Units
        logger.info("  [3/3] Finalizing Knowledge Units...")
        kus = []
        
        for idx, chunk in enumerate(text_chunks):
            ku = {
                "source_id": source_id,
                "source_type": "website",
                "source_uri": source_uri,
                "ku_id": f"website_{source_id}_text_{idx}",
                "ku_type": "text_chunk",
                "created_at": datetime.utcnow(),
                
                # Raw content
                "raw_content": {
                    "text": chunk['text_chunk'],
                    "asset_uri": None,
                },
                
                # Context
                "context": {
                    "page_number": None,
                    "bounding_box": None,
                    "start_time_seconds": None,
                    "end_time_seconds": None,
                    "direct_url": url
                },
                
                # Enriched content
                "enriched_content": None,
                
                # Embeddings
                "embeddings": {
                    "vector": chunk['vector'],
                    "model": self.embedder.text_embedder.model_name
                }
            }
            kus.append(ku)
        
        logger.info(f"  ✓ Created {len(kus)} knowledge units")
        return kus


    async def _process_video(
        self,
        video_path: str,
        source_id: ObjectId,
        source_uri: str,
        video_extractor  # Pass extractor from outside
    ) -> List[Dict]:
        """Optimized local video processing workflow"""
        
        # Step 1: Extract transcript
        logger.info("  [1/3] Extracting audio and transcribing...")
        transcripts = await video_extractor.extract_from_sources([video_path])
        transcript_data = transcripts.get(video_path, [])
        
        if not transcript_data:
            raise ValueError("No transcript extracted from video file")
        
        logger.info(f"  ✓ Extracted {len(transcript_data)} transcript segments")
        
        # Step 2: Batch embedding
        logger.info("  [2/3] Generating embeddings in batches...")
        texts_to_embed = [seg.get('text', '').strip() for seg in transcript_data if seg.get('text', '').strip()]
        
        if not texts_to_embed:
            raise ValueError("No valid text found in transcript")
        
        # Embed all texts in batch
        vectors = self.embedder.text_embedder.embed_batch(texts_to_embed, input_type="document")
        
        logger.info(f"  ✓ Generated embeddings for {len(vectors)} segments")
        
        # Step 3: Create Knowledge Units
        logger.info("  [3/3] Finalizing Knowledge Units...")
        kus = []
        
        for idx, (segment, vector) in enumerate(zip(transcript_data, vectors)):
            text = segment.get('text', '').strip()
            if not text or not vector:
                continue
            
            ku = {
                "source_id": source_id,
                "source_type": "video",
                "source_uri": source_uri,
                "ku_id": f"video_{source_id}_segment_{idx}",
                "ku_type": "text_chunk",
                "created_at": datetime.utcnow(),
                
                # Raw content
                "raw_content": {
                    "text": text,
                    "asset_uri": None,
                },
                
                # Context
                "context": {
                    "page_number": None,
                    "bounding_box": None,
                    "start_time_seconds": segment.get('start'),
                    "end_time_seconds": segment.get('end'),
                    "direct_url": None  # No direct URL for local files
                },
                
                # Enriched content
                "enriched_content": None,
                
                # Embeddings
                "embeddings": {
                    "vector": vector,
                    "model": self.embedder.text_embedder.model_name
                }
            }
            kus.append(ku)
        
        logger.info(f"  ✓ Created {len(kus)} knowledge units")
        return kus


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
                    "analysis_model": "gemini-2.5-pro"
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