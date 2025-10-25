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
    """Optimized pipeline with better text chunking"""
    
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
        video_extractor = None,
        website_extractor = None
    ) -> ObjectId:
        """Process a document with all optimizations"""
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
                if source_type == "pdf":
                    kus = await self._process_pdf_optimized(file_path, source_id, source_uri)
                elif source_type == "youtube":
                    if not video_extractor:
                        raise ValueError("video_extractor required")
                    kus = await self._process_youtube(source_uri, source_id, source_uri, video_extractor)
                elif source_type == "video":
                    if not video_extractor:
                        raise ValueError("video_extractor required")
                    kus = await self._process_video(file_path, source_id, source_uri, video_extractor)
                elif source_type == "website":
                    if not website_extractor:
                        raise ValueError("website_extractor required")
                    kus = await self._process_website(source_uri, source_id, source_uri, website_extractor)
                else:
                    raise NotImplementedError(f"Source type '{source_type}' not supported")
                
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
        """
        FIXED: Improved PDF processing with better text chunking
        """
        
        # Step 1: Extract in parallel
        logger.info("  [1/4] Extracting content in parallel...")
        raw_objects = self.pdf_extractor.extract_parallel(pdf_path)
        logger.info(f"  ✓ Extracted {len(raw_objects)} raw objects")

        # FIXED: Separate full text from other objects
        text_objects = [obj for obj in raw_objects if obj['type'] == 'text_chunk']
        other_objects = [obj for obj in raw_objects if obj['type'] != 'text_chunk']
        
        # Combine all text from same page
        page_texts = {}
        for obj in text_objects:
            page = obj.get('page', 1)
            if page not in page_texts:
                page_texts[page] = []
            page_texts[page].append(obj['text'])
        
        # Create full text per page
        full_text_objects = []
        for page, texts in page_texts.items():
            combined_text = "\n\n".join(texts)
            if combined_text.strip():
                full_text_objects.append({
                    'type': 'text_chunk',
                    'text': combined_text,
                    'page': page,
                    'bbox': None  # Full page text
                })
        
        logger.info(f"  ✓ Combined text from {len(text_objects)} blocks into {len(full_text_objects)} full-page texts")

        # Step 2: Batch enrichment (only for tables and figures)
        logger.info("  [2/4] Enriching tables and figures...")
        items_to_enrich = []
        
        for obj in other_objects:
            item = {
                "type": obj['type'],
                "caption": obj.get('caption', ''),
                "id": obj.get('figure_id') or obj.get('table_id', '')
            }
            
            if obj['type'] == 'table':
                item['data'] = obj.get('data', {})
            elif obj['type'] == 'figure':
                try:
                    img_data = obj.get('image_data')
                    if isinstance(img_data, bytes):
                        item['data'] = Image.open(io.BytesIO(img_data))
                    else:
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
        
        # Map enrichment back
        enrichment_map = {item.get('id'): item.get('enrichment', {}) for item in enriched_results}
        for obj in other_objects:
            obj_id = obj.get('figure_id') or obj.get('table_id', '')
            obj['enrichment'] = enrichment_map.get(obj_id, {})
        
        logger.info(f"  ✓ Enriched {len(enriched_results)} tables/figures")

        # Step 3: Generate embeddings
        logger.info("  [3/4] Generating embeddings...")
        kus_ready = []

        # FIXED: Process full-page text with proper chunking
        for text_obj in full_text_objects:
            text = text_obj['text']
            page = text_obj['page']
            
            # Chunk and embed the full text
            chunks = self.embedder.embed_text_chunk(text)
            
            logger.info(f"    Page {page}: {len(chunks)} chunks created")
            
            for chunk_idx, chunk in enumerate(chunks):
                kus_ready.append({
                    'type': 'text_chunk',
                    'page': page,
                    'bbox': None,
                    'chunk_index': chunk_idx,
                    'text': chunk['text_chunk'],
                    'embedding_vector': chunk['vector'],
                    'source_text': chunk['text_chunk'],
                    'embedding_type': 'text',
                    'enrichment': {}
                })

        # Process tables
        for obj in [o for o in other_objects if o['type'] == 'table']:
            enrichment = obj.get('enrichment', {})
            summary = enrichment.get('summary', '')
            caption = obj.get('caption', '')
            
            try:
                embedding_data = self.embedder.embed_table(
                    table_data=obj.get('data', {}),
                    caption=caption,
                    summary=summary,
                    table_image=None
                )
                
                if embedding_data['vector']:  # Only add if embedding succeeded
                    kus_ready.append({
                        'type': 'table',
                        'page': obj.get('page'),
                        'bbox': obj.get('bbox'),
                        'data': obj.get('data'),
                        'caption': caption,
                        'enrichment': enrichment,
                        'embedding_vector': embedding_data['vector'],
                        'source_text': embedding_data['source_text'],
                        'embedding_type': embedding_data.get('embedding_type', 'text')
                    })
            except Exception as e:
                logger.error(f"Failed to embed table: {e}")

        # Process figures
        for obj in [o for o in other_objects if o['type'] == 'figure']:
            enrichment = obj.get('enrichment', {})
            summary = enrichment.get('summary', '')
            caption = obj.get('caption', '')
            
            try:
                img_data = obj.get('image_data')
                if isinstance(img_data, bytes):
                    image = Image.open(io.BytesIO(img_data))
                else:
                    continue
                
                embedding_data = self.embedder.embed_figure(
                    image=image,
                    caption=caption,
                    summary=summary
                )
                
                if embedding_data['vector']:  # Only add if embedding succeeded
                    kus_ready.append({
                        'type': 'figure',
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
        
        logger.info(f"  ✓ Generated embeddings for {len(kus_ready)} knowledge units")
        logger.info(f"    - Text chunks: {sum(1 for k in kus_ready if k['type'] == 'text_chunk')}")
        logger.info(f"    - Tables: {sum(1 for k in kus_ready if k['type'] == 'table')}")
        logger.info(f"    - Figures: {sum(1 for k in kus_ready if k['type'] == 'figure')}")

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
        video_extractor
    ) -> List[Dict]:
        """Process YouTube video"""
        logger.info("  [1/3] Extracting transcript...")
        transcripts = await video_extractor.extract_from_sources([youtube_url])
        transcript_data = transcripts.get(youtube_url, [])
        
        if not transcript_data:
            raise ValueError("No transcript extracted")
        
        logger.info(f"  ✓ Extracted {len(transcript_data)} segments")
        
        logger.info("  [2/3] Generating embeddings...")
        texts_to_embed = [seg.get('text', '').strip() for seg in transcript_data if seg.get('text', '').strip()]
        
        if not texts_to_embed:
            raise ValueError("No valid text found")
        
        vectors = self.embedder.text_embedder.embed_batch(texts_to_embed, input_type="document")
        logger.info(f"  ✓ Generated {len(vectors)} embeddings")
        
        logger.info("  [3/3] Creating Knowledge Units...")
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
                "raw_content": {"text": text, "asset_uri": None},
                "context": {
                    "page_number": None,
                    "bounding_box": None,
                    "start_time_seconds": segment.get('start'),
                    "end_time_seconds": segment.get('end'),
                    "direct_url": f"{youtube_url}&t={int(segment.get('start', 0))}s"
                },
                "enriched_content": None,
                "embeddings": {
                    "vector": vector,
                    "model": self.embedder.text_embedder.model_name
                }
            }
            kus.append(ku)
        
        return kus

    async def _process_website(
        self,
        url: str,
        source_id: ObjectId,
        source_uri: str,
        website_extractor
    ) -> List[Dict]:
        """Process website"""
        logger.info("  [1/3] Extracting content...")
        extracted = await website_extractor.extract_from_urls([url])
        content = extracted.get(url)
        
        if not content:
            raise ValueError("No content extracted")
        
        logger.info(f"  ✓ Extracted {len(content)} characters")
        
        logger.info("  [2/3] Chunking and embedding...")
        text_chunks = self.embedder.text_embedder.chunk_and_embed(content, chunk_size=1000, chunk_overlap=200)
        
        if not text_chunks:
            raise ValueError("No chunks generated")
        
        logger.info(f"  ✓ Generated {len(text_chunks)} chunks")
        
        logger.info("  [3/3] Creating Knowledge Units...")
        kus = []
        
        for idx, chunk in enumerate(text_chunks):
            ku = {
                "source_id": source_id,
                "source_type": "website",
                "source_uri": source_uri,
                "ku_id": f"website_{source_id}_text_{idx}",
                "ku_type": "text_chunk",
                "created_at": datetime.utcnow(),
                "raw_content": {"text": chunk['text_chunk'], "asset_uri": None},
                "context": {
                    "page_number": None,
                    "bounding_box": None,
                    "start_time_seconds": None,
                    "end_time_seconds": None,
                    "direct_url": url
                },
                "enriched_content": None,
                "embeddings": {
                    "vector": chunk['vector'],
                    "model": self.embedder.text_embedder.model_name
                }
            }
            kus.append(ku)
        
        return kus

    async def _process_video(
        self,
        video_path: str,
        source_id: ObjectId,
        source_uri: str,
        video_extractor
    ) -> List[Dict]:
        """Process local video - similar to YouTube"""
        logger.info("  [1/3] Extracting transcript...")
        transcripts = await video_extractor.extract_from_sources([video_path])
        transcript_data = transcripts.get(video_path, [])
        
        if not transcript_data:
            raise ValueError("No transcript extracted")
        
        logger.info(f"  ✓ Extracted {len(transcript_data)} segments")
        
        logger.info("  [2/3] Generating embeddings...")
        texts_to_embed = [seg.get('text', '').strip() for seg in transcript_data if seg.get('text', '').strip()]
        vectors = self.embedder.text_embedder.embed_batch(texts_to_embed, input_type="document")
        
        logger.info("  [3/3] Creating Knowledge Units...")
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
                "raw_content": {"text": text, "asset_uri": None},
                "context": {
                    "page_number": None,
                    "bounding_box": None,
                    "start_time_seconds": segment.get('start'),
                    "end_time_seconds": segment.get('end'),
                    "direct_url": None
                },
                "enriched_content": None,
                "embeddings": {
                    "vector": vector,
                    "model": self.embedder.text_embedder.model_name
                }
            }
            kus.append(ku)
        
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
            
            ku = {
                "source_id": source_id,
                "source_type": source_type,
                "source_uri": source_uri,
                "ku_id": f"{source_type}_{source_id}_{obj_type}_{idx}",
                "ku_type": obj_type,
                "created_at": datetime.utcnow(),
                "raw_content": {
                    "text": obj.get('text') or obj.get('source_text'),
                    "asset_uri": None,
                },
                "context": {
                    "page_number": obj.get('page'),
                    "bounding_box": obj.get('bbox'),
                    "start_time_seconds": None,
                    "end_time_seconds": None,
                    "direct_url": None
                },
                "enriched_content": {
                    "summary": enrichment.get('summary'),
                    "keywords": enrichment.get('keywords', []),
                    "analysis_model": "gemini-2.5-pro"
                } if enrichment else None,
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
        """Synchronous wrapper"""
        return asyncio.run(
            self.process_document_async(file_path, source_type, source_uri)
        )