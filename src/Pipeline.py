import asyncio
import io
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from bson import ObjectId
from PIL import Image
import pandas as pd
from PIL import Image
from .enrichment.schema import EnrichmentOutput

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
        batch_size: int = 16,
        use_vlm_extraction: bool = True 
    ):
        self.db = mongo_handler
        self.batch_size = batch_size
        
        # Optimized components
        self.pdf_extractor = PDFExtractor(max_workers=max_workers)
        self.enricher = ContentEnricher(client=llm_client, max_concurrent=10)
        self.embedder = content_embedder
        
        # Semaphore for controlling concurrent document processing
        self.executor = asyncio.Semaphore(max_workers)
        self.use_vlm_extraction = use_vlm_extraction
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
    
    async def _process_pdf_optimized(self, pdf_path, source_id, source_uri):
        from .extraction.utils import PDFUtils
        
        logger.info("[1/5] Extracting text & page images...")
        raw_objects = self.pdf_extractor.extract_parallel(pdf_path)
        
        text_objs = [o for o in raw_objects if o['type']=='text_chunk']
        page_images = [o for o in raw_objects if o['type']=='page_image_for_vlm']
        
        logger.info(f"[2/5] VLM detecting tables & figures on {len(page_images)} pages...")
        tables_detected = []
        figures_detected = []

        for page_img_obj in page_images:
            page_num = page_img_obj['page'] - 1
            page_image = page_img_obj['image']
            
            # Wrap VLM call in try-except để không block text extraction
            try:
                layout = await PDFUtils.detect_layout_with_vlm(
                    page_image, self.enricher.client, page_num
                )
                
                if not layout:
                    logger.warning(f"Page {page_num+1}: VLM layout detection failed, skipping VLM extraction")
                    continue
                
                for region in layout.regions:
                    try:
                        bbox = region.bbox
                        
                        # Validate bbox
                        if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
                            logger.warning(f"Page {page_num+1}: Invalid bbox, skipping region")
                            continue
                        
                        cropped = page_image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                        
                        if region.type == "table":
                            # Try VLM-based table extraction first (pass pdf_path for potential fallback)
                            table_data = await PDFUtils.extract_table_with_vlm(
                                cropped, self.enricher.client, page_num, region.description, pdf_path=pdf_path, bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                            )

                            # If VLM failed or returned empty, attempt pdfplumber fallback using page bbox
                            if not table_data or not table_data.get('data'):
                                logger.info(f"Page {page_num+1}: VLM table extraction empty, trying pdfplumber fallback")
                                try:
                                    pdf_table = PDFUtils.extract_table_with_pdfplumber(pdf_path, page_num, bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                                    if pdf_table and pdf_table.get('data'):
                                        table_data = pdf_table
                                except Exception as e:
                                    logger.warning(f"pdfplumber fallback failed: {e}")

                            if table_data and table_data.get('data') and len(table_data['data']) > 0:
                                tables_detected.append(table_data)
                            else:
                                logger.warning(f"Page {page_num+1}: Table extraction returned empty data")
                        
                        elif region.type == "figure":
                            figure_data = await PDFUtils.analyze_figure_with_vlm(
                                cropped, self.enricher.client, page_num, region.description
                            )
                            if figure_data:
                                # Attach image bytes for later embedding/storage if needed
                                figure_data['image_data'] = cropped
                                figure_data['analysis'] = {
                                    'caption': region.description,
                                    'raw_findings': figure_data.get('key_findings', []),
                                    'numerical_data': figure_data.get('numerical_data', [])
                                }
                                figures_detected.append(figure_data)
                    
                    except Exception as e:
                        logger.error(f"Page {page_num+1}: Region processing failed - {e}")
                        continue  # Continue to next region
            
            except Exception as e:
                logger.error(f"Page {page_num+1}: VLM processing failed - {e}")
                continue  # Continue to next page

        logger.info(f"  Detected {len(tables_detected)} tables, {len(figures_detected)} figures")
        
        # ✅ [3/5] Text processing - ALWAYS runs regardless of VLM success
        logger.info("[3/5] Chunking & enriching text...")
        text_kus = []

        for txt_obj in text_objs:
            try:
                # Chunk text
                chunks = self.embedder.embed_text_chunk(txt_obj['text'])
                logger.info(f"  Page {txt_obj['page']}: Created {len(chunks)} text chunks")
                
                for idx, chunk in enumerate(chunks):
                    # ✅ Enrich với error handling
                    try:
                        enrichment = await self._enrich_text_chunk(chunk['text_chunk'])
                    except Exception as e:
                        logger.error(f"Enrichment failed for chunk {idx}: {e}")
                        enrichment = {'summary': chunk['text_chunk'][:200], 'keywords': []}
                    
                    text_kus.append({
                        'type': 'text_chunk',
                        'page': txt_obj['page'],
                        'chunk_index': idx,
                        'text': chunk['text_chunk'],
                        'embedding_vector': chunk['vector'],
                        'source_text': chunk['text_chunk'],
                        'embedding_type': 'text',
                        'enrichment': enrichment
                    })
            
            except Exception as e:
                logger.error(f"Failed to process text from page {txt_obj['page']}: {e}")
                continue  # ✅ Skip failed page but continue

        logger.info(f"  ✓ Created {len(text_kus)} text chunks with enrichment")
        
        # [4/5] Process tables/figures (optional)
        logger.info("[4/5] Enriching & embedding tables/figures...")
        table_kus = []
        for table in tables_detected:
            try:
                enrichment = await self._enrich_table(table)
                emb_data = self.embedder.embed_table(
                    table['data'], table['caption'], enrichment.get('summary','')
                )
                if emb_data['vector']:
                    table_kus.append({
                        'type':'table',
                        'page':table['page'],
                        'data':table['data'],
                        'caption':table.get('caption', ''),
                        'raw_content': {
                            'table': table.get('data'),
                            'headers': table.get('headers'),
                            'asset_uri': None
                        },
                        'enrichment': enrichment,
                        'embeddings': {
                            'vector': emb_data['vector'],
                            'model': emb_data.get('model') if isinstance(emb_data, dict) else None,
                            'type': emb_data.get('embedding_type') if isinstance(emb_data, dict) else None
                        }
                    })
            except Exception as e:
                logger.error(f"Failed to process table: {e}")
                continue
        
        figure_kus = []
        for figure in figures_detected:
            try:
                enrichment = await self._enrich_figure(figure)
                emb_data = self.embedder.embed_figure(
                    figure['image_data'], figure['caption'], enrichment.get('summary','')
                )
                if emb_data['vector']:
                    # Save preview/analysis in raw_content so it is retrievable during multimodal queries
                    figure_kus.append({
                        'type': 'figure',
                        'page': figure['page'],
                        'caption': figure.get('caption', ''),
                        'raw_content': {
                            'caption': figure.get('caption', ''),
                            'analysis': figure.get('analysis', {}),
                            'asset_uri': None
                        },
                        'enrichment': enrichment,
                        'embeddings': {
                            'vector': emb_data['vector'],
                            'model': emb_data.get('model') if isinstance(emb_data, dict) else None,
                            'type': emb_data.get('embedding_type') if isinstance(emb_data, dict) else None
                        }
                    })
            except Exception as e:
                logger.error(f"Failed to process figure: {e}")
                continue
        
        logger.info(f"[5/5] Creating {len(text_kus)+len(table_kus)+len(figure_kus)} KUs...")
        all_kus = text_kus + table_kus + figure_kus
        
        if not all_kus:
            logger.error(" No KUs created - all extraction failed!")
            return []
        
        return self._create_knowledge_units(all_kus, source_id, "pdf", source_uri)

    async def _enrich_text_chunk(self, text):
            """Enrich text chunk with error handling"""
            try:
                result = await self.enricher.enrich_batch([{
                    "type": "text",
                    "data": text
                }])
                
                if result and len(result) > 0:
                    enrichment = result[0].get('enrichment')
                    if enrichment:
                        return enrichment
                
                # Fallback if enrichment failed
                logger.warning("Text enrichment failed, using fallback")
                return {
                    'summary': text[:200],
                    'keywords': [],
                    'research_metadata': None
                }
                
            except Exception as e:
                logger.error(f"Text enrichment error: {e}")
                return {
                    'summary': text[:200],
                    'keywords': [],
                    'research_metadata': None
                }

    async def _enrich_table(self, table_data):
        """Enrich table summary with error handling"""
        try:
            # Handle empty table
            if not table_data.get('data'):
                return {
                    'summary': 'Empty table',
                    'keywords': [],
                    'research_metadata': None
                }
            
            # Convert to string representation
            try:
                df = pd.DataFrame(table_data['data'])
                df_str = df.to_string(max_rows=20, index=False)
            except Exception:
                df_str = str(table_data['data'])[:500]
            
            # Enrich with LLM
            result = await self.enricher.enrich_batch([{
                "type": "table",
                "data": table_data['data'],
                "caption": table_data.get('caption', '')
            }])
            
            if result and len(result) > 0:
                enrichment = result[0].get('enrichment')
                if enrichment:
                    return enrichment
            
            # Fallback
            return {
                'summary': f"Table with {len(table_data.get('data', []))} rows",
                'keywords': [],
                'research_metadata': None
            }
            
        except Exception as e:
            logger.error(f"Table enrichment failed: {e}")
            return {
                'summary': 'Table content',
                'keywords': [],
                'research_metadata': None
            }

    async def _enrich_figure(self, figure_data):
        """Enrich figure with error handling"""
        try:
            # Use VLM analysis if available
            findings = "\n".join(figure_data.get('key_findings', []))
            
            if findings:
                return {
                    'summary': findings[:300],
                    'keywords': [
                        str(nd.get('label', '')) 
                        for nd in figure_data.get('numerical_data', [])[:5]
                    ],
                    'research_metadata': None
                }
            
            # Fallback to caption
            caption = figure_data.get('caption', 'Figure')
            return {
                'summary': caption[:300],
                'keywords': [],
                'research_metadata': None
            }
            
        except Exception as e:
            logger.error(f"Figure enrichment failed: {e}")
            return {
                'summary': 'Figure content',
                'keywords': [],
                'research_metadata': None
            }
    
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
                    "text": (obj.get('text') or (obj.get('raw_content') or {}).get('text') or obj.get('source_text')),
                    "asset_uri": (obj.get('raw_content') or {}).get('asset_uri') if obj.get('raw_content') else None,
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
                    "analysis_model": "gemini-2.5-flash"
                } if enrichment else None,
                "embeddings": {
                    "vector": (obj.get('embeddings') or {}).get('vector') or obj.get('embedding_vector') or obj.get('vector') or [],
                    "model": (obj.get('embeddings') or {}).get('model') or (
                        self.embedder.text_embedder.model_name if obj.get('embedding_type') == 'text' else self.embedder.multimodal_embedder.model_name
                    )
                }
            }
            
            # Add type-specific metadata
            if obj_type == 'table':
                # support both legacy and new structures
                ku['raw_content']['table_data'] = obj.get('data') or (obj.get('raw_content') or {}).get('table')
                ku['context']['caption'] = obj.get('caption') or (obj.get('raw_content') or {}).get('caption')
                # copy extraction metadata if available
                if obj.get('metadata'):
                    ku['raw_content']['metadata'] = obj.get('metadata')
            elif obj_type == 'figure':
                ku['context']['figure_id'] = obj.get('figure_id') or (obj.get('raw_content') or {}).get('figure_id')
                ku['context']['caption'] = obj.get('caption') or (obj.get('raw_content') or {}).get('caption')
                ku['raw_content']['analysis'] = (obj.get('raw_content') or {}).get('analysis') or obj.get('analysis')
                ku['context']['dimensions'] = {
                    'width': obj.get('width') or (obj.get('raw_content') or {}).get('width'),
                    'height': obj.get('height') or (obj.get('raw_content') or {}).get('height')
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