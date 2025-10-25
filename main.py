"""
main.py
Complete optimized multimodal RAG system with centralized configuration
"""
import os
import logging
from pathlib import Path
import time
import asyncio
from typing import List, Dict
from datetime import datetime, timezone
# Configuration
from src.config import get_config

# Core components
from src.storage.MongoDBHandler import MongoDBHandler
from src.Pipeline import OptimizedPipeline
from src.query_engine import EnhancedQueryEngine, VoyageReranker
from src.embedder import TextEmbedder, MultimodalEmbedder, ContentEmbedder
from src.enrichment.clients.gemini_client import GeminiClient

# Extractors
from src.extraction.pdf import PDFExtractor
from src.extraction.video import VideoExtractor
from src.extraction.website import WebsiteExtractor

# Setup logging
def setup_logging(config):
    """Setup logging configuration"""
    log_config = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    if config.paths.log_file:
        log_config['filename'] = config.paths.log_file
    
    logging.basicConfig(**log_config)

logger = logging.getLogger(__name__)


class MultimodalRAGSystem:
    """Complete multimodal RAG system with optimized components"""
    
    def __init__(
        self,
        pipeline: OptimizedPipeline,
        query_engine: EnhancedQueryEngine,
        pdf_extractor: PDFExtractor,
        video_extractor: VideoExtractor,
        website_extractor: WebsiteExtractor,
        config
    ):
        self.pipeline = pipeline
        self.query_engine = query_engine
        self.pdf_extractor = pdf_extractor
        self.video_extractor = video_extractor
        self.website_extractor = website_extractor
        self.config = config
        
        logger.info("✓ Multimodal RAG System initialized")
        logger.info("  - Supported sources: PDF, Images, Videos, YouTube, Websites")

    async def ingest_sources(self, sources: List[Dict[str, str]]):
        """Ingest from multiple source types"""
        logger.info(f"Starting batch ingestion of {len(sources)} sources...")
        start_time = time.time()
        
        results = []
        for source in sources:
            source_type = source.get('type')
            
            try:
                if source_type == 'pdf':
                    result = await self._ingest_pdf(source['path'])
                elif source_type == 'youtube':
                    result = await self._ingest_youtube(source['url'])
                elif source_type == 'video':
                    result = await self._ingest_video(source['path'])
                elif source_type == 'website':
                    result = await self._ingest_website(source['url'])
                elif source_type == 'image':
                    result = await self._ingest_image(source['path'])
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    result = {'status': 'skipped', 'reason': 'unknown type'}
                
                results.append({'source': source, 'result': result})
                
            except Exception as e:
                logger.error(f"Failed to ingest {source}: {e}", exc_info=True)
                results.append({
                    'source': source,
                    'result': {'status': 'failed', 'error': str(e)}
                })
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r['result'].get('status') != 'failed')
        
        logger.info(f"✓ Batch ingestion completed in {elapsed:.2f}s")
        logger.info(f"  - Successful: {successful}/{len(sources)}")
        
        return results

    async def _ingest_pdf(self, pdf_path: str):
        """Ingest a PDF document"""
        logger.info(f"Ingesting PDF: {pdf_path}")
        source_id = await self.pipeline.process_document_async(
            pdf_path, 
            source_type='pdf',
            source_uri=pdf_path
        )
        return {'status': 'success', 'source_id': str(source_id)}

    async def _ingest_youtube(self, youtube_url: str):
        """Ingest YouTube video - unified with pipeline"""
        logger.info(f"Ingesting YouTube: {youtube_url}")
        
        try:
            source_id = await self.pipeline.process_document_async(
                youtube_url,  # file_path can be URL for youtube
                source_type='youtube',
                source_uri=youtube_url,
                video_extractor=self.video_extractor  # Pass extractor to pipeline
            )
            
            # Get saved KU count
            source_doc = self.pipeline.db.get_source(source_id)
            total_kus = source_doc.get('total_kus', 0) if source_doc else 0
            
            return {
                'status': 'success',
                'source_id': str(source_id),
                'transcript_chunks': total_kus
            }
        
        except Exception as e:
            logger.error(f"YouTube ingestion failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    async def _ingest_video(self, video_path: str):
        """Ingest local video file - unified with pipeline"""
        logger.info(f"Ingesting video: {video_path}")
        
        try:
            source_id = await self.pipeline.process_document_async(
                video_path,
                source_type='video',
                source_uri=video_path,
                video_extractor=self.video_extractor
            )
            
            source_doc = self.pipeline.db.get_source(source_id)
            total_kus = source_doc.get('total_kus', 0) if source_doc else 0
            
            return {
                'status': 'success',
                'source_id': str(source_id),
                'transcript_chunks': total_kus
            }
        
        except Exception as e:
            logger.error(f"Video ingestion failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    async def _ingest_website(self, url: str):
        """Ingest website content - unified with pipeline"""
        logger.info(f"Ingesting website: {url}")
        
        try:
            source_id = await self.pipeline.process_document_async(
                url,  # file_path can be URL for website
                source_type='website',
                source_uri=url,
                website_extractor=self.website_extractor
            )
            
            source_doc = self.pipeline.db.get_source(source_id)
            total_kus = source_doc.get('total_kus', 0) if source_doc else 0
            
            return {
                'status': 'success',
                'source_id': str(source_id),
                'chunks_created': total_kus
            }
        
        except Exception as e:
            logger.error(f"Website ingestion failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    async def _ingest_image(self, image_path: str):
        """Ingest a standalone image"""
        logger.info(f"Ingesting image: {image_path}")
        return {'status': 'success'}

    async def query(
        self, 
        question: str, 
        initial_k: int = None,
        final_k: int = None
    ) -> Dict:
        """Query the RAG system"""
        initial_k = initial_k or self.config.query.default_initial_k
        final_k = final_k or self.config.query.default_final_k
        
        logger.info(f"Processing query: \"{question[:50]}...\"")
        start_time = time.time()
        
        result = await self.query_engine.query(
            question=question,
            initial_k=initial_k,
            final_k=final_k
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Query completed in {elapsed:.3f}s")
        
        return result

    def print_query_result(self, result: Dict):
        """Pretty print query results"""
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result.get("answer", "No answer found."))
        
        print("\n" + "="*80)
        print("SOURCES:")
        print("="*80)
        
        for source in result.get("sources", []):
            print(f"\n[{source['id']}] {source['ku_type'].upper()}")
            print(f"  Source: {source.get('source_uri', 'N/A')}")
            
            if 'page' in source:
                print(f"  Page: {source['page']}")
            if 'timestamp' in source:
                print(f"  Time: {source['timestamp']}")
            
            print(f"  Vector Score: {source.get('score', 0):.4f}")
            if source.get('rerank_score'):
                print(f"  Rerank Score: {source['rerank_score']:.4f}")
            
            print(f"  Preview: {source.get('preview', 'N/A')}")
        
        print("\n" + "="*80)
        print("RETRIEVAL STATS:")
        print("="*80)
        stats = result.get("retrieval_stats", {})
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")


async def initialize_system() -> MultimodalRAGSystem:
    """Initialize all system components with configuration"""
    # Load configuration
    config = get_config()
    config.print_summary()
    
    logger.info("Initializing system components...")
    
    # Database
    mongo_handler = MongoDBHandler(
        config.database.uri,
        config.database.database_name
    )
    if not mongo_handler.test_connection():
        raise ConnectionError("Failed to connect to MongoDB")

    gemini_client = GeminiClient(
        api_key=config.gemini.api_key,
        model_name=config.gemini.model_name,
        temperature=config.gemini.temperature
    )

    # Embedding models
    text_embedder = TextEmbedder(
        api_key=config.voyage.api_key,
        model_name=config.voyage.text_model
    )
    
    multimodal_embedder = MultimodalEmbedder(
        api_key=config.voyage.api_key,
        model_name=config.voyage.multimodal_model
    )
    
    content_embedder = ContentEmbedder(
        text_embedder=text_embedder,
        multimodal_embedder=multimodal_embedder
    )

    # Reranker
    reranker = VoyageReranker(
        api_key=config.voyage.api_key,
        model_name=config.voyage.rerank_model
    )

    # Extractors
    pdf_extractor = PDFExtractor(
        use_layoutparser=config.processing.use_layoutparser,
        extract_images=True,
        extract_tables=True
    )
    
    video_extractor = VideoExtractor(
        whisper_model=config.processing.whisper_model,
        max_workers=config.processing.max_workers
    )
    
    website_extractor = WebsiteExtractor(max_workers=10)

    # Pipeline and query engine
    pipeline = OptimizedPipeline(
        mongo_handler=mongo_handler,
        llm_client=gemini_client, 
        content_embedder=content_embedder,
        max_workers=config.processing.max_workers,
        batch_size=config.processing.batch_size
    )
    
    query_engine = EnhancedQueryEngine(
        mongo_handler=mongo_handler,
        text_embedder=text_embedder,
        multimodal_embedder=multimodal_embedder,
        reranker=reranker,
        synthesis_llm=gemini_client,
        use_query_cache=config.query.use_cache
    )

    # Initialize RAG system
    return MultimodalRAGSystem(
        pipeline=pipeline,
        query_engine=query_engine,
        pdf_extractor=pdf_extractor,
        video_extractor=video_extractor,
        website_extractor=website_extractor,
        config=config
    )


async def main():
    """Main execution flow"""
    
    # Load config and setup logging
    config = get_config()
    setup_logging(config)
    
    # ==================== CONFIGURATION ====================
    SOURCES_TO_INGEST = [
        {'type': 'pdf', 'path': 'documents/1706.03762v7.pdf'},
        # {'type': 'youtube', 'url': 'https://youtu.be/YCzL96nL7j0?si=TLIpYP6Yvv8sLQyo'},
        # {'type': 'website', 'url': 'https://viblo.asia/p/recurrent-neural-networkphan-1-tong-quan-va-ung-dung-jvElaB4m5kw'},
    ]
    
    QUESTION = """
    What is the attention mechanism in transformers and how does it improve sequence modeling compared to traditional RNNs?
    """
    # =======================================================

    try:
        # Initialize system
        rag_system = await initialize_system()

        # Step 1: Ingest documents
        logger.info("\n" + "="*80)
        logger.info("STEP 1: INGESTING DOCUMENTS")
        logger.info("="*80)
        
        # Create documents directory if needed
        Path(config.paths.documents_dir).mkdir(exist_ok=True, parents=True)
        
        if SOURCES_TO_INGEST:
            ingestion_results = await rag_system.ingest_sources(SOURCES_TO_INGEST)
            
            # Print summary
            print("\nIngestion Summary:")
            for result in ingestion_results:
                source = result['source']
                status = result['result'].get('status', 'unknown')
                print(f"  [{status.upper()}] {source['type']}: {source.get('path') or source.get('url')}")
                if status == 'failed':
                    print(f"    Error: {result['result'].get('error')}")

        # Step 2: Query the system
        if QUESTION:
            logger.info("\n" + "="*80)
            logger.info("STEP 2: QUERYING THE SYSTEM")
            logger.info("="*80)
            
            result = await rag_system.query(question=QUESTION)
            rag_system.print_query_result(result)

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        raise
    
    finally:
        if 'rag_system' in locals():
            rag_system.pipeline.db.close()
        logger.info("System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())