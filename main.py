"""
main.py
Complete optimized multimodal RAG system with centralized configuration
"""
import json
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
    """R&D-optimized workflow"""
    
    config = get_config()
    setup_logging(config)
    
    # ==================== CONFIGURATION ====================
    # Test papers for R&D automation
    SOURCES_TO_INGEST = [
        {'type': 'pdf', 'path': 'documents/attention_is_all_you_need.pdf'},
        {'type': 'pdf', 'path': 'documents/bert.pdf'},
        # {'type': 'youtube', 'url': 'https://youtu.be/...'},  # Conference talk
    ]
    
    # R&D-focused queries
    RESEARCH_QUERIES = [
        {
            "question": "What is the self-attention mechanism and how is it computed?",
            "description": "Methodology inquiry"
        },
        {
            "question": "What are the key performance metrics reported in these papers?",
            "description": "Results extraction"
        },
        {
            "question": "Compare the model architectures across papers",
            "description": "Comparative analysis"
        },
        {
            "question": "What datasets were used for evaluation?",
            "description": "Experimental setup"
        }
    ]
    
    # Enable detailed logging for R&D
    SAVE_EXTRACTION_RESULTS = True
    PRINT_DETAILED_SOURCES = True
    # =======================================================

    try:
        # Initialize system with VLM
        logger.info("\n" + "="*80)
        logger.info("R&D MULTIMODAL RAG SYSTEM - VLM POWERED")
        logger.info("="*80)
        
        rag_system = await initialize_system()
        
        # Verify VLM is enabled
        if not rag_system.pipeline.use_vlm_extraction:
            logger.warning(" VLM extraction is DISABLED - results may be suboptimal")
            logger.warning("   Set USE_VLM_EXTRACTION=true in .env")

        # Step 1: Ingest research papers
        if SOURCES_TO_INGEST:
            logger.info("\n" + "="*80)
            logger.info("STEP 1: INGESTING RESEARCH PAPERS")
            logger.info("="*80)
            
            Path(config.paths.documents_dir).mkdir(exist_ok=True, parents=True)
            
            ingestion_results = await rag_system.ingest_sources(SOURCES_TO_INGEST)
            
            # Print detailed summary
            print("\n" + "="*80)
            print("INGESTION SUMMARY")
            print("="*80)
            
            for result in ingestion_results:
                source = result['source']
                status = result['result'].get('status', 'unknown')
                
                if status == 'success':
                    source_id = result['result'].get('source_id')
                    if source_id:
                        # Get detailed stats
                        source_doc = rag_system.pipeline.db.get_source(source_id)
                        kus = rag_system.pipeline.db.get_kus_by_source(source_id)
                        
                        ku_breakdown = {}
                        for ku in kus:
                            ku_type = ku['ku_type']
                            ku_breakdown[ku_type] = ku_breakdown.get(ku_type, 0) + 1
                        
                        print(f"\n✓ {source.get('path') or source.get('url')}")
                        print(f"  Status: {source_doc['status']}")
                        print(f"  Total KUs: {source_doc['total_kus']}")
                        print(f"  Breakdown:")
                        for ku_type, count in ku_breakdown.items():
                            print(f"    - {ku_type}: {count}")
                        
                        # Show extraction methods used
                        extraction_methods = set()
                        for ku in kus:
                            if ku['ku_type'] == 'table':
                                method = ku.get('raw_content', {}).get('table_data', {}).get('metadata', {}).get('extraction_method')
                                if method:
                                    extraction_methods.add(method)
                        
                        if extraction_methods:
                            print(f"  Extraction methods: {', '.join(extraction_methods)}")
                
                elif status == 'failed':
                    print(f"\n✗ {source.get('path') or source.get('url')}")
                    print(f"  Error: {result['result'].get('error')}")
            
            print("="*80 + "\n")

        # Step 2: Execute R&D queries
        if RESEARCH_QUERIES:
            logger.info("\n" + "="*80)
            logger.info("STEP 2: EXECUTING R&D QUERIES")
            logger.info("="*80)
            
            for idx, query_info in enumerate(RESEARCH_QUERIES, 1):
                question = query_info['question']
                description = query_info.get('description', '')
                
                print(f"\n{'='*80}")
                print(f"QUERY {idx}/{len(RESEARCH_QUERIES)}")
                print(f"{'='*80}")
                print(f"Question: {question}")
                if description:
                    print(f"Type: {description}")
                print(f"{'='*80}\n")
                
                # Execute query
                result = await rag_system.query(
                    question=question,
                    initial_k=20,
                    final_k=5
                )
                
                # Print answer
                print("ANSWER:")
                print("-"*80)
                print(result.get("answer", "No answer found."))
                print("-"*80)
                
                # Print sources
                sources = result.get("sources", [])
                print(f"\nSOURCES ({len(sources)}):")
                print("-"*80)
                
                for source in sources:
                    print(f"\n[{source['id']}] {source['ku_type'].upper()}")
                    print(f"  From: {Path(source.get('source_uri', 'N/A')).name}")
                    
                    if 'page' in source:
                        print(f"  Page: {source['page']}")
                    if 'timestamp' in source:
                        print(f"  Time: {source['timestamp']}")
                    
                    print(f"  Vector Score: {source.get('score', 0):.4f}")
                    if source.get('rerank_score'):
                        print(f"  Rerank Score: {source['rerank_score']:.4f}")
                    
                    if PRINT_DETAILED_SOURCES:
                        print(f"  Preview: {source.get('preview', 'N/A')}")
                    # Print retrieval stats
                    stats = result.get("retrieval_stats", {})
                    print(f"\nRETRIEVAL STATS:")
                    print("-"*80)
                    print(f"  Initial candidates: {stats.get('initial_candidates', 'N/A')}")
                    print(f"  Reranked results: {stats.get('reranked_results', 'N/A')}")
                    print(f"  Vector model: {stats.get('vector_search_model', 'N/A')}")
                    print(f"  Rerank model: {stats.get('rerank_model', 'N/A')}")
                    print(f"  Synthesis model: {stats.get('synthesis_model', 'N/A')}")
                    print("="*80 + "\n")
                    
                    # Optional: Save results
                    if SAVE_EXTRACTION_RESULTS:
                        output_dir = Path("output/queries")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_file = output_dir / f"query_{idx}_result.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'question': question,
                                'description': description,
                                'answer': result.get('answer'),
                                'sources': result.get('sources'),
                                'stats': stats
                            }, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"  → Saved results to {output_file}")
            
            # Step 3: Generate research insights report
            logger.info("\n" + "="*80)
            logger.info("STEP 3: GENERATING RESEARCH INSIGHTS")
            logger.info("="*80)
            
            insights = await generate_research_insights(rag_system)
            print_research_insights(insights)
            
            # Step 4: Database statistics
            logger.info("\n" + "="*80)
            logger.info("STEP 4: DATABASE STATISTICS")
            logger.info("="*80)
            
            stats = rag_system.pipeline.db.get_statistics()
            
            print(f"\nTotal Sources: {stats['total_sources']}")
            print(f"Total Knowledge Units: {stats['total_kus']}")
            
            print(f"\nSources by Status:")
            for status, count in stats['sources_by_status'].items():
                print(f"  {status}: {count}")
            
            print(f"\nKUs by Type:")
            for ku_type, count in stats['kus_by_type'].items():
                print(f"  {ku_type}: {count}")
            
            print(f"\nKUs by Source Type:")
            for source_type, count in stats['kus_by_source_type'].items():
                print(f"  {source_type}: {count}")
            
            print("\n" + "="*80)

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        raise
        
    finally:
        if 'rag_system' in locals():
            rag_system.pipeline.db.close()
        logger.info("System shutdown complete")


    async def generate_research_insights(rag_system) -> Dict[str, Any]:
        """
        Generate aggregated research insights from ingested papers
        """
        insights = {
            'total_papers': 0,
            'methodologies': set(),
            'datasets': set(),
            'metrics': {},
            'key_findings': []
        }
        
        # Get all sources
        sources = rag_system.pipeline.db.list_sources(status='completed')
        insights['total_papers'] = len(sources)
        
        # Aggregate metadata from all KUs
        for source in sources:
            source_id = source['_id']
            kus = rag_system.pipeline.db.get_kus_by_source(source_id)
            
            for ku in kus:
                enriched = ku.get('enriched_content', {})
                
                # Extract research metadata if available
                research_meta = enriched.get('research_metadata', {})
                
                if research_meta:
                    # Methodologies
                    methods = research_meta.get('methodology_keywords', [])
                    insights['methodologies'].update(methods)
                    
                    # Datasets
                    datasets = research_meta.get('datasets_mentioned', [])
                    insights['datasets'].update(datasets)
                    
                    # Metrics
                    metrics = research_meta.get('metrics_mentioned', {})
                    for metric, value in metrics.items():
                        if metric not in insights['metrics']:
                            insights['metrics'][metric] = []
                        insights['metrics'][metric].append(value)
        
        # Convert sets to lists for JSON serialization
        insights['methodologies'] = list(insights['methodologies'])
        insights['datasets'] = list(insights['datasets'])
        
        return insights


    def print_research_insights(insights: Dict[str, Any]):
        """Pretty print research insights"""
        print("\n" + "="*80)
        print("AGGREGATED RESEARCH INSIGHTS")
        print("="*80)
        
        print(f"\nPapers Analyzed: {insights['total_papers']}")
        
        if insights['methodologies']:
            print(f"\nMethodologies Mentioned:")
            for method in sorted(insights['methodologies'])[:10]:  # Top 10
                print(f"  • {method}")
        
        if insights['datasets']:
            print(f"\nDatasets Mentioned:")
            for dataset in sorted(insights['datasets'])[:10]:  # Top 10
                print(f"  • {dataset}")
        
        if insights['metrics']:
            print(f"\nPerformance Metrics:")
            for metric, values in sorted(insights['metrics'].items())[:10]:
                if values:
                    avg_value = sum(values) / len(values)
                    print(f"  • {metric}: {avg_value:.2f} (avg of {len(values)} mentions)")
        
        print("="*80)

if __name__ == "__main__":
    asyncio.run(main())