import asyncio
import sys
from pathlib import Path

# # Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from main import initialize_system

async def test_full_workflow():
    """
    Integration test: Full R&D workflow
    """
    print("="*80)
    print("INTEGRATION TEST: R&D WORKFLOW")
    print("="*80)
    
    # Test papers (create these fixtures)
    TEST_PAPERS = [
        "tests/fixtures/attention_is_all_you_need.pdf",
        "tests/fixtures/bert.pdf"
    ]
    
    # Initialize system
    print("\n[1/4] Initializing system...")
    rag_system = await initialize_system()
    print("✓ System initialized")
    
    # Ingest papers
    print("\n[2/4] Ingesting research papers...")
    sources = [{'type': 'pdf', 'path': path} for path in TEST_PAPERS]
    
    ingestion_results = await rag_system.ingest_sources(sources)
    
    successful = sum(1 for r in ingestion_results 
                    if r['result'].get('status') == 'success')
    print(f"✓ Ingested {successful}/{len(sources)} papers")
    
    # Verify ingestion
    for result in ingestion_results:
        source_info = result['source']
        status = result['result'].get('status')
        
        if status == 'success':
            source_id = result['result']['source_id']
            source_doc = rag_system.pipeline.db.get_source(source_id)
            
            print(f"\n  Paper: {Path(source_info['path']).name}")
            print(f"    - Status: {source_doc['status']}")
            print(f"    - Total KUs: {source_doc['total_kus']}")
            
            # Get KU breakdown
            kus = rag_system.pipeline.db.get_kus_by_source(source_id)
            ku_types = {}
            for ku in kus:
                ku_type = ku['ku_type']
                ku_types[ku_type] = ku_types.get(ku_type, 0) + 1
            
            print(f"    - Text chunks: {ku_types.get('text_chunk', 0)}")
            print(f"    - Tables: {ku_types.get('table', 0)}")
            print(f"    - Figures: {ku_types.get('figure', 0)}")
    
    # Test queries
    print("\n[3/4] Testing R&D queries...")
    
    test_queries = [
        "What is the self-attention mechanism in Transformers?",
        "Compare the model architectures discussed in the papers",
        "What evaluation metrics are reported?",
        "Explain the multi-head attention computation"
    ]
    
    for idx, question in enumerate(test_queries, 1):
        print(f"\n  Query {idx}: {question}")
        
        result = await rag_system.query(
            question=question,
            initial_k=20,
            final_k=5
        )
        
        answer = result.get('answer', '')
        sources = result.get('sources', [])
        
        print(f"  Answer length: {len(answer)} chars")
        print(f"  Sources used: {len(sources)}")
        
        # Show source breakdown
        source_types = {}
        for source in sources:
            ku_type = source['ku_type']
            source_types[ku_type] = source_types.get(ku_type, 0) + 1
        
        print(f"  Source types: {source_types}")
        
        # Verify answer quality
        assert len(answer) > 100, "Answer too short"
        assert len(sources) > 0, "No sources retrieved"
    
    # Test statistics
    print("\n[4/4] Checking database statistics...")
    
    stats = rag_system.pipeline.db.get_statistics()
    
    print(f"\nDatabase Statistics:")
    print(f"  - Total sources: {stats['total_sources']}")
    print(f"  - Total KUs: {stats['total_kus']}")
    print(f"  - KUs by type: {stats['kus_by_type']}")
    print(f"  - Sources by status: {stats['sources_by_status']}")
    
    # Cleanup
    rag_system.pipeline.db.close()
    
    print("\n" + "="*80)
    print("✓ INTEGRATION TEST PASSED")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_full_workflow())