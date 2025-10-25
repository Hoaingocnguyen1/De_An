import asyncio
import pytest
from pathlib import Path
from PIL import Image
import json

from src.config import get_config
from src.Pipeline import OptimizedPipeline
from src.storage.MongoDBHandler import MongoDBHandler
from src.embedder import TextEmbedder, MultimodalEmbedder, ContentEmbedder
from src.enrichment.clients.gemini_client import GeminiClient
from src.extraction.utils import PDFUtils
from src.enrichment.schema import (
    LayoutDetectionOutput,
    TableExtractionOutput, 
    FigureAnalysisOutput
)

@pytest.fixture
async def setup_pipeline():
    """Initialize pipeline for testing"""
    config = get_config()
    
    mongo_handler = MongoDBHandler(
        config.database.uri,
        config.database.database_name + "_test"  # Use test DB
    )
    
    gemini_client = GeminiClient(
        api_key=config.gemini.api_key,
        model_name=config.gemini.vision_model
    )
    
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
    
    pipeline = OptimizedPipeline(
        mongo_handler=mongo_handler,
        llm_client=gemini_client,
        content_embedder=content_embedder,
        use_vlm_extraction=True
    )
    
    yield pipeline
    
    # Cleanup
    mongo_handler.db.knowledge_units.delete_many({})
    mongo_handler.db.sources.delete_many({})
    mongo_handler.close()


class TestVLMLayoutDetection:
    """Test VLM-based layout detection"""
    
    @pytest.mark.asyncio
    async def test_detect_single_column_layout(self):
        """Test detection of single-column paper layout"""
        config = get_config()
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.vision_model
        )
        
        # Create synthetic test image or use fixture
        test_image = Image.open("tests/fixtures/single_column_page.png")
        
        import fitz
        # Mock page for testing
        doc = fitz.open("tests/fixtures/test_paper.pdf")
        page = doc.load_page(0)
        
        result = await PDFUtils.detect_layout_with_vlm(page, client, 0)
        
        assert result is not None, "Layout detection returned None"
        assert isinstance(result, LayoutDetectionOutput)
        assert len(result.regions) > 0, "No regions detected"
        assert result.page_type in ["single_column", "double_column", "mixed"]
        
        # Validate bounding boxes
        for region in result.regions:
            assert region.bbox.validate_box(), f"Invalid bbox: {region.bbox}"
            assert 0 <= region.confidence <= 1.0
        
        doc.close()
        print(f"✓ Detected {len(result.regions)} regions in single-column layout")
    
    @pytest.mark.asyncio
    async def test_detect_double_column_layout(self):
        """Test detection of double-column paper layout"""
        config = get_config()
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.vision_model
        )
        
        import fitz
        doc = fitz.open("tests/fixtures/double_column_paper.pdf")
        page = doc.load_page(0)
        
        result = await PDFUtils.detect_layout_with_vlm(page, client, 0)
        
        assert result is not None
        assert result.page_type == "double_column" or len(result.regions) >= 2
        
        # Should detect multiple text blocks for columns
        text_blocks = [r for r in result.regions if r.type == "text_block"]
        assert len(text_blocks) >= 2, "Should detect at least 2 text blocks in double-column"
        
        doc.close()
        print(f"✓ Detected double-column layout with {len(text_blocks)} text blocks")


class TestVLMTableExtraction:
    """Test VLM-based table extraction"""
    
    @pytest.mark.asyncio
    async def test_extract_simple_table(self):
        """Test extraction of simple table"""
        config = get_config()
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.vision_model
        )
        
        # Use table image fixture
        table_image = Image.open("tests/fixtures/simple_table.png")
        
        prompt = """Extract this table structure.
        
Provide all headers and data rows accurately."""
        
        result = await client.create_structured_completion(
            prompt=prompt,
            response_model=TableExtractionOutput,
            image=table_image
        )
        
        assert result is not None, "Table extraction failed"
        assert result.num_rows > 0, "No rows extracted"
        assert result.num_cols > 0, "No columns extracted"
        assert len(result.headers) == result.num_cols, "Header count mismatch"
        assert len(result.rows) == result.num_rows, "Row count mismatch"
        
        # Validate each row has correct number of columns
        for row in result.rows:
            assert len(row) == result.num_cols, f"Row has wrong column count: {row}"
        
        # Test DataFrame conversion
        df = result.to_dataframe()
        assert df.shape == (result.num_rows, result.num_cols)
        
        print(f"✓ Extracted table: {result.num_rows}×{result.num_cols}")
        print(f"  Headers: {result.headers}")
        print(f"  Confidence: {result.extraction_confidence:.2f}")
    
    @pytest.mark.asyncio
    async def test_extract_complex_table_with_merged_cells(self):
        """Test extraction of table with merged headers"""
        config = get_config()
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.vision_model
        )
        
        table_image = Image.open("tests/fixtures/complex_table.png")
        
        prompt = """Extract this complex table.

Handle multi-level headers and merged cells carefully.
Preserve hierarchical structure in header names."""
        
        result = await client.create_structured_completion(
            prompt=prompt,
            response_model=TableExtractionOutput,
            image=table_image
        )
        
        assert result is not None
        assert result.has_merged_cells == True, "Should detect merged cells"
        assert result.extraction_confidence >= 0.7, "Low confidence on complex table"
        
        print(f"✓ Extracted complex table with merged cells")
        print(f"  Confidence: {result.extraction_confidence:.2f}")


class TestVLMFigureAnalysis:
    """Test VLM-based figure analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_bar_chart(self):
        """Test analysis of bar chart"""
        config = get_config()
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.vision_model
        )
        
        chart_image = Image.open("tests/fixtures/bar_chart.png")
        
        prompt = """Analyze this bar chart.

Extract:
- Chart type
- Key findings
- Numerical values
- Labels and legend"""
        
        result = await client.create_structured_completion(
            prompt=prompt,
            response_model=FigureAnalysisOutput,
            image=chart_image
        )
        
        assert result is not None
        assert result.figure_type == "bar_chart"
        assert len(result.key_findings) > 0, "No findings extracted"
        assert len(result.numerical_data) > 0, "No numerical data extracted"
        assert result.has_legend in [True, False], "Legend detection missing"
        
        print(f"✓ Analyzed bar chart:")
        print(f"  Findings: {result.key_findings}")
        print(f"  Data points: {len(result.numerical_data)}")
    
    @pytest.mark.asyncio
    async def test_analyze_architecture_diagram(self):
        """Test analysis of architecture diagram"""
        config = get_config()
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.vision_model
        )
        
        diagram_image = Image.open("tests/fixtures/architecture_diagram.png")
        
        prompt = """Analyze this architecture diagram.

Focus on:
- Components and their connections
- Data flow
- Key architectural patterns
- Research relevance"""
        
        result = await client.create_structured_completion(
            prompt=prompt,
            response_model=FigureAnalysisOutput,
            image=diagram_image
        )
        
        assert result is not None
        assert result.figure_type == "architecture_diagram"
        assert len(result.labels_detected) > 0, "No labels detected"
        assert len(result.relevance_to_research) > 0, "No research relevance"
        
        print(f"✓ Analyzed architecture diagram")
        print(f"  Labels: {result.labels_detected}")


class TestEndToEndPipeline:
    """Test complete PDF processing pipeline"""
    
    @pytest.mark.asyncio
    async def test_process_simple_paper(self, setup_pipeline):
        """Test processing a simple research paper"""
        pipeline = await setup_pipeline
        
        test_pdf = "tests/fixtures/simple_paper.pdf"
        
        # Process PDF
        source_id = await pipeline.process_document_async(
            test_pdf,
            source_type="pdf",
            source_uri=test_pdf
        )
        
        assert source_id is not None, "Processing failed"
        
        # Verify source created
        source = pipeline.db.get_source(source_id)
        assert source is not None
        assert source['status'] == 'completed'
        assert source['total_kus'] > 0
        
        # Verify KUs created
        kus = pipeline.db.get_kus_by_source(source_id)
        assert len(kus) > 0
        
        # Check KU types
        ku_types = {ku['ku_type'] for ku in kus}
        assert 'text_chunk' in ku_types, "No text chunks created"
        
        # Check embeddings
        for ku in kus:
            assert 'embeddings' in ku
            assert 'vector' in ku['embeddings']
            assert len(ku['embeddings']['vector']) > 0
        
        print(f"✓ Processed paper: {source['total_kus']} KUs created")
        print(f"  KU types: {ku_types}")
    
    @pytest.mark.asyncio
    async def test_process_paper_with_tables_and_figures(self, setup_pipeline):
        """Test processing paper with complex content"""
        pipeline = await setup_pipeline
        
        test_pdf = "tests/fixtures/transformer_paper.pdf"
        
        source_id = await pipeline.process_document_async(
            test_pdf,
            source_type="pdf",
            source_uri=test_pdf
        )
        
        assert source_id is not None
        
        # Get KUs by type
        kus = pipeline.db.get_kus_by_source(source_id)
        text_kus = [ku for ku in kus if ku['ku_type'] == 'text_chunk']
        table_kus = [ku for ku in kus if ku['ku_type'] == 'table']
        figure_kus = [ku for ku in kus if ku['ku_type'] == 'figure']
        
        assert len(text_kus) > 0, "No text chunks"
        assert len(table_kus) > 0, "No tables extracted"
        assert len(figure_kus) > 0, "No figures extracted"
        
        # Validate table structure
        for table_ku in table_kus:
            raw_content = table_ku.get('raw_content', {})
            assert 'table_data' in raw_content
            enrichment = table_ku.get('enriched_content', {})
            assert enrichment is not None
        
        # Validate figure enrichment
        for figure_ku in figure_kus:
            enrichment = figure_ku.get('enriched_content', {})
            assert enrichment is not None
            assert 'analysis' in enrichment or 'summary' in enrichment
        
        print(f"✓ Processed complex paper:")
        print(f"  - Text chunks: {len(text_kus)}")
        print(f"  - Tables: {len(table_kus)}")
        print(f"  - Figures: {len(figure_kus)}")


class TestPerformanceComparison:
    """Compare VLM vs fallback extraction"""
    
    @pytest.mark.asyncio
    async def test_vlm_vs_fallback_accuracy(self):
        """Compare extraction accuracy"""
        config = get_config()
        
        # Process same PDF with both methods
        test_pdf = "tests/fixtures/test_paper.pdf"
        
        # VLM pipeline
        vlm_pipeline = OptimizedPipeline(
            mongo_handler=MongoDBHandler(config.database.uri, "test_vlm"),
            llm_client=GeminiClient(config.gemini.api_key, config.gemini.vision_model),
            content_embedder=ContentEmbedder(
                TextEmbedder(config.voyage.api_key, config.voyage.text_model),
                MultimodalEmbedder(config.voyage.api_key, config.voyage.multimodal_model)
            ),
            use_vlm_extraction=True
        )
        
        # Fallback pipeline
        fallback_pipeline = OptimizedPipeline(
            mongo_handler=MongoDBHandler(config.database.uri, "test_fallback"),
            llm_client=GeminiClient(config.gemini.api_key, config.gemini.text_model),
            content_embedder=ContentEmbedder(
                TextEmbedder(config.voyage.api_key, config.voyage.text_model),
                MultimodalEmbedder(config.voyage.api_key, config.voyage.multimodal_model)
            ),
            use_vlm_extraction=False
        )
        
        # Process with VLM
        vlm_source_id = await vlm_pipeline.process_document_async(
            test_pdf, "pdf", test_pdf
        )
        vlm_kus = vlm_pipeline.db.get_kus_by_source(vlm_source_id)
        
        # Process with fallback
        fallback_source_id = await fallback_pipeline.process_document_async(
            test_pdf, "pdf", test_pdf
        )
        fallback_kus = fallback_pipeline.db.get_kus_by_source(fallback_source_id)
        
        # Compare results
        vlm_tables = [ku for ku in vlm_kus if ku['ku_type'] == 'table']
        fallback_tables = [ku for ku in fallback_kus if ku['ku_type'] == 'table']
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPARISON")
        print(f"{'='*60}")
        print(f"VLM Method:")
        print(f"  - Total KUs: {len(vlm_kus)}")
        print(f"  - Tables: {len(vlm_tables)}")
        print(f"  - Figures: {len([ku for ku in vlm_kus if ku['ku_type'] == 'figure'])}")
        
        print(f"\nFallback Method:")
        print(f"  - Total KUs: {len(fallback_kus)}")
        print(f"  - Tables: {len(fallback_tables)}")
        print(f"  - Figures: {len([ku for ku in fallback_kus if ku['ku_type'] == 'figure'])}")
        
        # VLM should extract more or equal content
        assert len(vlm_tables) >= len(fallback_tables), "VLM extracted fewer tables"
        
        # Cleanup
        vlm_pipeline.db.close()
        fallback_pipeline.db.close()


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])