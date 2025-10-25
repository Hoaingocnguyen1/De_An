import asyncio
from src.enrichment.clients.gemini_client import GeminiClient
from src.enrichment.schema import (
    LayoutDetectionOutput, 
    TableExtractionOutput,
    FigureAnalysisOutput
)
from src.config import get_config
from PIL import Image

async def test_layout_detection():
    """Test VLM layout detection"""
    config = get_config()
    client = GeminiClient(
        api_key=config.gemini.api_key,
        model_name=config.gemini.vision_model
    )
    
    # Load test image
    test_image = Image.open("tests/fixtures/test_page.png")
    
    prompt = """Analyze this document page layout.
    
Identify all regions with their bounding boxes and types.
Focus on: tables, figures, and caption """
    
    result = await client.create_structured_completion(
        prompt=prompt,
        response_model=LayoutDetectionOutput,
        image=test_image
    )
    
    assert result is not None, "Layout detection failed"
    assert len(result.regions) > 0, "No regions detected"
    
    print(f"✓ Detected {len(result.regions)} regions:")
    for region in result.regions:
        print(f"  - {region.type}: {region.description}")
        print(f"    bbox: [{region.bbox.x1:.0f}, {region.bbox.y1:.0f}, "
              f"{region.bbox.x2:.0f}, {region.bbox.y2:.0f}]")
    
    return result

async def test_table_extraction():
    """Test VLM table extraction"""
    config = get_config()
    client = GeminiClient(
        api_key=config.gemini.api_key,
        model_name=config.gemini.vision_model
    )
    
    table_image = Image.open("tests/fixtures/test_table.png")
    
    prompt = """Extract the complete table structure.
    
Provide all headers and data rows.
Handle merged cells if present."""
    
    result = await client.create_structured_completion(
        prompt=prompt,
        response_model=TableExtractionOutput,
        image=table_image
    )
    
    assert result is not None, "Table extraction failed"
    assert result.num_rows > 0, "No rows extracted"
    assert result.num_cols > 0, "No columns extracted"
    
    print(f"✓ Extracted table: {result.num_rows} rows × {result.num_cols} cols")
    print(f"  Headers: {result.headers}")
    print(f"  Confidence: {result.extraction_confidence:.2f}")
    
    # Test DataFrame conversion
    df = result.to_dataframe()
    print(f"\n{df}")
    
    return result

async def test_figure_analysis():
    """Test VLM figure analysis"""
    config = get_config()
    client = GeminiClient(
        api_key=config.gemini.api_key,
        model_name=config.gemini.vision_model
    )
    
    figure_image = Image.open("tests/fixtures/test_figure.png")
    
    prompt = """Analyze this research figure in detail.
    
Identify:
- Figure type (chart, diagram, etc.)
- Key findings or trends
- Numerical values visible
- Labels and legend
- Research relevance"""
    
    result = await client.create_structured_completion(
        prompt=prompt,
        response_model=FigureAnalysisOutput,
        image=figure_image
    )
    
    assert result is not None, "Figure analysis failed"
    assert result.figure_type != "other", "Figure type not identified"
    
    print(f"✓ Figure analysis:")
    print(f"  Type: {result.figure_type}")
    print(f"  Findings: {result.key_findings}")
    print(f"  Numerical data: {result.numerical_data}")
    print(f"  Confidence: {result.extraction_confidence:.2f}")
    
    return result

async def main():
    print("="*80)
    print("TESTING STRUCTURED OUTPUT")
    print("="*80)
    
    print("\n[1/3] Testing Layout Detection...")
    await test_layout_detection()
    
    print("\n[2/3] Testing Table Extraction...")
    await test_table_extraction()
    
    print("\n[3/3] Testing Figure Analysis...")
    await test_figure_analysis()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())