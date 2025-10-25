from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(description="Left X coordinate")
    y1: float = Field(description="Top Y coordinate")
    x2: float = Field(description="Right X coordinate")
    y2: float = Field(description="Bottom Y coordinate")
    
    def validate_box(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

class LayoutRegion(BaseModel):
    """Single layout region detected by VLM"""
    type: Literal["table", "figure"]
    bbox: BoundingBox
    confidence: float
    description: str

class LayoutDetectionOutput(BaseModel):
    """Complete page layout analysis"""
    regions: List[LayoutRegion]
    page_type: Literal["single_column", "double_column", "mixed"]
    has_header: bool
    has_footer: bool

class TableCell(BaseModel):
    """Single table cell with position"""
    row: int
    col: int
    value: str
    is_header: bool
    rowspan: int
    colspan: int

class TableExtractionOutput(BaseModel):
    """Structured table extraction with flexible validation"""
    headers: List[str] = Field(
        description="Column headers (at least 1 required)"
    )
    rows: List[List[str]] = Field(description="Data rows (can be empty)")
    num_rows: int = Field(description="Number of data rows")
    num_cols: int = Field(description="Number of columns (at least 1)")
    has_merged_cells: bool
    extraction_confidence: float = Field( 
        description="Confidence score"
    )
    
    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd
        if not self.rows:
            # Empty table with headers only
            return pd.DataFrame(columns=self.headers)
        return pd.DataFrame(self.rows, columns=self.headers)
    
class NumericalData(BaseModel):
    """Numerical data extracted from figure"""
    label: str
    value: float
    unit: Optional[str]

class FigureAnalysisOutput(BaseModel):
    """Deep figure analysis for R&D"""
    figure_type: Literal[
        "bar_chart", "line_plot", "scatter_plot", "heatmap",
        "architecture_diagram", "flowchart", "confusion_matrix",
        "photo", "illustration", "other"
    ]
    key_findings: List[str] = Field(description="Main insights")
    numerical_data: List[NumericalData]
    labels_detected: List[str]
    has_legend: bool
    relevance_to_research: str
    extraction_confidence: float

class ResearchMetadata(BaseModel):
    """R&D-specific enrichment"""
    paper_section: Literal[
        "abstract", "introduction", "related_work", "methodology",
        "experiments", "results", "discussion", "conclusion", "unknown"
    ]
    
    methodology_keywords: List[str] = Field(
        description="ML methods, algorithms mentioned"
    )
    
    metrics_mentioned: Dict[str, float] = Field(
        description="Performance metrics with values"
    )
    
    citations_mentioned: List[str] = Field(
        description="Paper citations in this chunk"
    )
    
    datasets_mentioned: List[str]
    
    contributions: List[str] = Field(
        description="Key contributions in this section"
    )

class EnrichmentOutput(BaseModel):
    """Enrichment with research metadata"""
    summary: str
    keywords: List[str]
    research_metadata: Optional[ResearchMetadata]