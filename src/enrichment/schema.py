from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(ge=0, description="Left X coordinate")
    y1: float = Field(ge=0, description="Top Y coordinate")
    x2: float = Field(gt=0, description="Right X coordinate")
    y2: float = Field(gt=0, description="Bottom Y coordinate")
    
    def validate_box(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

class LayoutRegion(BaseModel):
    """Single layout region detected by VLM"""
    type: Literal["table", "figure"]
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    description: str = Field(max_length=10000)

class LayoutDetectionOutput(BaseModel):
    """Complete page layout analysis"""
    regions: List[LayoutRegion] = Field(min_items=0)
    page_type: Literal["single_column", "double_column", "mixed"] = "single_column"
    has_header: bool = False
    has_footer: bool = False

class TableCell(BaseModel):
    """Single table cell with position"""
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: str
    is_header: bool = False
    rowspan: int = Field(ge=1, default=1)
    colspan: int = Field(ge=1, default=1)

class TableExtractionOutput(BaseModel):
    """Structured table extraction with flexible validation"""
    headers: List[str] = Field(
        min_length=1, 
        description="Column headers (at least 1 required)"
    )
    rows: List[List[str]] = Field(
        default_factory=list, 
        description="Data rows (can be empty)"
    )
    metadata: dict = Field(default_factory=dict)
    num_rows: int = Field(ge=0, description="Number of data rows")
    num_cols: int = Field(ge=1, description="Number of columns (at least 1)")
    has_merged_cells: bool = False
    extraction_confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        default=0.8,
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
    unit: Optional[str] = None

class FigureAnalysisOutput(BaseModel):
    """Deep figure analysis for R&D"""
    figure_type: Literal[
        "bar_chart", "line_plot", "scatter_plot", "heatmap",
        "architecture_diagram", "flowchart", "confusion_matrix",
        "photo", "illustration", "other"
    ]
    key_findings: List[str] = Field(max_items=10, description="Main insights")
    numerical_data: List[NumericalData] = Field(default_factory=list)
    labels_detected: List[str] = Field(default_factory=list)
    has_legend: bool = False
    relevance_to_research: str = Field(max_length=10000)
    extraction_confidence: float = Field(ge=0.0, le=1.0)

class ResearchMetadata(BaseModel):
    """R&D-specific enrichment"""
    paper_section: Literal[
        "abstract", "introduction", "related_work", "methodology",
        "experiments", "results", "discussion", "conclusion", "unknown"
    ] = "unknown"
    
    methodology_keywords: List[str] = Field(
        default_factory=list,
        description="ML methods, algorithms mentioned"
    )
    
    metrics_mentioned: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics with values"
    )
    
    citations_mentioned: List[str] = Field(
        default_factory=list,
        max_items=10,
        description="Paper citations in this chunk"
    )
    
    datasets_mentioned: List[str] = Field(default_factory=list)
    
    contributions: List[str] = Field(
        default_factory=list,
        max_items=3,
        description="Key contributions in this section"
    )

class EnrichmentOutput(BaseModel):
    """Enrichment with research metadata"""
    summary: str = Field(min_length=20, max_length=500)
    keywords: List[str] = Field(min_items=1, max_items=10)
    research_metadata: Optional[ResearchMetadata] = None