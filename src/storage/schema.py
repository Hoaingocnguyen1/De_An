"""
src/storage/schema.py
MongoDB Schema Definitions for Knowledge Units
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class RawContent(BaseModel):
    """Raw content extracted from source"""
    text: Optional[str] = None
    asset_uri: Optional[str] = None  # GCS path to image/video frame


class Context(BaseModel):
    """Positional and contextual metadata"""
    page_number: Optional[int] = None
    bounding_box: Optional[List[float]] = None  # [x1, y1, x2, y2]
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None
    direct_url: Optional[str] = None


class EnrichedContent(BaseModel):
    """AI-enriched metadata"""
    summary: Optional[str] = None
    keywords: Optional[List[str]] = []
    analysis_model: str = "gemini-1.5-flash"


class EmbeddingVector(BaseModel):
    """Vector embedding with model info"""
    vector: List[float]
    model: str = "openai/clip-vit-base-patch32"


class Embeddings(BaseModel):
    """Multi-modal embeddings"""
    text_embedding: Optional[EmbeddingVector] = None
    image_embedding: Optional[EmbeddingVector] = None


class KnowledgeUnit(BaseModel):
    """Main Knowledge Unit schema"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    # Source metadata
    source_id: PyObjectId
    source_type: Literal["pdf", "youtube", "video", "text"]
    source_uri: str
    
    # KU metadata
    ku_id: str  # Human-readable ID like "pdf_page5_fig1"
    ku_type: Literal["text_chunk", "figure", "table", "video_frame"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Content layers
    raw_content: RawContent
    context: Context
    enriched_content: Optional[EnrichedContent] = None
    embeddings: Optional[Embeddings] = None
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SourceDocument(BaseModel):
    """Source document tracking schema"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    source_type: Literal["pdf", "youtube", "video", "text"]
    source_uri: str
    original_filename: Optional[str] = None
    
    # Processing status
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    error_message: Optional[str] = None
    
    # Statistics
    total_kus: int = 0
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}