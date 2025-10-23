"""
src/config.py
Centralized configuration management with validation
"""
import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseModel):
    """MongoDB configuration"""
    uri: str = Field(..., description="MongoDB connection string")
    database_name: str = Field(default="multimodal_rag_db")
    vector_index_name: str = Field(default="vector_index")
    
    @validator('uri')
    def validate_uri(cls, v):
        if not v or not v.startswith('mongodb'):
            raise ValueError("Invalid MongoDB URI")
        return v


class VoyageConfig(BaseModel):
    """Voyage AI configuration"""
    api_key: str = Field(..., description="Voyage AI API key")
    text_model: str = Field(default="voyage-3")
    multimodal_model: str = Field(default="voyage-multimodal-3")
    rerank_model: str = Field(default="rerank-2.5")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.startswith('pa-'):
            raise ValueError("Invalid Voyage API key")
        return v


class GeminiConfig(BaseModel):
    """Google Gemini configuration"""
    api_key: str = Field(..., description="Gemini API key")
    model_name: str = Field(default="gemini-2.0-flash-exp")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.startswith('AIzaSy'):
            raise ValueError("Invalid Gemini API key")
        return v


class QwenConfig(BaseModel):
    """Qwen/DashScope configuration"""
    api_key: str = Field(..., description="DashScope API key")
    model_name: str = Field(default="qwen-vl-max")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError("Invalid DashScope API key")
        return v


class ProcessingConfig(BaseModel):
    """Pipeline processing configuration"""
    max_workers: int = Field(default=4, ge=1, le=32)
    batch_size: int = Field(default=16, ge=1, le=128)
    use_layoutparser: bool = Field(default=False)
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(default="base")
    max_concurrent_enrichment: int = Field(default=10, ge=1, le=50)


class QueryConfig(BaseModel):
    """Query engine configuration"""
    use_cache: bool = Field(default=True)
    default_initial_k: int = Field(default=20, ge=1, le=100)
    default_final_k: int = Field(default=5, ge=1, le=50)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)


class PathConfig(BaseModel):
    """File paths configuration"""
    documents_dir: str = Field(default="documents")
    temp_dir: str = Field(default="temp")
    output_dir: Optional[str] = Field(default=None)
    log_file: Optional[str] = Field(default=None)


class Config(BaseModel):
    """Main configuration class combining all sub-configs"""
    database: DatabaseConfig
    voyage: VoyageConfig
    gemini: GeminiConfig
    qwen: QwenConfig
    processing: ProcessingConfig
    query: QueryConfig
    paths: PathConfig
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            database=DatabaseConfig(
                uri=os.getenv("MONGO_URI", ""),
                database_name=os.getenv("DB_NAME", "multimodal_rag_db"),
                vector_index_name=os.getenv("VECTOR_INDEX_NAME", "vector_index")
            ),
            voyage=VoyageConfig(
                api_key=os.getenv("VOYAGE_API_KEY", ""),
                text_model=os.getenv("VOYAGE_TEXT_MODEL", "voyage-3"),
                multimodal_model=os.getenv("VOYAGE_MULTIMODAL_MODEL", "voyage-multimodal-3"),
                rerank_model=os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
            ),
            gemini=GeminiConfig(
                api_key=os.getenv("GEMINI_API_KEY", ""),
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
            ),
            qwen=QwenConfig(
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                model_name=os.getenv("QWEN_MODEL", "qwen-vl-max")
            ),
            processing=ProcessingConfig(
                max_workers=int(os.getenv("MAX_WORKERS", "4")),
                batch_size=int(os.getenv("BATCH_SIZE", "16")),
                use_layoutparser=os.getenv("USE_LAYOUTPARSER", "false").lower() == "true",
                whisper_model=os.getenv("WHISPER_MODEL", "base"),
                max_concurrent_enrichment=int(os.getenv("MAX_CONCURRENT_ENRICHMENT", "10"))
            ),
            query=QueryConfig(
                use_cache=os.getenv("USE_QUERY_CACHE", "true").lower() == "true",
                default_initial_k=int(os.getenv("INITIAL_K", "20")),
                default_final_k=int(os.getenv("FINAL_K", "5")),
                vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7"))
            ),
            paths=PathConfig(
                documents_dir=os.getenv("DOCUMENTS_DIR", "documents"),
                temp_dir=os.getenv("TEMP_DIR", "temp"),
                output_dir=os.getenv("OUTPUT_DIR"),
                log_file=os.getenv("LOG_FILE")
            )
        )
    
    def validate_all(self) -> list[str]:
        """Validate all configurations and return list of errors"""
        errors = []
        
        # Check required API keys
        if not self.voyage.api_key or self.voyage.api_key == "":
            errors.append("VOYAGE_API_KEY is not set")
        
        if not self.gemini.api_key or self.gemini.api_key == "":
            errors.append("GEMINI_API_KEY is not set")
        
        if not self.qwen.api_key or self.qwen.api_key == "":
            errors.append("DASHSCOPE_API_KEY is not set")
        
        if not self.database.uri or self.database.uri == "":
            errors.append("MONGO_URI is not set")
        
        return errors
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print(f"\n Database:")
        print(f"  - URI: {self.database.uri[:30]}...") 
        print(f"  - Database: {self.database.database_name}")
        print(f"  - Vector Index: {self.database.vector_index_name}")
        
        print(f"\n Voyage AI:")
        print(f"  - Text Model: {self.voyage.text_model}")
        print(f"  - Multimodal Model: {self.voyage.multimodal_model}")
        print(f"  - Rerank Model: {self.voyage.rerank_model}")
        
        print(f"\n Gemini:")
        print(f"  - Model: {self.gemini.model_name}")
        print(f"  - Temperature: {self.gemini.temperature}")
        
        print(f"\n Qwen:")
        print(f"  - Model: {self.qwen.model_name}")
        
        print(f"\n  Processing:")
        print(f"  - Workers: {self.processing.max_workers}")
        print(f"  - Batch Size: {self.processing.batch_size}")
        print(f"  - LayoutParser: {self.processing.use_layoutparser}")
        print(f"  - Whisper Model: {self.processing.whisper_model}")
        
        print(f"\n Query:")
        print(f"  - Cache Enabled: {self.query.use_cache}")
        print(f"  - Initial K: {self.query.default_initial_k}")
        print(f"  - Final K: {self.query.default_final_k}")
        
        print(f"\n Paths:")
        print(f"  - Documents: {self.paths.documents_dir}")
        print(f"  - Temp: {self.paths.temp_dir}")
        print("="*70 + "\n")


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create configuration singleton"""
    global _config
    if _config is None:
        _config = Config.from_env()
        
        # Validate configuration
        errors = _config.validate_all()
        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )
    
    return _config


def reload_config():
    """Reload configuration from environment"""
    global _config
    _config = None
    return get_config()