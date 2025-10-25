"""
src/extraction/utils.py
"""
import io
import requests
from src.enrichment.schema import LayoutDetectionOutput, TableExtractionOutput, FigureAnalysisOutput, BoundingBox
import trafilatura
import os
import sys
import whisper
import yt_dlp
from moviepy.editor import VideoFileClip
import torch
from youtube_transcript_api import YouTubeTranscriptApi

import fitz
import pdfplumber
import pandas as pd
import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# try:
#     import layoutparser as lp
#     LAYOUTPARSER_AVAILABLE = True
# except ImportError:
#     LAYOUTPARSER_AVAILABLE = False
#     print(" LayoutParser not available. Install with: pip install layoutparser detectron2")

_MODEL_CACHE: Dict[str, whisper.Whisper] = {}
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    logger.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")

class PDFUtils:
    """PDF utilities with complete Windows LayoutParser fix"""
    
    # _layout_model = None
    _init_attempted = False
    @staticmethod
    async def detect_layout_with_vlm(
        page: fitz.Page,
        vlm_client,
        page_num: int
    ) -> Optional[LayoutDetectionOutput]:
        """
        VLM-based layout detection (replaces LayoutParser)
        """
        try:
            # Convert to high-res image
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            prompt = f"""Analyze page {page_num + 1} layout.

Detect all regions with precise bounding boxes:
- Tables (structured data in rows/columns)
- Figures (charts, diagrams, images)
- Text blocks (paragraphs, sections)
- Titles/headings
- Lists

Provide pixel coordinates relative to image dimensions: {pix.width}×{pix.height}"""
            
            result = await vlm_client.create_structured_completion(
                prompt=prompt,
                response_model=LayoutDetectionOutput,
                image=img
            )
            
            if result:
                logger.info(f"Page {page_num + 1}: VLM detected "
                          f"{len(result.regions)} regions")
            
            return result
            
        except Exception as e:
            logger.error(f"VLM layout detection failed: {e}")
            return None
    
    @staticmethod
    async def extract_table_with_vlm(
        page: fitz.Page,
        table_bbox: BoundingBox,
        vlm_client,
        page_num: int
    ) -> Optional[Dict]:
        """
        VLM-based table extraction (replaces Camelot/pdfplumber)
        """
        try:
            # Crop table region with padding
            padding = 10
            crop_rect = fitz.Rect(
                table_bbox.x1 - padding,
                table_bbox.y1 - padding,
                table_bbox.x2 + padding,
                table_bbox.y2 + padding
            )
            
            pix = page.get_pixmap(dpi=300, clip=crop_rect)
            table_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            prompt = """Extract this table's complete structure.

Critical requirements:
1. Identify ALL column headers (handle multi-level if present)
2. Extract ALL data rows accurately
3. Preserve numerical values exactly
4. Note any merged cells or special formatting

Return precise structured data."""
            
            result = await vlm_client.create_structured_completion(
                prompt=prompt,
                response_model=TableExtractionOutput,
                image=table_img
            )
            
            if result:
                logger.info(f"Page {page_num + 1}: Extracted table "
                          f"({result.num_rows}×{result.num_cols})")
                
                return {
                    "type": "table",
                    "page": page_num + 1,
                    "bbox": [table_bbox.x1, table_bbox.y1, 
                            table_bbox.x2, table_bbox.y2],
                    "headers": result.headers,
                    "rows": result.rows,
                    "data": result.to_dataframe().to_dict(orient='records'),
                    "metadata": {
                        "extraction_method": "vlm",
                        "confidence": result.extraction_confidence,
                        "has_merged_cells": result.has_merged_cells
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"VLM table extraction failed: {e}")
            return None
    
    @staticmethod
    async def analyze_figure_with_vlm(
        image_data: bytes,
        caption: str,
        vlm_client,
        page_num: int,
        figure_id: str
    ) -> Optional[Dict]:
        """
        Deep figure analysis with VLM
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            
            prompt = f"""Analyze this research figure from page {page_num + 1}.

Caption: {caption}

Provide comprehensive analysis:
1. Identify figure type (chart type, diagram type, etc.)
2. Extract key findings and trends shown
3. Detect any numerical values, labels, legends
4. Explain research relevance and insights

Focus on information useful for R&D literature review."""
            
            result = await vlm_client.create_structured_completion(
                prompt=prompt,
                response_model=FigureAnalysisOutput,
                image=img
            )
            
            if result:
                logger.info(f"Page {page_num + 1}: Analyzed figure "
                          f"({result.figure_type})")
                
                return {
                    "type": "figure",
                    "page": page_num + 1,
                    "figure_id": figure_id,
                    "image_data": image_data,
                    "width": img.width,
                    "height": img.height,
                    "caption": caption,
                    "enrichment": {
                        "summary": f"{result.figure_type}: {result.relevance_to_research}",
                        "keywords": result.labels_detected[:10],
                        "analysis": {
                            "figure_type": result.figure_type,
                            "key_findings": result.key_findings,
                            "numerical_data": [nd.model_dump() for nd in result.numerical_data],
                            "confidence": result.extraction_confidence
                        }
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"VLM figure analysis failed: {e}")
            return None
    # @staticmethod
    # def _sanitize_windows_path(path_str: str) -> str:
    #     """
    #     Sanitize path by removing Windows-illegal characters
    #     Fixes: 'config.yml?dl=1.lock' → 'config_dl1.lock'
    #     """
    #     # Remove/replace illegal Windows filename characters
    #     illegal_chars = ['?', '<', '>', ':', '"', '|', '*']
    #     sanitized = path_str
        
    #     for char in illegal_chars:
    #         sanitized = sanitized.replace(char, '_')
        
    #     # Also handle URL query strings
    #     if '?' in sanitized or '&' in sanitized:
    #         sanitized = re.sub(r'[?&=]', '_', sanitized)
        
    #     return sanitized

    # @staticmethod
    # def _patch_fvcore_cache():
    #     """
    #     Monkey-patch fvcore's PathManager to handle Windows paths
    #     This fixes the 'Invalid argument' error on Windows
    #     """
    #     try:
    #         from fvcore.common.file_io import PathManager
    #         from iopath.common.file_io import HTTPURLHandler
            
    #         # Store original _get_local_path
    #         original_get_local_path = HTTPURLHandler._get_local_path
            
    #         def patched_get_local_path(self, path, **kwargs):
    #             """Patched version that sanitizes paths for Windows"""
    #             local_path = original_get_local_path(self, path, **kwargs)
                
    #             if sys.platform == "win32" and local_path:
    #                 # Sanitize the path
    #                 parts = list(Path(local_path).parts)
    #                 sanitized_parts = [PDFUtils._sanitize_windows_path(p) for p in parts]
    #                 local_path = str(Path(*sanitized_parts))
                
    #             return local_path
            
    #         # Apply patch
    #         HTTPURLHandler._get_local_path = patched_get_local_path
    #         logger.info(" Applied fvcore Windows path patch")
    #         return True
            
    #     except Exception as e:
    #         logger.warning(f"Could not patch fvcore: {e}")
    #         return False

    # @staticmethod
    # def initialize_layoutparser_model():
    #     """
    #     Initialize LayoutParser with complete Windows fix
    #     """
    #     global LAYOUTPARSER_AVAILABLE
        
    #     if not LAYOUTPARSER_AVAILABLE:
    #         logger.warning("LayoutParser module not available")
    #         return False
        
    #     if PDFUtils._layout_model:
    #         logger.info("LayoutParser model already loaded")
    #         return True
        
    #     if PDFUtils._init_attempted:
    #         logger.warning("LayoutParser initialization already failed")
    #         return False
        
    #     PDFUtils._init_attempted = True
        
    #     try:
    #         logger.info(" Initializing LayoutParser model...")
            
    #         # Check detectron2
    #         try:
    #             import detectron2
    #             logger.info("✅ Detectron2 available")
    #         except ImportError:
    #             logger.error(" Detectron2 not found!")
    #             logger.error("Install: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    #             LAYOUTPARSER_AVAILABLE = False
    #             return False
            
    #         # Windows-specific fixes
    #         if sys.platform == "win32":
    #             logger.info("   Windows detected - Applying fixes...")
                
    #             # Fix 1: Set simple cache directory
    #             cache_dir = Path.home() / ".layoutparser_cache"
    #             cache_dir.mkdir(parents=True, exist_ok=True)
                
    #             # Fix 2: Override torch cache locations
    #             torch_cache = cache_dir / "torch"
    #             torch_cache.mkdir(exist_ok=True)
                
    #             os.environ['TORCH_HOME'] = str(torch_cache)
    #             os.environ['FVCORE_CACHE'] = str(cache_dir / "fvcore")
                
    #             # Fix 3: Patch fvcore to handle URL query strings
    #             PDFUtils._patch_fvcore_cache()
                
    #             logger.info(f"   Cache directory: {cache_dir}")
            
    #         # Initialize model with timeout
    #         import signal
            
    #         def timeout_handler(signum, frame):
    #             raise TimeoutError("Model initialization timeout")
            
    #         # Only set alarm on Unix systems (Windows doesn't support SIGALRM)
    #         if hasattr(signal, 'SIGALRM'):
    #             signal.signal(signal.SIGALRM, timeout_handler)
    #             signal.alarm(120)  # 2 minutes timeout
            
    #         try:
    #             logger.info("   Downloading model (first time only)...")
    #             PDFUtils._layout_model = lp.Detectron2LayoutModel(
    #                 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    #                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
    #                 label_map={
    #                     0: "Text", 
    #                     1: "Title", 
    #                     2: "List", 
    #                     3: "Table", 
    #                     4: "Figure"
    #                 }
    #             )
                
    #             if hasattr(signal, 'SIGALRM'):
    #                 signal.alarm(0)  # Cancel timeout
                
    #             logger.info(" LayoutParser model loaded successfully")
    #             return True
                
    #         except TimeoutError:
    #             if hasattr(signal, 'SIGALRM'):
    #                 signal.alarm(0)
    #             logger.error(" Model initialization timeout (>2min)")
    #             raise
            
    #     except Exception as e:
    #         logger.error(f" LayoutParser initialization failed: {type(e).__name__}: {e}")
    #         logger.error(f"   Error details: {str(e)}")
            
    #         # Provide helpful error messages
    #         if "Invalid argument" in str(e):
    #             logger.error("   → This is a Windows path issue with fvcore")
    #             logger.error("   → Workaround: Set USE_LAYOUTPARSER=false in .env")
    #         elif "connection" in str(e).lower() or "download" in str(e).lower():
    #             logger.error("   → Model download failed (network issue)")
    #             logger.error("   → Try again with better internet connection")
    #         else:
    #             logger.error("   → Unknown error, disabling LayoutParser")
            
    #         logger.warning("Falling back to basic extraction without layout detection")
    #         LAYOUTPARSER_AVAILABLE = False
    #         return False

    # @staticmethod
    # def detect_layout_regions(page):
    #     """Detect layout regions using LayoutParser"""
    #     if not LAYOUTPARSER_AVAILABLE or not PDFUtils._layout_model:
    #         return None
        
    #     try:
    #         # Convert page to image
    #         pix = page.get_pixmap(dpi=150)
    #         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
    #         # Detect layout
    #         layout = PDFUtils._layout_model.detect(img)
            
    #         # Organize by type
    #         regions = {
    #             'text': [],
    #             'tables': [],
    #             'figures': []
    #         }
            
    #         for block in layout:
    #             label = block.type.lower()
    #             bbox = [block.block.x_1, block.block.y_1, 
    #                    block.block.x_2, block.block.y_2]
                
    #             if label in ['text', 'title', 'list']:
    #                 regions['text'].append(bbox)
    #             elif label == 'table':
    #                 regions['tables'].append(bbox)
    #             elif label == 'figure':
    #                 regions['figures'].append(bbox)
            
    #         logger.info(f"Detected: {len(regions['text'])} text, "
    #                    f"{len(regions['tables'])} tables, "
    #                    f"{len(regions['figures'])} figures")
            
    #         return regions
            
    #     except Exception as e:
    #         logger.error(f"Layout detection failed: {e}")
    #         return None
        
    @staticmethod
    def extract_all_text_optimized(page: fitz.Page, min_length: int = 50) -> str:
        """
        FIXED: Extract ALL text from page
        Returns full page text as single string
        """
        try:
            # Method 1: Get all text as single string
            full_text = page.get_text("text")
            
            # DEBUG: Log what we got
            logger.debug(f"Page {page.number + 1}: Extracted {len(full_text)} chars")
            
            if len(full_text.strip()) >= min_length:
                return full_text.strip()
            
            # Fallback: Try blocks method if text is too short
            logger.debug(f"Page {page.number + 1}: Text too short, trying blocks method")
            blocks = page.get_text("dict")["blocks"]
            text_parts = []
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))
            
            full_text = " ".join(text_parts)
            logger.debug(f"Page {page.number + 1}: Blocks method got {len(full_text)} chars")
            
            return full_text.strip()
        
        except Exception as e:
            logger.error(f"Failed to extract text from page {page.number + 1}: {e}")
            return ""

    @staticmethod
    def extract_images_improved(
        page: fitz.Page, 
        page_num: int, 
        min_width: int = 100,
        min_height: int = 100,
        min_pixels: int = 10000
    ) -> List[Dict]:
        """Extract images with size filtering"""
        image_objects = []
        
        try:
            image_list = page.get_images(full=True)
            logger.debug(f"Page {page_num + 1}: Found {len(image_list)} images")
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    
                    if not base_image:
                        continue
                    
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Size filtering
                    if width < min_width or height < min_height:
                        logger.debug(f"Skipping small image: {width}x{height}")
                        continue
                    
                    if width * height < min_pixels:
                        logger.debug(f"Skipping low-pixel image: {width*height} pixels")
                        continue
                    
                    image_data = base_image.get("image")
                    if not image_data:
                        continue
                    
                    # Verify image
                    try:
                        pil_image = Image.open(io.BytesIO(image_data))
                        if pil_image.width < min_width or pil_image.height < min_height:
                            continue
                    except Exception as e:
                        logger.warning(f"Could not verify image {img_idx}: {e}")
                        continue
                    
                    # Create figure object
                    figure = {
                        "type": "figure",
                        "image_data": image_data,
                        "page": page_num + 1,
                        "width": width,
                        "height": height,
                        "figure_id": f"page{page_num+1}_fig{img_idx+1}",
                        "xref": xref
                    }
                    
                    image_objects.append(figure)
                    logger.debug(f"✓ Extracted image {img_idx+1}: {width}x{height}px")
                
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx+1}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Image extraction failed on page {page_num+1}: {e}")
        
        logger.info(f"Page {page_num + 1}: Extracted {len(image_objects)} valid images")
        return image_objects

    @staticmethod
    def _extract_tables_advanced(page: fitz.Page, page_num: int, pdf_path: str) -> List[Dict]:
        """Advanced table extraction using multiple methods"""
        table_objects = []
        
        # Try Camelot first
        # try:
            # tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')
            # if not tables or (tables and tables[0].accuracy < 60):
            #     tables.extend(camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream', edge_tol=100))
            
        #     for idx, table in enumerate(tables):
        #         if table.accuracy > 50:
        #             processed = TableStructureReconstructor.process_table(table.df)
        #             if processed:
        #                 table_objects.append({
        #                     "type": "table",
        #                     "page": page_num + 1,
        #                     "bbox": list(table._bbox) if hasattr(table, '_bbox') else [],
        #                     "table_id": f"table_{page_num+1}_{idx+1}",
        #                     **processed,
        #                     "extraction_method": f"camelot_{table.flavor}",
        #                     "accuracy": table.accuracy
        #                 })
        # except Exception as e:
        #     logger.warning(f"Camelot failed: {e}")

        # Fallback to pdfplumber
        if not table_objects:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    plumber_tables = pdf.pages[page_num].extract_tables()
                    for idx, raw_table in enumerate(plumber_tables):
                        if raw_table and len(raw_table) > 1:
                            processed = TableStructureReconstructor.process_table(pd.DataFrame(raw_table))
                            if processed:
                                table_objects.append({
                                    "type": "table",
                                    "page": page_num + 1,
                                    "bbox": [],
                                    "table_id": f"table_{page_num+1}_{idx+1}",
                                    **processed,
                                    "extraction_method": "pdfplumber"
                                })
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        return table_objects

    @staticmethod
    def extract_text_blocks(
        page: fitz.Page, 
        page_num: int, 
        min_length: int = 50
    ) -> List[Dict]:
        """
        FIXED: Return text as single block (will be chunked later)
        """
        full_text = PDFUtils.extract_all_text_optimized(page, min_length=10)  # Lower threshold
        
        if not full_text or len(full_text) < min_length:
            logger.warning(f"Page {page_num + 1}: No text extracted (got {len(full_text)} chars)")
            return []
        
        logger.info(f"Page {page_num + 1}: Extracted {len(full_text)} chars of text")
        
        # Return as single chunk (will be split by embedder later)
        return [{
            "type": "text_chunk",
            "text": full_text,
            "page": page_num + 1,
            "bbox": list(page.rect)
        }]

    @staticmethod
    def extract_images(page: fitz.Page, page_num: int, min_pixels: int = 10000) -> List[Dict]:
        """Wrapper for improved image extraction"""
        return PDFUtils.extract_images_improved(page, page_num, min_pixels=min_pixels)

    @staticmethod
    def extract_tables(page: fitz.Page, page_num: int, pdf_path: Optional[str] = None, 
                       use_layoutparser: bool = True) -> List[Dict]:
        """Extract tables with multiple fallback methods"""
        table_objects = []
        
        # Method 1: PyMuPDF built-in
        try:
            tables_found = list(page.find_tables())
            logger.debug(f"PyMuPDF found {len(tables_found)} tables")

            for idx, t in enumerate(tables_found):
                try:
                    df = t.to_pandas()
                    if not df.empty:
                        processed = TableStructureReconstructor.process_table(df)
                        if processed:
                            table_objects.append({
                                "type": "table",
                                "page": page_num + 1,
                                "bbox": list(t.bbox),
                                "table_id": f"table_{page_num+1}_{idx+1}_pymupdf",
                                **processed,
                                "extraction_method": "pymupdf",
                                "has_hierarchical_headers": processed["metadata"]["has_hierarchical_headers"]
                            })
                except Exception as e:
                    logger.warning(f"Failed to process PyMuPDF table {idx+1}: {e}")
                    
        except Exception as e:
            logger.warning(f"PyMuPDF table extraction failed: {e}")

        # Method 2: Advanced extraction if needed
        if not table_objects and pdf_path:
            table_objects.extend(PDFUtils._extract_tables_advanced(page, page_num, pdf_path))
        
        if table_objects:
            logger.info(f"✓ Extracted {len(table_objects)} tables from page {page_num+1}")
        
        return table_objects

class VideoUtils:
    @staticmethod
    def load_model(model_name: str = "base") -> whisper.Whisper:
        """
        Tải mô hình Whisper. Nếu mô hình đã được tải, nó sẽ trả về
        thể hiện (instance) đã được cache để tránh tải lại.

        Args:
            model_name: Kích thước của mô hình Whisper.

        Returns:
            Đối tượng mô hình Whisper đã được tải.
        """
        if model_name not in _MODEL_CACHE:
            logger.info(f"Loading Whisper model '{model_name}' onto device '{_DEVICE}'...")
            _MODEL_CACHE[model_name] = whisper.load_model(model_name, device=_DEVICE)
            logger.info(f"Model '{model_name}' loaded successfully.")
        return _MODEL_CACHE[model_name]

    @staticmethod
    def download_youtube_audio(url: str, output_dir: Path) -> Optional[str]:
        """
        Chỉ tải xuống luồng âm thanh từ YouTube và chuyển đổi nó thành MP3.
        """
        logger.info(f"Downloading audio for: {url}")
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
            'quiet': True, 'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = ydl.prepare_filename(info).replace(info['ext'], 'mp3')
                if Path(audio_path).exists():
                    logger.info(f"Audio downloaded to: {audio_path}")
                    return audio_path
                return None
        except Exception as e:
            logger.error(f"Failed to download audio from {url}: {e}")
            return None

    @staticmethod
    def extract_audio_from_file(video_path: str) -> Optional[str]:
        """
        Trích xuất âm thanh từ một tệp video cục bộ.
        """
        logger.info(f"Extracting audio from: {video_path}")
        try:
            video_file = Path(video_path)
            audio_path = video_file.with_suffix('.mp3')
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    return None
                clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            logger.info(f"Audio extracted to: {audio_path}")
            return str(audio_path)
        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path}: {e}")
            return None

    @staticmethod
    def transcribe_audio(audio_path: str, model_name: str) -> List[Dict]:
        """
        Chuyển đổi một tệp âm thanh thành các đoạn văn bản bằng mô hình Whisper đã được cache.
        """
        logger.info(f"Transcribing: {audio_path}")
        try:
            model = VideoUtils.load_model(model_name)
            # Tối ưu: Bật batch_size lớn hơn nếu dùng GPU để tăng tốc
            transcribe_options = dict(task="transcribe", verbose=False)
            if _DEVICE == "cuda":
                transcribe_options["batch_size"] = 16 
                
            result = model.transcribe(audio_path, **transcribe_options)
            logger.info(f"Transcription complete for: {audio_path}")
            return result.get("segments", [])
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            return []
    @staticmethod
    def get_youtube_transcript_direct(video_url: str) -> List[Dict]:
        """
        Get transcript directly from YouTube without downloading audio.
        
        Much faster than download -> transcribe approach.
        Falls back to audio download if transcript not available.
        
        Returns:
            List of transcript segments with 'text', 'start', 'duration'
        """
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            logger.warning("YouTube Transcript API not available, falling back to audio download")
            return None
        
        try:
            video_id = VideoUtils.extract_video_id(video_url)
            
            if not video_id:
                logger.error(f"Could not extract video ID from: {video_url}")
                return None
            
            logger.info(f"Fetching transcript for video ID: {video_id}")
            
            # Try to get transcript (priority: English -> any available language)
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=['en', 'en-US', 'en-GB']
                )
            except:
                # If English not available, get any available transcript
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_generated_transcript(['en']).fetch()
            
            if not transcript:
                logger.warning(f"No transcript available for {video_id}")
                return None
            
            # Convert to Whisper-like format for compatibility
            segments = []
            for idx, entry in enumerate(transcript):
                segments.append({
                    'id': idx,
                    'text': entry['text'],
                    'start': entry['start'],
                    'end': entry['start'] + entry['duration'],
                    'duration': entry['duration']
                })
            
            logger.info(f"✓ Retrieved {len(segments)} transcript segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to get YouTube transcript: {e}")
            return None
    
class WebsiteUtils:
    @staticmethod
    def extract_main_text(url: str, timeout: int = 10) -> Optional[str]:
        """
        Lấy nội dung HTML từ một URL và trích xuất chỉ phần văn bản chính,
        loại bỏ các thành phần thừa như menu, quảng cáo, và footer.

        Args:
            url: URL của trang web cần trích xuất.
            timeout: Thời gian chờ tối đa cho request (giây).

        Returns:
            Chuỗi văn bản chính đã được làm sạch, hoặc None nếu có lỗi xảy ra.
        """
        logger.info(f"Fetching content from: {url}")
        try:
            # Gửi yêu cầu HTTP để lấy nội dung trang web
            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            # Kiểm tra nếu request không thành công
            response.raise_for_status()

            # Sử dụng trafilatura để trích xuất nội dung chính từ HTML
            # Đây là bước "thần kỳ", nó tự động làm sạch và lấy ra phần quan trọng nhất
            main_text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
            
            if main_text:
                logger.info(f"Successfully extracted text from: {url}")
                return main_text
            else:
                logger.warning(f"Could not extract main content from {url}, the page might be JavaScript-heavy or have no main text.")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {url}: {e}")
            return None