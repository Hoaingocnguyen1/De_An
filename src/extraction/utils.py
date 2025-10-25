"""
src/extraction/utils.py
"""
import io
import requests
from src.enrichment.schema import LayoutDetectionOutput, TableExtractionOutput, FigureAnalysisOutput, BoundingBox
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
    async def detect_layout_with_vlm(page_image: Image.Image, vlm_client, page_num: int) -> Optional[LayoutDetectionOutput]:
        """VLM detect tables & figures"""
        try:
            prompt = f"""Analyze page {page_num + 1} layout.

    Detect ALL regions with PRECISE bounding boxes (pixel coordinates):
    - Tables (structured data in rows/columns)
    - Figures (charts, diagrams, images)

    Image size: {page_image.width}×{page_image.height}px"""
            
            result = await vlm_client.create_structured_completion(
                prompt=prompt,
                response_model=LayoutDetectionOutput,
                image=page_image
            )
            
            if result:
                logger.info(f"Page {page_num+1}: VLM detected {len(result.regions)} regions")
            return result
            
        except Exception as e:
            logger.error(f"VLM layout detection failed: {e}")
            return None
    
    @staticmethod
    async def extract_table_with_vlm(table_image: Image.Image, vlm_client, page_num: int, caption: str = "") -> Optional[Dict]:
        """VLM extract table structure with robust handling"""
        try:
            prompt = f"""Extract this table's structure. Caption: {caption}

    INSTRUCTIONS:
    1. Identify ALL column headers (if no headers visible, use "Col1", "Col2", etc.)
    2. Extract ALL data rows as strings
    3. Count rows and columns accurately
    4. Set has_merged_cells to true if you see merged cells
    5. Set extraction_confidence between 0.0 and 1.0

    IMPORTANT: You MUST return a valid JSON object with all fields, even if the table is difficult to read.
    """
            result = await vlm_client.create_structured_completion(
                prompt=prompt,
                response_model=TableExtractionOutput,
                image=table_image
            )
            
            if not result:
                logger.warning(f"Page {page_num+1}: VLM returned None for table")
                return None
            
            # Validate result
            if not result.headers or len(result.headers) == 0:
                logger.warning(f"Page {page_num+1}: Table has no headers")
                return None
            
            if result.num_cols < 1:
                logger.warning(f"Page {page_num+1}: Table has 0 columns")
                return None
            
            # Convert to dict
            df = result.to_dataframe()
            if df.empty:
                logger.warning(f"Page {page_num+1}: Table DataFrame is empty")
                return None
            
            logger.info(f"Page {page_num+1}: Extracted table ({result.num_rows}×{result.num_cols})")
            
            return {
                "type": "table",
                "page": page_num + 1,
                "headers": result.headers,
                "rows": result.rows,
                "data": df.to_dict(orient='records'),
                "caption": caption,
                "metadata": {
                    "extraction_method": "vlm",
                    "confidence": result.extraction_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Page {page_num+1}: VLM table extraction failed - {e}")
            return None
    
    @staticmethod
    async def analyze_figure_with_vlm(figure_image: Image.Image, vlm_client, page_num: int, caption: str = "") -> Optional[Dict]:
        """VLM analyze figure"""
        try:
            prompt = f"""Analyze this research figure. Caption: {caption}

    Provide:
    1. Figure type (chart type, diagram type, etc.)
    2. Key findings and trends
    3. Numerical values, labels, legends
    4. Research relevance

    Focus on information useful for R&D."""
            
            result = await vlm_client.create_structured_completion(
                prompt=prompt,
                response_model=FigureAnalysisOutput,
                image=figure_image
            )
            
            if result:
                logger.info(f"Page {page_num+1}: Analyzed figure ({result.figure_type})")
                return {
                    "type": "figure",
                    "page": page_num + 1,
                    "caption": caption,
                    "figure_type": result.figure_type,
                    "key_findings": result.key_findings,
                    "numerical_data": [nd.model_dump() for nd in result.numerical_data],
                    "confidence": result.extraction_confidence
                }
            return None
            
        except Exception as e:
            logger.error(f"VLM figure analysis failed: {e}")
            return None

    @staticmethod
    def extract_all_text_optimized(page: fitz.Page, min_length: int = 1000) -> str:
        """
        Extract ALL text from page
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
    def extract_text_blocks(
        page: fitz.Page, 
        page_num: int, 
        min_length: int = 50
    ) -> List[Dict]:
        """
        Return text as single block (will be chunked later)
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


class VideoUtils:
    @staticmethod
    def load_model(model_name: str = "base") -> whisper.Whisper:
        if model_name not in _MODEL_CACHE:
            logger.info(f"Loading Whisper model '{model_name}' onto device '{_DEVICE}'...")
            _MODEL_CACHE[model_name] = whisper.load_model(model_name, device=_DEVICE)
            logger.info(f"Model '{model_name}' loaded successfully.")
        return _MODEL_CACHE[model_name]

    @staticmethod
    def download_youtube_audio(url: str, output_dir: Path) -> Optional[str]:
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