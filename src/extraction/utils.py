"""
src/extraction/utils.py
"""
import requests
import trafilatura

import whisper
import yt_dlp
from moviepy.editor import VideoFileClip
import torch
from youtube_transcript_api import YouTubeTranscriptApi

import fitz
import pdfplumber
import camelot
import pandas as pd
import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    print(" LayoutParser not available. Install with: pip install layoutparser detectron2")

_MODEL_CACHE: Dict[str, whisper.Whisper] = {}
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    logger.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")


class TableStructureReconstructor:
    """Smart table structure reconstruction for multi-level headers"""

    @staticmethod
    def detect_header_rows(df: pd.DataFrame, max_header_rows: int = 5) -> int:
        """
        Detect how many rows are headers (not data)
        """
        if len(df) < 2:
            return 1

        header_count = 0
        for i in range(min(max_header_rows, len(df))):
            row = df.iloc[i]
            row_str = ' '.join(str(x) for x in row if pd.notna(x) and str(x).strip())
            tokens = row_str.split()
            if not tokens:
                header_count += 1
                continue
            
            numeric_tokens = sum(1 for t in tokens if re.match(r'^[\d\.\,\%\±]+$', t))
            text_tokens = len(tokens) - numeric_tokens
            
            header_keywords = ['retrieval', 'caption', 'accuracy', 'score', 'model', 'method', 'task', 'dataset', 'metric']
            has_keywords = any(kw in row_str.lower() for kw in header_keywords)
            
            if text_tokens > numeric_tokens or has_keywords or numeric_tokens == 0:
                header_count += 1
            else:
                break
        
        return max(1, header_count)

    @staticmethod
    def clean_noise_rows(df: pd.DataFrame, caption: str = "") -> pd.DataFrame:
        """Remove noise rows (captions, empty rows)"""
        cleaned_rows = []
        for _, row in df.iterrows():
            row_text = ' '.join(str(x) for x in row if pd.notna(x) and str(x).strip())
            if caption and len(caption) > 20:
                caption_words = set(caption.lower().split()[:10])
                row_words = set(row_text.lower().split())
                if len(caption_words & row_words) > len(caption_words) * 0.5:
                    continue
            
            non_empty = sum(1 for x in row if pd.notna(x) and str(x).strip())
            if non_empty < len(row) * 0.3:
                continue
            
            if re.search(r'Table\s+\d+[\.:]?\s+', row_text, re.IGNORECASE):
                continue
            
            cleaned_rows.append(row)
        
        return pd.DataFrame(cleaned_rows).reset_index(drop=True) if cleaned_rows else pd.DataFrame()

    @staticmethod
    def reconstruct_hierarchical_headers(df: pd.DataFrame, header_rows: int) -> List[str]:
        """Reconstruct column names from multi-level headers"""
        if header_rows <= 1:
            return [str(x) if pd.notna(x) else f"col_{i}" for i, x in enumerate(df.iloc[0])]
        
        header_matrix = [[str(x).strip() if pd.notna(x) and str(x).strip() else "" for x in df.iloc[i]] for i in range(header_rows)]
        num_cols = len(header_matrix[0])
        
        for row_idx in range(len(header_matrix)):
            current_value = ""
            for col_idx in range(num_cols):
                if header_matrix[row_idx][col_idx]:
                    current_value = header_matrix[row_idx][col_idx]
                else:
                    header_matrix[row_idx][col_idx] = current_value
        
        final_headers = []
        for col_idx in range(num_cols):
            parts = [header_matrix[row_idx][col_idx] for row_idx in range(len(header_matrix))]
            unique_parts = []
            for part in parts:
                if part and part not in unique_parts:
                    unique_parts.append(part)
            
            final_headers.append("_".join(unique_parts) if unique_parts else f"col_{col_idx}")
        
        return final_headers

    @staticmethod
    def process_table(df: pd.DataFrame, caption: str = "") -> Optional[Dict]:
        """Complete table processing pipeline"""
        df = TableStructureReconstructor.clean_noise_rows(df, caption)
        if df.empty or len(df) < 2:
            return None
        
        header_rows = TableStructureReconstructor.detect_header_rows(df)
        column_names = TableStructureReconstructor.reconstruct_hierarchical_headers(df, header_rows)
        
        data_df = df.iloc[header_rows:].reset_index(drop=True)
        data_df.columns = column_names[:len(data_df.columns)]
        data_df = data_df.fillna("").astype(str)
        data_df = data_df[data_df.apply(lambda x: x.str.strip().str.len().sum(), axis=1) > 0]
        
        if data_df.empty:
            return None
        
        return {
            "headers": column_names, "rows": data_df.values.tolist(),
            "data": data_df.to_dict(orient='records'),
            "shape": {"rows": len(data_df), "columns": len(data_df.columns)},
            "metadata": {"header_rows_detected": header_rows, "has_hierarchical_headers": header_rows > 1}
        }


class PDFUtils:
    _layout_model = None

    @staticmethod
    def initialize_layoutparser_model():
        """Pre-load model and handle initialization errors gracefully."""
        global LAYOUTPARSER_AVAILABLE
        if not LAYOUTPARSER_AVAILABLE or PDFUtils._layout_model:
            return
        try:
            logger.info(" Initializing LayoutParser model...")
            # This model requires Detectron2
            PDFUtils._layout_model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            logger.info("✓ LayoutParser model loaded successfully")
        except Exception as e:
            logger.warning(f"LayoutParser model initialization failed: {e}")
            logger.warning("This is likely because 'Detectron2' is not installed. Please install it.")
            LAYOUTPARSER_AVAILABLE = False

    @staticmethod
    def detect_layout_regions(page: fitz.Page) -> Optional[Dict[str, List]]:
        """Use LayoutParser to detect table and figure regions on a page"""
        if not LAYOUTPARSER_AVAILABLE:
            return None
        
        if not PDFUtils._layout_model:
            PDFUtils.initialize_layoutparser_model()
            if not PDFUtils._layout_model:
                return None

        try:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            layout = PDFUtils._layout_model.detect(img)
            
            regions = {'tables': [], 'figures': [], 'text': []}
            for block in layout:
                bbox = [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2]
                if block.type == "Table": regions['tables'].append(bbox)
                elif block.type == "Figure": regions['figures'].append(bbox)
                elif block.type in ["Text", "Title", "List"]: regions['text'].append(bbox)
            return regions
        except Exception as e:
            logger.warning(f"LayoutParser detection error on page {page.number + 1}: {e}")
            return None

    @staticmethod
    def _bbox_overlaps(bbox1: List[float], bbox2: List[float], threshold: float = 0.5) -> bool:
        """Check if two bboxes overlap significantly"""
        if not all([bbox1, bbox2, len(bbox1) >= 4, len(bbox2) >= 4]):
            return False
        
        x1_min, y1_min, x1_max, y1_max = bbox1[:4]
        x2_min, y2_min, x2_max, y2_max = bbox2[:4]
        
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold

    @staticmethod
    def _bbox_overlaps_any(bbox: List[float], regions: List[List[float]], threshold: float = 0.3) -> bool:
        """Check if bbox overlaps with any region in list"""
        return any(PDFUtils._bbox_overlaps(bbox, region, threshold) for region in regions)

    @staticmethod
    def extract_text_with_layout(page: fitz.Page, page_num: int, min_length: int, 
                                 layout_blocks: Dict) -> List[Dict]:
        """Extract text blocks, avoiding table/figure regions"""
        text_objects = []
        avoid_regions = layout_blocks.get('tables', []) + layout_blocks.get('figures', [])
        
        logger.debug(f"Extracting text with layout awareness "
                    f"(avoiding {len(avoid_regions)} table/figure regions)...")
        
        extracted_count = 0
        skipped_count = 0
        
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    bbox = block.get("bbox", [])
                    
                    # Skip if overlaps with table/figure
                    if PDFUtils._bbox_overlaps_any(bbox, avoid_regions):
                        skipped_count += 1
                        continue
                    
                    text = " ".join(
                        span.get("text", "") 
                        for line in block.get("lines", []) 
                        for span in line.get("spans", [])
                    ).strip()
                    
                    if len(text) >= min_length:
                        text_objects.append({
                            "type": "text_chunk", 
                            "text": text, 
                            "page": page_num + 1,
                            "bbox": list(bbox), 
                            "detected_by": "layoutparser"
                        })
                        extracted_count += 1
        except Exception as e:
            logger.warning(f" Failed to extract text with layout on page {page_num+1}: {e}")
        
        logger.debug(f"  ✓ Extracted {extracted_count} text blocks, skipped {skipped_count} overlapping regions")
        return text_objects

    @staticmethod
    def extract_table_from_bbox(page: fitz.Page, page_num: int, pdf_path: str, bbox: List[float]) -> List[Dict]:
        """Extract table from specific bbox detected by LayoutParser"""
        logger.debug(f"    Extracting table from detected bbox...")
        try:
            tables = PDFUtils.extract_tables(page, page_num, pdf_path=pdf_path, use_layoutparser=False)
            result = []
            
            for table in tables:
                if PDFUtils._bbox_overlaps(table.get('bbox', []), bbox):
                    table['detected_by'] = 'layoutparser'
                    table['detection_bbox'] = bbox
                    result.append(table)
                    logger.debug(f" Matched table in detected region")
            
            if not result:
                logger.debug(f" No tables matched the detected bbox")
            
            return result
        except Exception as e:
            logger.warning(f"    Failed to extract table from bbox: {e}")
            return []

    @staticmethod
    def extract_figure_from_bbox(page: fitz.Page, page_num: int, bbox: List[float]) -> List[Dict]:
        """Extract figure image from specific bbox detected by LayoutParser"""
        logger.debug(f"    Extracting figure from detected bbox...")
        try:
            x0, y0, x1, y1 = bbox
            clip_rect = fitz.Rect(
                max(0, x0 - 10), 
                max(0, y0 - 10), 
                min(page.rect.width, x1 + 10), 
                min(page.rect.height, y1 + 10)
            )
            pix = page.get_pixmap(clip=clip_rect, dpi=200)
            
            figure = {
                "type": "figure", 
                "image_data": pix.tobytes(), 
                "page": page_num + 1,
                "width": pix.width, 
                "height": pix.height, 
                "bbox": bbox,
                "figure_id": f"page{page_num+1}_fig_lp", 
                "detected_by": "layoutparser"
            }
            
            logger.debug(f"      ✓ Extracted figure: {pix.width}x{pix.height}px from LayoutParser region")
            return [figure]
            
        except Exception as e:
            logger.warning(f"    Failed to extract figure from bbox: {e}")
            return []
        
    @staticmethod
    def extract_images(page: fitz.Page, page_num: int, min_pixels: int = 10000) -> List[Dict]:
        """Extract images with size filtering and logging"""
        image_objects = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                width, height = base_image.get("width", 0), base_image.get("height", 0)
                if width * height >= min_pixels:
                    image_objects.append({"type": "figure", "image_data": base_image["image"], "page": page_num + 1, "width": width, "height": height, "figure_id": f"page{page_num+1}_fig{img_idx+1}"})
            except Exception as e:
                logger.warning(f"Failed to extract image {img_idx+1}: {e}")
        return image_objects

    @staticmethod
    def _extract_tables_advanced(page: fitz.Page, page_num: int, pdf_path: str) -> List[Dict]:
        """Advanced table extraction using Camelot and pdfplumber"""
        table_objects = []
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')
            if not tables or tables[0].accuracy < 60:
                tables.extend(camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream', edge_tol=100))
            for idx, table in enumerate(tables):
                if table.accuracy > 50:
                    processed = TableStructureReconstructor.process_table(table.df)
                    if processed:
                        table_objects.append({"type": "table", "page": page_num + 1, "bbox": list(table._bbox) if hasattr(table, '_bbox') else [], "table_id": f"table_{page_num+1}_{idx+1}", **processed, "extraction_method": f"camelot_{table.flavor}", "accuracy": table.accuracy})
        except Exception as e:
            if "Ghostscript" not in str(e): # Avoid redundant logging if Ghostscript is the known issue
                logger.warning(f"Camelot extraction failed on page {page_num+1}: {e}")

        if not table_objects:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    plumber_tables = pdf.pages[page_num].extract_tables()
                    for idx, raw_table in enumerate(plumber_tables):
                        if raw_table and len(raw_table) > 1:
                            processed = TableStructureReconstructor.process_table(pd.DataFrame(raw_table))
                            if processed:
                                table_objects.append({"type": "table", "page": page_num + 1, "bbox": [], "table_id": f"table_{page_num+1}_{idx+1}", **processed, "extraction_method": "pdfplumber"})
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed on page {page_num+1}: {e}")
        
        return table_objects
    
    # ----- Public method ----- #
    @staticmethod
    def extract_text_blocks(page: fitz.Page, page_num: int, min_length: int = 50) -> List[Dict]:
        """Extract text blocks with position info and logging"""
        text_objects = []
        logger.debug(f"Extracting text blocks from page {page_num + 1}...")
        
        try:
            blocks = page.get_text("dict")["blocks"]
            text_count = 0
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    text_lines = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_lines.append(span.get("text", ""))
                    text = " ".join(text_lines).strip()
                    
                    if len(text) >= min_length:
                        text_objects.append({
                            "type": "text_chunk",
                            "text": text,
                            "page": page_num + 1,
                            "bbox": list(block.get("bbox", []))
                        })
                        text_count += 1
            
            logger.debug(f"  ✓ Extracted {text_count} text blocks (≥{min_length} chars)")
            
        except Exception as e:
            logger.error(f" Failed to extract text on page {page_num+1}: {e}")
        
        return text_objects

    @staticmethod
    def extract_images(page: fitz.Page, page_num: int, min_pixels: int = 10000) -> List[Dict]:
        """Extract images with size filtering and logging"""
        image_objects = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                width, height = base_image.get("width", 0), base_image.get("height", 0)
                if width * height >= min_pixels:
                    image_objects.append({"type": "figure", "image_data": base_image["image"], "page": page_num + 1, "width": width, "height": height, "figure_id": f"page{page_num+1}_fig{img_idx+1}"})
            except Exception as e:
                logger.warning(f"Failed to extract image {img_idx+1}: {e}")
        return image_objects

    @staticmethod
    def extract_tables(page: fitz.Page, page_num: int, pdf_path: Optional[str] = None, 
                       use_layoutparser: bool = True) -> List[Dict]:
        """Extract tables with corrected logic and fallbacks."""
        table_objects = []
        logger.debug(f"Extracting tables from page {page_num + 1}...")

        # Method 1: PyMuPDF built-in (Corrected)
        logger.debug("  Trying PyMuPDF table extraction...")
        try:
            # The find_tables() result is an iterable TableFinder object, not a list.
            # We iterate over it. If no tables are found, the loop is just empty.
            tables_found = list(page.find_tables()) # Convert to list to get count
            logger.debug(f"  Found {len(tables_found)} potential tables with PyMuPDF")

            for idx, t in enumerate(tables_found):
                try:
                    df = t.to_pandas()
                    if not df.empty:
                        processed = TableStructureReconstructor.process_table(df)
                        if processed:
                            table_objects.append({
                                "type": "table", "page": page_num + 1, "bbox": list(t.bbox),
                                "table_id": f"table_{page_num+1}_{idx+1}_pymupdf", 
                                **processed, "extraction_method": "pymupdf",
                                "has_hierarchical_headers": processed["metadata"]["has_hierarchical_headers"]
                            })
                except Exception as e:
                    logger.warning(f"  Failed to process PyMuPDF table {idx+1}: {e}")
                    
        except Exception as e:
            logger.warning(f"  PyMuPDF table extraction failed on page {page_num+1}: {e}")

        # Method 2: Fallback to advanced extraction (Camelot + pdfplumber)
        if not table_objects and pdf_path:
            logger.debug("  No tables found via PyMuPDF, trying advanced extraction (Camelot/pdfplumber)...")
            table_objects.extend(PDFUtils._extract_tables_advanced(page, page_num, pdf_path))
        
        if table_objects:
            logger.info(f"  ✓ Extracted {len(table_objects)} tables from page {page_num+1}")
        else:
            logger.debug(f"  No tables found on page {page_num+1}")
        
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