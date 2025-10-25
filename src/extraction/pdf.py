"""
src/extraction/pdf.py
High-performance PDF extraction using asyncio for handling multiple files
and multiprocessing for parallel page processing.
"""
from PIL import Image
import fitz
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Import các tiện ích đã được nâng cấp
from .utils import PDFUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _process_page_worker(args: tuple) -> List[Dict]:
    """Extract text and page image for VLM processing"""
    pdf_path, page_num, min_text_length, extract_images_flag, extract_tables_flag = args
    extracted_data = []
    
    try:
        # 1. Extract full text using the new robust method
        full_text = PDFUtils.extract_all_text_optimized(pdf_path, page_num, min_text_length)
        
        # We still need a fitz object for the image
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)

        if full_text:
            extracted_data.append({
                'type': 'text_chunk',
                'text': full_text,
                'page': page_num + 1,
                'bbox': list(page.rect)
            })
        
        # 2. Convert page to image for VLM
        pix = page.get_pixmap(dpi=200)
        from PIL import Image
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        extracted_data.append({
            'type': 'page_image_for_vlm',
            'page': page_num + 1,
            'image': page_image,
            'width': pix.width,
            'height': pix.height
        })
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
    
    return extracted_data

class PDFExtractor:
    def __init__(
        self, 
        min_text_length: int = 50, 
        extract_images: bool = True, 
        extract_tables: bool = True,
        max_workers: Optional[int] = None,
        output_dir: Optional[str] = None,
        use_vlm: bool = True  # NEW: replace use_layoutparser
    ):
        self.min_text_length = min_text_length
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_vlm = use_vlm
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFExtractor initialized (VLM: {use_vlm})")

    def extract_parallel(self, pdf_path: str) -> List[Dict]:
        """Processes a single PDF in parallel using a pool of processes."""
        logger.info(f"Starting parallel extraction for: {pdf_path}")
        if self.use_vlm:
            logger.info("  🔍 Using VLM for layout detection")

        try:
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            
            # Bỏ vlm_config khỏi tasks
            tasks = [
                (pdf_path, i, self.min_text_length, self.extract_images, self.extract_tables) 
                for i in range(num_pages)
            ]
            
            results = []
            for page_results in self.executor.map(_process_page_worker, tasks):
                results.extend(page_results)
            
            logger.info(f"Completed parallel extraction for: {pdf_path}")
            logger.info(f"  → Extracted {len(results)} elements")
            
            if self.output_dir:
                self._save_extraction_results(pdf_path, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path}: {e}")
            return []

    async def process_files_async(self, pdf_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Processes a list of PDF files asynchronously.
        Each file's processing is offloaded to the process pool executor
        to avoid blocking the asyncio event loop.
        
        Returns:
            Dictionary mapping pdf_path -> extracted_data
        """
        loop = asyncio.get_running_loop()
        
        # Create a list of asyncio Tasks.
        # loop.run_in_executor runs our synchronous, CPU-bound function (extract_parallel)
        # in a separate process pool without blocking the event loop.
        tasks = [
            loop.run_in_executor(self.executor, self.extract_parallel, pdf_path)
            for pdf_path in pdf_paths
        ]
        
        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks)
        
        # Combine results into a dictionary
        final_results = {
            pdf_paths[i]: results_list[i] 
            for i in range(len(pdf_paths))
        }
        
        # Generate summary report
        self._print_summary(final_results)
        
        return final_results

    def _save_extraction_results(self, pdf_path: str, results: List[Dict]):
        """Save extraction results to JSON file"""
        try:
            pdf_name = Path(pdf_path).stem
            output_file = self.output_dir / f"{pdf_name}_extracted.json"
            
            # Prepare serializable data (remove binary image data)
            serializable_results = []
            for item in results:
                item_copy = item.copy()
                if item_copy.get('type') == 'figure' and 'image_data' in item_copy:
                    # Remove binary data, keep metadata
                    item_copy['image_data'] = '<binary_data_removed>'
                serializable_results.append(item_copy)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'pdf_file': pdf_path,
                    'total_elements': len(results),
                    'elements': serializable_results
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  → Saved results to: {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save results for {pdf_path}: {e}")

    def _print_summary(self, results: Dict[str, List[Dict]]):
        """Print extraction summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 EXTRACTION SUMMARY")
        logger.info("="*60)
        
        for pdf_path, data in results.items():
            text_count = sum(1 for item in data if item.get('type') == 'text_chunk')
            table_count = sum(1 for item in data if item.get('type') == 'table')
            image_count = sum(1 for item in data if item.get('type') == 'figure')
            
            # Count hierarchical tables
            hierarchical_tables = sum(
                1 for item in data 
                if item.get('type') == 'table' and item.get('has_hierarchical_headers')
            )
            
            logger.info(f"\n📄 {Path(pdf_path).name}")
            logger.info(f"  ├─ Text blocks: {text_count}")
            logger.info(f"  ├─ Tables: {table_count}")
            if hierarchical_tables > 0:
                logger.info(f"  │  └─ Multi-level headers: {hierarchical_tables}")
            logger.info(f"  └─ Images: {image_count}")
        
        logger.info("\n" + "="*60 + "\n")

    def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()