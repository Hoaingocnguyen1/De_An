"""
src/extraction/pdf.py
High-performance PDF extraction using asyncio for handling multiple files
and multiprocessing for parallel page processing.
"""
import fitz
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Import các tiện ích đã được nâng cấp
from .utils import PDFUtils, LAYOUTPARSER_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# def _process_page_worker(args: tuple) -> List[Dict]:
#     """
#     Worker xử lý một trang PDF.
#     Nó gọi các hàm từ PDFUtils để thực hiện công việc trích xuất.
#     """
#     pdf_path, page_num, min_text_length, extract_images_flag, extract_tables_flag, use_layoutparser = args
#     extracted_data = []
    
#     try:
#         doc = fitz.open(pdf_path)
#         page = doc.load_page(page_num)

#         # Nếu LayoutParser được bật, phát hiện các vùng trước tiên
#         layout_blocks = None
#         if use_layoutparser and LAYOUTPARSER_AVAILABLE:
#             layout_blocks = PDFUtils.detect_layout_regions(page)

#         # 1. Trích xuất dựa trên bố cục nếu có
#         if layout_blocks:
#             # Trích xuất văn bản từ các vùng không phải là bảng/hình
#             extracted_data.extend(
#                 PDFUtils.extract_text_with_layout(page, page_num, min_text_length, layout_blocks)
#             )
#             # Trích xuất bảng từ các vùng đã phát hiện
#             if extract_tables_flag and layout_blocks['tables']:
#                 for table_bbox in layout_blocks['tables']:
#                     extracted_data.extend(
#                         PDFUtils.extract_table_from_bbox(page, page_num, pdf_path, table_bbox)
#                     )
#             # Trích xuất hình ảnh từ các vùng đã phát hiện
#             if extract_images_flag and layout_blocks['figures']:
#                  for fig_bbox in layout_blocks['figures']:
#                     extracted_data.extend(
#                         PDFUtils.extract_figure_from_bbox(page, page_num, fig_bbox)
#                     )
        
#         # 2. Trích xuất thông thường nếu không dùng LayoutParser
#         else:
#             extracted_data.extend(
#                 PDFUtils.extract_text_blocks(page, page_num, min_text_length)
#             )
#             if extract_tables_flag:
#                 extracted_data.extend(
#                     PDFUtils.extract_tables(page, page_num, pdf_path=pdf_path, use_layoutparser=False)
#                 )
#             if extract_images_flag:
#                 extracted_data.extend(
#                     PDFUtils.extract_images(page, page_num)
#                 )
            
#         doc.close()
        
#     except Exception as e:
#         logger.error(f"Error processing page {page_num} of {pdf_path}: {e}")
        
#     return extracted_data
def _process_page_worker(args: tuple) -> List[Dict]:
    """
    SIMPLIFIED: VLM-based processing (no LayoutParser)
    """
    pdf_path, page_num, min_text_length, extract_images_flag, extract_tables_flag, vlm_config = args
    extracted_data = []
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # 1. Extract full text (always)
        full_text = PDFUtils.extract_all_text_optimized(page, min_text_length)
        if full_text:
            extracted_data.append({
                'type': 'text_chunk',
                'text': full_text,
                'page': page_num + 1,
                'bbox': list(page.rect)
            })
        
        # 2. VLM layout detection (async, handled separately)
        # This is now done in async pipeline
        
        # 3. Extract images
        if extract_images_flag:
            extracted_data.extend(
                PDFUtils.extract_images_improved(page, page_num)
            )
        
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
        """
        Processes a single PDF in parallel using a pool of processes.
        """
        logger.info(f"Starting parallel extraction for: {pdf_path}")
        if self.use_vlm:
            logger.info("  🔍 Using VLM for layout detection")

        try:
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            
            tasks = [
                (pdf_path, i, self.min_text_length, self.extract_images, 
                 self.extract_tables, self.use_vlm) 
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


# --- Entrypoint Example ---
# async def main():
#     # List of PDF files to process
#     # note: Create dummy PDF files named 'test1.pdf', 'test2.pdf' for this to run.
#     pdf_files = ["test1.pdf", "test2.pdf"] 
    
#     extractor = AdvancedPDFExtractor(
#         min_text_length=50,
#         extract_images=True,
#         extract_tables=True
#     )
    
#     # Run the asynchronous processing
#     all_data = await extractor.process_files_async(pdf_files)
    
#     # Print summary
#     for pdf_path, data in all_data.items():
#         print(f"\n--- Results for {pdf_path} ---")
#         text_count = sum(1 for item in data if item['type'] == 'text_chunk')
#         table_count = sum(1 for item in data if item['type']  == 'table')
#         image_count = sum(1 for item in data if item['type'] == 'figure')
#         print(f"Found {text_count} text blocks, {table_count} tables, and {image_count} images.")
#         # print(data[0] if data else "No data extracted.")

# if __name__ == "__main__":
#     # To run this, you need to have some PDF files in the same directory.
#     # For example, create two dummy PDFs: test1.pdf and test2.pdf
#     asyncio.run(main())
