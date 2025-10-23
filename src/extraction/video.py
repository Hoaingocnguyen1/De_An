"""
src/extraction/video.py
A high-performance, parallel video transcription orchestrator.
Processes multiple local files or YouTube URLs concurrently.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union
from pathlib import Path
import logging
import time

from .utils import VideoUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoExtractor:
    def __init__(self, whisper_model: str = "base", max_workers: int = 4):
        """
        Khởi tạo bộ điều phối.

        """
        self.model_name = whisper_model
        self.max_workers = max_workers
        # Tải trước mô hình để nó sẵn sàng cho các luồng (thread) worker
        VideoUtils.load_model(self.model_name)

    def _process_single_source(self, source: str, temp_dir: Path) -> List[Dict]:
        """
        Quy trình xử lý hoàn chỉnh cho một nguồn duy nhất (URL hoặc tệp cục bộ).
        Đây là đơn vị công việc sẽ được thực thi song song.
        """
        audio_path = None
        # Phân biệt giữa URL và tệp cục bộ
        if source.startswith("http://") or source.startswith("https://"):
            audio_path = VideoUtils.download_youtube_audio(source, temp_dir)
            source_info = {"source_url": source}
        else:
            if Path(source).exists():
                audio_path = VideoUtils.extract_audio_from_file(source)
                source_info = {"source_file": source}
            else:
                logger.error(f"Source file not found: {source}")
                return []

        if not audio_path:
            return []

        # Chuyển đổi âm thanh thành văn bản
        segments = VideoUtils.transcribe_audio(audio_path, self.model_name)
        
        # Dọn dẹp tệp âm thanh tạm
        try:
            Path(audio_path).unlink()
        except OSError as e:
            logger.warning(f"Could not delete temp audio file {audio_path}: {e}")

        # Định dạng kết quả cuối cùng
        return [
            {"type": "text_chunk", "text": seg["text"].strip(), **seg, **source_info}
            for seg in segments
        ]

    async def extract_from_sources(self, sources: List[str], 
                                  temp_dir_str: str = "./temp") -> Dict[str, List[Dict]]:
        """
        Xử lý một danh sách các nguồn video (URL hoặc tệp) song song.
        
        Args:
            sources: Một danh sách các chuỗi, mỗi chuỗi là một URL YouTube hoặc đường dẫn tệp.
            temp_dir_str: Thư mục để lưu trữ các tệp âm thanh tạm thời.

        Returns:
            Một dictionary ánh xạ mỗi nguồn đầu vào với danh sách các đoạn transcript của nó.
        """
        start_time = time.time()
        temp_dir = Path(temp_dir_str)
        temp_dir.mkdir(exist_ok=True, parents=True)

        loop = asyncio.get_running_loop()
        # Sử dụng ThreadPoolExecutor để chạy các tác vụ blocking (I/O, CPU) mà không chặn event loop
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tạo một tác vụ (task) cho mỗi nguồn
            tasks = [
                loop.run_in_executor(executor, self._process_single_source, source, temp_dir)
                for source in sources
            ]
            # Chờ tất cả các tác vụ hoàn thành
            results = await asyncio.gather(*tasks)

        # Ánh xạ kết quả trở lại với nguồn ban đầu
        final_output = {source: result for source, result in zip(sources, results)}
        
        duration = time.time() - start_time
        logger.info(f"Processed {len(sources)} sources in {duration:.2f} seconds.")
        return final_output


# # --- VÍ DỤ SỬ DỤNG ---
# async def main():
#     # Danh sách các nguồn video cần xử lý
#     video_sources = [
#         "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley
#         "https://www.youtube.com/watch?v=yPYZpwSpKmA",  # Google I/O Keynote
#         "path/to/your/local_video1.mp4",  # Thay bằng đường dẫn tệp của bạn
#         "path/to/your/local_video2.mov",  # Thay bằng đường dẫn tệp của bạn
#     ]
    
#     # Lọc ra các tệp không tồn tại để tránh lỗi
#     valid_sources = [
#         src for src in video_sources 
#         if src.startswith("http") or Path(src).exists()
#     ]
#     if len(valid_sources) != len(video_sources):
#         logger.warning("Some local video files were not found and will be skipped.")

#     # Khởi tạo và chạy bộ điều phối
#     # Tăng max_workers nếu bạn có nhiều CPU core và băng thông mạng tốt
#     extractor = VideoExtractor(whisper_model="base", max_workers=4)
#     transcripts = await extractor.extract_from_sources(valid_sources)

#     # In kết quả
#     for source, segments in transcripts.items():
#         print("\n" + "="*50)
#         print(f"SOURCE: {source}")
#         print(f"SEGMENTS FOUND: {len(segments)}")
#         if segments:
#             # In 5 giây đầu tiên của transcript
#             first_segment_text = segments[0]['text']
#             print(f"TRANSCRIPT (start): '{first_segment_text[:100]}...'")
#         print("="*50)


# if __name__ == "__main__":
#     # Để chạy mã async, chúng ta cần `asyncio.run()`
#     asyncio.run(main())