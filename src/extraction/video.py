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
        Initialize video transcription orchestrator.
        
        Priority:
        1. YouTube Transcript API (fastest, no download needed)
        2. Download + Whisper (fallback for videos without transcripts)
        """
        self.model_name = whisper_model
        self.max_workers = max_workers
        # Pre-load Whisper model for fallback
        VideoUtils.load_model(self.model_name)

    def _process_single_source(self, source: str, temp_dir: Path) -> List[Dict]:
        """
        Complete processing pipeline for a single source.
        
        For YouTube URLs:
        1. Try to get transcript directly (fast)
        2. If fails, download audio and transcribe (slow but reliable)
        
        For local files:
        - Extract audio and transcribe with Whisper
        """
        # Check if it's a YouTube URL
        if source.startswith("http://") or source.startswith("https://"):
            if "youtube.com" in source or "youtu.be" in source:
                return self._process_youtube(source, temp_dir)
            else:
                logger.warning(f"Non-YouTube URL not supported: {source}")
                return []
        else:
            # Local video file
            return self._process_local_video(source, temp_dir)

    def _process_youtube(self, url: str, temp_dir: Path) -> List[Dict]:
        """
        Process YouTube video with transcript API first, fallback to download.
        """
        logger.info(f"Processing YouTube URL: {url}")
        
        # Method 1: Try direct transcript (fast - no download!)
        segments = VideoUtils.get_youtube_transcript_direct(url)
        
        if segments:
            logger.info("✓ Used YouTube Transcript API (no download needed)")
            # Add source info
            return [
                {
                    "type": "text_chunk",
                    "text": seg["text"].strip(),
                    **seg,
                    "source_url": url,
                    "extraction_method": "youtube_transcript_api"
                }
                for seg in segments
            ]
        
        # Method 2: Fallback to download + transcribe
        logger.info("Transcript not available, falling back to audio download...")
        audio_path = VideoUtils.download_youtube_audio(url, temp_dir)
        
        if not audio_path:
            logger.error(f"Failed to download audio from {url}")
            return []
        
        segments = VideoUtils.transcribe_audio(audio_path, self.model_name)
        
        # Cleanup temp audio
        try:
            Path(audio_path).unlink()
        except OSError as e:
            logger.warning(f"Could not delete temp audio file {audio_path}: {e}")
        
        # Format results
        return [
            {
                "type": "text_chunk",
                "text": seg["text"].strip(),
                **seg,
                "source_url": url,
                "extraction_method": "whisper"
            }
            for seg in segments
        ]

    def _process_local_video(self, video_path: str, temp_dir: Path) -> List[Dict]:
        """
        Process local video file by extracting audio and transcribing.
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f"Processing local video: {video_path}")
        
        # Extract audio
        audio_path = VideoUtils.extract_audio_from_file(video_path)
        
        if not audio_path:
            logger.error(f"Failed to extract audio from {video_path}")
            return []
        
        # Transcribe
        segments = VideoUtils.transcribe_audio(audio_path, self.model_name)
        
        # Cleanup
        try:
            Path(audio_path).unlink()
        except OSError as e:
            logger.warning(f"Could not delete temp audio file {audio_path}: {e}")
        
        # Format results
        return [
            {
                "type": "text_chunk",
                "text": seg["text"].strip(),
                **seg,
                "source_file": video_path,
                "extraction_method": "whisper"
            }
            for seg in segments
        ]

    async def extract_from_sources(
        self, 
        sources: List[str], 
        temp_dir_str: str = "./temp"
    ) -> Dict[str, List[Dict]]:
        start_time = time.time()
        temp_dir = Path(temp_dir_str)
        temp_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Processing {len(sources)} video sources...")

        loop = asyncio.get_running_loop()
        
        # Use ThreadPoolExecutor for I/O and CPU-bound tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(
                    executor, 
                    self._process_single_source, 
                    source, 
                    temp_dir
                )
                for source in sources
            ]
            
            results = await asyncio.gather(*tasks)

        # Map results back to sources
        final_output = {source: result for source, result in zip(sources, results)}
        
        # Print summary
        duration = time.time() - start_time
        successful = sum(1 for result in results if result)
        
        logger.info(f"✓ Processed {successful}/{len(sources)} sources in {duration:.2f}s")
        
        # Show extraction methods used
        for source, segments in final_output.items():
            if segments:
                method = segments[0].get('extraction_method', 'unknown')
                logger.info(f"  - {source[:50]}... : {len(segments)} segments ({method})")
        
        return final_output