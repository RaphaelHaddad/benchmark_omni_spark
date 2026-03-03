"""Main benchmark orchestrator for Qwen3-Omni video analysis."""

import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from benchmark_framework.config.settings import BenchmarkConfig, VideoConfig
from benchmark_framework.api.client import QwenOmniClient, InferenceResult
from benchmark_framework.core.clip_extractor import ClipExtractor, VideoClip
from benchmark_framework.core.metrics import ClipMetrics, MetricsCalculator, BenchmarkSummary
from benchmark_framework.preprocessing.video_processor import FFmpegProcessor
from benchmark_framework.storage.results import (
    BenchmarkResult, ClipResult, BenchmarkConfig as StorageConfig
)
from benchmark_framework.storage.exporter import ResultsExporter


class VideoBenchmark:
    """Orchestrator for video benchmarking."""

    def __init__(
        self,
        config: BenchmarkConfig,
        video_config: VideoConfig
    ):
        """
        Initialize video benchmark.

        Args:
            config: Benchmark configuration
            video_config: Video processing configuration
        """
        self.config = config
        self.video_config = video_config

        # Store modality flags for convenience
        self.enable_audio = config.enable_audio
        self.enable_video = config.enable_video

        # Initialize components
        self.api_client = QwenOmniClient(
            api_url=config.api_url,
            api_key=config.api_key,
            model_name=config.model_name
        )
        self.extractor = ClipExtractor(
            clip_duration=config.clip_duration,
            frames_per_clip=config.frames_per_clip,
            temp_dir=video_config.temp_dir
        )
        self.video_processor = FFmpegProcessor(temp_dir=video_config.temp_dir)
        self.exporter = ResultsExporter()

    def run_benchmark(
        self,
        video_path: Path,
        prompt: str = "Please analyze this video content. Describe what you observe in both the visual and audio elements. What is happening in this clip?"
    ) -> BenchmarkResult:
        """
        Run complete benchmark on a video.

        Args:
            video_path: Path to video file
            prompt: Prompt to use for inference

        Returns:
            BenchmarkResult with all results
        """
        run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()

        # Get video info
        video_info = self.video_processor.get_video_info(video_path)

        # Extract clips
        print(f"Extracting clips from video (duration: {video_info.duration:.1f}s)...")
        clips = self.extractor.extract_clips(
            video_path,
            audio_sample_rate=self.video_config.audio_sample_rate,
            audio_channels=self.video_config.audio_channels,
            enable_audio=self.enable_audio,
            enable_video=self.enable_video
        )
        print(f"Extracted {len(clips)} clips")

        # Process each clip
        clip_results = []
        for i, clip in enumerate(clips):
            print(f"Processing clip {i+1}/{len(clips)} ({clip.start_time:.1f}s - {clip.end_time:.1f}s)...")

            clip_result = self.process_clip(clip, i, prompt)
            clip_results.append(clip_result)

            # Print quick stats
            if clip_result.metrics.error is None:
                print(f"  TTFT: {clip_result.metrics.time_to_first_token:.3f}s, "
                      f"TTLT: {clip_result.metrics.time_to_last_token:.3f}s, "
                      f"Tokens/sec: {clip_result.metrics.tokens_per_second:.1f}")
            else:
                print(f"  Error: {clip_result.metrics.error}")

        # Calculate summary
        summary = MetricsCalculator.calculate_summary(
            [cr.metrics for cr in clip_results]
        )

        total_time = time.time() - start_time
        summary.total_time = total_time

        # Create benchmark result
        storage_config = StorageConfig(
            clip_duration=self.config.clip_duration,
            frames_per_clip=self.config.frames_per_clip,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            api_url=self.config.api_url,
            model_name=self.config.model_name,
            audio_sample_rate=self.video_config.audio_sample_rate,
            audio_channels=self.video_config.audio_channels,
            enable_audio=self.config.enable_audio,
            enable_video=self.config.enable_video
        )

        result = BenchmarkResult(
            run_id=run_id,
            timestamp=datetime.now(),
            video_path=str(video_path),
            video_info=video_info.to_dict(),
            config=storage_config,
            clip_results=clip_results,
            summary=summary
        )

        # Cleanup
        self.extractor.cleanup()

        return result

    def process_clip(self, clip: VideoClip, index: int, prompt: str) -> ClipResult:
        """
        Process a single clip.

        Args:
            clip: VideoClip to process
            index: Clip index
            prompt: Prompt for inference

        Returns:
            ClipResult with metrics
        """
        metrics = ClipMetrics()

        # Preprocess (load frames and audio)
        preprocess_start = time.time()

        # Load frames as bytes
        frames_bytes = []
        for frame_path in clip.frames:
            with open(frame_path, 'rb') as f:
                frames_bytes.append(f.read())

        # Load audio as bytes (if available)
        audio_bytes = b""
        if clip.audio is not None:
            with open(clip.audio, 'rb') as f:
                audio_bytes = f.read()

        metrics.preprocessing_time = time.time() - preprocess_start

        # Run inference
        try:
            result: InferenceResult = self.api_client.infer_sync(
                frames=frames_bytes if self.enable_video else [],
                audio=audio_bytes if self.enable_audio else b"",
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                enable_audio=self.enable_audio,
                enable_video=self.enable_video
            )

            # Update metrics from API result
            metrics.time_to_first_token = result.metrics.time_to_first_token
            metrics.time_to_last_token = result.metrics.time_to_last_token
            metrics.tokens_per_second = result.metrics.tokens_per_second
            metrics.total_tokens = result.metrics.total_tokens
            metrics.total_inference_time = result.metrics.total_inference_time
            metrics.response_length = len(result.text)

            return ClipResult(
                clip_index=index,
                start_time=clip.start_time,
                end_time=clip.end_time,
                duration=clip.duration,
                metrics=metrics,
                response=result.text,
                frame_count=len(clip.frames),
                audio_path=str(clip.audio)
            )

        except Exception as e:
            metrics.error = str(e)
            return ClipResult(
                clip_index=index,
                start_time=clip.start_time,
                end_time=clip.end_time,
                duration=clip.duration,
                metrics=metrics,
                frame_count=len(clip.frames),
                audio_path=str(clip.audio)
            )

    def save_results(
        self,
        result: BenchmarkResult,
        output_dir: Path,
        formats: List[str] = None
    ) -> List[Path]:
        """
        Save benchmark results.

        Args:
            result: Benchmark result to save
            output_dir: Output directory
            formats: List of formats ("json", "csv", "markdown")

        Returns:
            List of saved file paths
        """
        if formats is None:
            formats = ["json", "csv", "markdown"]

        output_dir = Path(output_dir)
        saved_files = []

        # JSON
        if "json" in formats:
            json_path = output_dir / "raw" / f"{result.run_id}.json"
            self.exporter.export_json(result, json_path)
            saved_files.append(json_path)

        # CSV
        if "csv" in formats:
            csv_summary_path = output_dir / "processed" / "summary.csv"
            csv_clips_path = output_dir / "processed" / "clips.csv"

            # Append to summary or create new
            if csv_summary_path.exists():
                # For simplicity, just overwrite for now
                pass

            self.exporter.export_csv_summary([result], csv_summary_path)
            self.exporter.export_csv_clips(result, csv_clips_path)
            saved_files.extend([csv_summary_path, csv_clips_path])

        # Markdown report
        if "markdown" in formats:
            md_path = output_dir / "reports" / f"{result.run_id}.md"
            self.exporter.export_markdown_report(result, md_path)
            saved_files.append(md_path)

        return saved_files
