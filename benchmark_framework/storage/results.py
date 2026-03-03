"""Results data models for Qwen3-Omni benchmark."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from benchmark_framework.core.metrics import ClipMetrics, BenchmarkSummary


@dataclass
class ClipResult:
    """Result from processing a single clip."""
    clip_index: int
    start_time: float
    end_time: float
    duration: float
    metrics: ClipMetrics
    response: str = ""
    frame_count: int = 0
    audio_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "clip_index": self.clip_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metrics": self.metrics.to_dict(),
            "response": self.response,
            "frame_count": self.frame_count,
            "audio_path": self.audio_path
        }


@dataclass
class BenchmarkConfig:
    """Configuration used for benchmark."""
    clip_duration: int
    frames_per_clip: int
    max_tokens: int
    temperature: float
    api_url: str
    model_name: str
    audio_sample_rate: int
    audio_channels: int
    enable_audio: bool = True
    enable_video: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "clip_duration": self.clip_duration,
            "frames_per_clip": self.frames_per_clip,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_url": self.api_url,
            "model_name": self.model_name,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "enable_audio": self.enable_audio,
            "enable_video": self.enable_video
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    run_id: str
    timestamp: datetime
    video_path: str
    video_info: Dict[str, Any]
    config: BenchmarkConfig
    clip_results: List[ClipResult]
    summary: BenchmarkSummary

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "video_path": self.video_path,
            "video_info": self.video_info,
            "config": self.config.to_dict(),
            "clip_results": [cr.to_dict() for cr in self.clip_results],
            "summary": self.summary.to_dict()
        }
