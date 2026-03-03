"""Configuration management for Qwen3-Omni benchmark framework."""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class BenchmarkConfig(BaseModel):
    """Main benchmark configuration."""

    # Hyperparameters
    clip_duration: int = Field(default=5, description="Clip duration in seconds")
    frames_per_clip: int = Field(default=5, description="Number of frames per clip")

    # Modality selection
    enable_audio: bool = Field(default=True, description="Enable audio modality")
    enable_video: bool = Field(default=True, description="Enable video modality")

    # API settings
    api_url: str = Field(default="http://localhost:8078/v1", description="API base URL")
    api_key: str = Field(default="qwen3-omni-api-key", description="API authentication key")
    model_name: str = Field(default="qwen3-omni", description="Model name")

    # Inference settings
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")

    # Benchmark settings
    output_dir: Path = Field(default=Path("results"), description="Output directory for results")
    num_runs: int = Field(default=1, description="Number of benchmark runs")

    # Metrics to collect
    enable_metrics: List[str] = Field(
        default_factory=lambda: ["time_to_first_token", "time_to_last_token", "tokens_per_second"],
        description="Metrics to collect"
    )

    @field_validator("clip_duration", "frames_per_clip")
    @classmethod
    def validate_positive(cls, v: int, info) -> int:
        # For frames_per_clip, allow 0 when video is disabled
        if info.field_name == "frames_per_clip":
            if v < 0:
                raise ValueError("Must be non-negative")
            return v
        # For clip_duration, must be positive
        if v <= 0:
            raise ValueError("Must be positive")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("enable_audio", "enable_video")
    @classmethod
    def validate_modality(cls, v: bool, info) -> bool:
        """Ensure at least one modality is enabled."""
        # Get all values for validation
        if hasattr(info, "data"):
            data = info.data
            audio = data.get("enable_audio", True)
            video = data.get("enable_video", True)
            # Only validate if both fields are set
            if "enable_audio" in data and "enable_video" in data:
                if not audio and not video:
                    raise ValueError("At least one modality (audio or video) must be enabled")
        return v


class VideoConfig(BaseModel):
    """Video processing configuration."""

    input_path: Path = Field(description="Path to input video file")

    # Output settings
    frame_format: str = Field(default="png", description="Frame image format")
    audio_format: str = Field(default="wav", description="Audio format")

    # Audio settings
    audio_sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    audio_channels: int = Field(default=1, description="Number of audio channels")

    # Temporary directory for extracted content
    temp_dir: Optional[Path] = Field(default=None, description="Temporary directory for extraction")

    @field_validator("audio_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError("Common sample rates are 8000, 16000, 22050, 44100, 48000")
        return v

    @field_validator("audio_channels")
    @classmethod
    def validate_channels(cls, v: int) -> int:
        if v not in [1, 2]:
            raise ValueError("Audio must be mono (1) or stereo (2)")
        return v


class MetricsConfig(BaseModel):
    """Metrics collection configuration."""

    # Metrics to enable/disable
    measure_ttft: bool = Field(default=True, description="Measure time to first token")
    measure_ttlt: bool = Field(default=True, description="Measure time to last token")
    measure_tokens_per_sec: bool = Field(default=True, description="Measure tokens per second")
    measure_preprocessing: bool = Field(default=True, description="Measure preprocessing time")

    # Percentiles to calculate
    percentiles: List[float] = Field(
        default_factory=lambda: [0.5, 0.9, 0.95, 0.99],
        description="Percentiles to calculate"
    )


class StorageConfig(BaseModel):
    """Results storage configuration."""

    output_dir: Path = Field(default=Path("results"), description="Output directory")

    # Format settings
    save_format: str = Field(default="json", description="Save format (json, csv, both)")
    save_raw_responses: bool = Field(default=True, description="Save raw API responses")
    save_clips: bool = Field(default=False, description="Save extracted clips")

    # Compression
    compress_old_results: bool = Field(default=False, description="Compress old results")

    @field_validator("save_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ["json", "csv", "both"]:
            raise ValueError("Format must be 'json', 'csv', or 'both'")
        return v


class Config:
    """Pydantic configuration."""

    arbitrary_types_allowed = True
    extra = "allow"
