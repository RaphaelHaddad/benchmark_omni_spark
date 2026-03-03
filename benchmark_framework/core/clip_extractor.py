"""Sliding window clip extractor for video benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import uuid

from benchmark_framework.preprocessing.video_processor import FFmpegProcessor, VideoInfo


@dataclass
class VideoClip:
    """A single clip extracted from a video."""
    clip_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    frames: List[Path] = field(default_factory=list)
    audio: Path = field(default=None)
    enable_audio: bool = True
    enable_video: bool = True

    def to_dict(self) -> dict:
        """Convert clip to dictionary."""
        return {
            "clip_id": self.clip_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "frame_count": len(self.frames),
            "audio_path": str(self.audio) if self.audio else None,
            "enable_audio": self.enable_audio,
            "enable_video": self.enable_video
        }


class ClipExtractor:
    """Extract clips from video using sliding window approach."""

    def __init__(
        self,
        clip_duration: float = 5.0,
        frames_per_clip: int = 5,
        overlap: float = 0.0,
        temp_dir: Path = None
    ):
        """
        Initialize clip extractor.

        Args:
            clip_duration: Duration of each clip in seconds
            frames_per_clip: Number of frames to extract per clip
            overlap: Overlap between consecutive clips in seconds
            temp_dir: Temporary directory for extracted content
        """
        self.clip_duration = clip_duration
        self.frames_per_clip = frames_per_clip
        self.overlap = overlap
        self.processor = FFmpegProcessor(temp_dir=temp_dir)

    def extract_clips(
        self,
        video_path: Path,
        audio_sample_rate: int = 16000,
        audio_channels: int = 1,
        enable_audio: bool = True,
        enable_video: bool = True
    ) -> List[VideoClip]:
        """
        Extract all clips from video using sliding window.

        Args:
            video_path: Path to video file
            audio_sample_rate: Audio sample rate in Hz
            audio_channels: Number of audio channels
            enable_audio: Enable audio extraction
            enable_video: Enable video (frames) extraction

        Returns:
            List of VideoClip objects
        """
        # Validate at least one modality is enabled
        if not enable_audio and not enable_video:
            raise ValueError("At least one modality (audio or video) must be enabled")

        # Get video info
        video_info = self.processor.get_video_info(video_path)
        duration = video_info.duration

        clips = []
        step = self.clip_duration - self.overlap

        # Extract clips
        start_time = 0.0
        clip_index = 0

        while start_time < duration:
            end_time = min(start_time + self.clip_duration, duration)
            actual_duration = end_time - start_time

            # Skip if clip is too short (less than 1 second)
            if actual_duration < 1.0:
                break

            # Extract clip with frames and/or audio based on modality flags
            frames = []
            audio = None

            # First extract the clip (needed for both frame and audio extraction)
            clip_path = self.processor.extract_clip(video_path, start_time, actual_duration)

            # Extract frames if video enabled
            if enable_video:
                num_frames = self.frames_per_clip if self.frames_per_clip > 0 else 1
                frames = self.processor.extract_frames(clip_path, num_frames)

            # Extract audio if audio enabled
            if enable_audio:
                audio = self.processor.extract_audio(
                    clip_path,
                    sample_rate=audio_sample_rate,
                    channels=audio_channels
                )

            # Clean up temporary clip
            clip_path.unlink(missing_ok=True)

            clip = VideoClip(
                start_time=start_time,
                end_time=end_time,
                duration=actual_duration,
                frames=frames,
                audio=audio,
                enable_audio=enable_audio,
                enable_video=enable_video
            )
            clips.append(clip)

            # Move to next clip
            start_time += step
            clip_index += 1

        return clips

    def cleanup(self):
        """Clean up temporary files."""
        self.processor.cleanup()
