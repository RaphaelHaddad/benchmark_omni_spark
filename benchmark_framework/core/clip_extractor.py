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

    def to_dict(self) -> dict:
        """Convert clip to dictionary."""
        return {
            "clip_id": self.clip_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "frame_count": len(self.frames),
            "audio_path": str(self.audio) if self.audio else None
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
        audio_channels: int = 1
    ) -> List[VideoClip]:
        """
        Extract all clips from video using sliding window.

        Args:
            video_path: Path to video file
            audio_sample_rate: Audio sample rate in Hz
            audio_channels: Number of audio channels

        Returns:
            List of VideoClip objects
        """
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

            # Extract clip with frames and audio
            frames, audio = self.processor.extract_clip_with_audio(
                video_path=video_path,
                start_time=start_time,
                duration=actual_duration,
                num_frames=self.frames_per_clip,
                sample_rate=audio_sample_rate,
                channels=audio_channels
            )

            clip = VideoClip(
                start_time=start_time,
                end_time=end_time,
                duration=actual_duration,
                frames=frames,
                audio=audio
            )
            clips.append(clip)

            # Move to next clip
            start_time += step
            clip_index += 1

        return clips

    def cleanup(self):
        """Clean up temporary files."""
        self.processor.cleanup()
