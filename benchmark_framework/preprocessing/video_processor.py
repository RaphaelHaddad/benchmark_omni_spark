"""FFmpeg wrapper for video processing."""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import json


class VideoInfo:
    """Information about a video file."""

    def __init__(self, duration: float, fps: float, width: int, height: int, codec: str):
        self.duration = duration
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.resolution = f"{width}x{height}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "codec": self.codec
        }


class FFmpegProcessor:
    """Wrapper for FFmpeg video processing operations."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize FFmpeg processor.

        Args:
            temp_dir: Temporary directory for extracted files
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _run_ffmpeg(self, args: List[str]) -> subprocess.CompletedProcess:
        """
        Run FFmpeg command.

        Args:
            args: FFmpeg command arguments

        Returns:
            Completed process result

        Raises:
            subprocess.CalledProcessError: If FFmpeg fails
        """
        cmd = ["ffmpeg", "-y"] + args  # -y to overwrite output files
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        return result

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """
        Get video information using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo with video metadata
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,r_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        # Extract video stream info
        stream = data.get("streams", [{}])[0]
        format_info = data.get("format", {})

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")

        # Parse FPS
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        duration = float(format_info.get("duration", 0))

        return VideoInfo(duration, fps, width, height, codec)

    def extract_clip(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Extract a clip from video.

        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            duration: Clip duration in seconds
            output_path: Output path (optional, auto-generated if not provided)

        Returns:
            Path to extracted clip
        """
        if output_path is None:
            output_path = self.temp_dir / f"clip_{start_time:.1f}_{duration:.1f}.mp4"

        args = [
            "-ss", str(start_time),      # Start time
            "-i", str(video_path),       # Input file
            "-t", str(duration),         # Duration
            "-c", "copy",                # Copy codec (no re-encoding)
            str(output_path)
        ]

        self._run_ffmpeg(args)
        return output_path

    def extract_frames(
        self,
        video_path: Path,
        num_frames: int,
        output_prefix: Optional[str] = None
    ) -> List[Path]:
        """
        Extract frames uniformly distributed throughout video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            output_prefix: Prefix for output files

        Returns:
            List of paths to extracted frames
        """
        if output_prefix is None:
            output_prefix = str(self.temp_dir / "frame")

        video_info = self.get_video_info(video_path)
        duration = video_info.duration

        # Calculate interval between frames
        if duration > 0 and num_frames > 1:
            interval = duration / (num_frames - 1)
        else:
            interval = 1.0

        # Extract frames at specific timestamps
        frame_paths = []
        for i in range(num_frames):
            timestamp = min(i * interval, duration - 0.1)  # Avoid going beyond video
            output_path = f"{output_prefix}_{i:03d}.png"

            args = [
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",  # High quality
                output_path
            ]

            try:
                self._run_ffmpeg(args)
                frame_paths.append(Path(output_path))
            except subprocess.CalledProcessError:
                # Skip this frame if extraction fails
                continue

        return frame_paths

    def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """
        Extract audio from video.

        Args:
            video_path: Path to video file
            output_path: Output path (optional, auto-generated if not provided)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)

        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = self.temp_dir / "audio.wav"

        args = [
            "-i", str(video_path),
            "-vn",                        # No video
            "-acodec", "pcm_s16le",       # PCM 16-bit little-endian
            "-ar", str(sample_rate),      # Sample rate
            "-ac", str(channels),          # Channels
            str(output_path)
        ]

        self._run_ffmpeg(args)
        return output_path

    def extract_clip_with_audio(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        num_frames: int = 5,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Tuple[List[Path], Path]:
        """
        Extract a clip with both frames and audio.

        This is a convenience method that:
        1. Extracts a clip from the video
        2. Extracts frames from the clip
        3. Extracts audio from the clip

        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            duration: Clip duration in seconds
            num_frames: Number of frames to extract
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels

        Returns:
            Tuple of (list of frame paths, audio path)
        """
        # Extract clip
        clip_path = self.extract_clip(video_path, start_time, duration)

        # Extract frames from clip
        frames = self.extract_frames(clip_path, num_frames)

        # Extract audio from clip
        audio = self.extract_audio(clip_path, sample_rate=sample_rate, channels=channels)

        # Clean up temporary clip
        clip_path.unlink(missing_ok=True)

        return frames, audio

    def cleanup(self):
        """Remove temporary files."""
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                file.unlink(missing_ok=True)
            self.temp_dir.rmdir()
