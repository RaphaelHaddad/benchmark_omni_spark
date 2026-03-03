"""Metrics collection and calculation for Qwen3-Omni benchmark."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import numpy as np


@dataclass
class ClipMetrics:
    """Metrics collected for a single clip inference."""
    # Timing metrics
    time_to_first_token: float = 0.0      # Request sent → First token received (seconds)
    time_to_last_token: float = 0.0       # Request sent → Last token received (seconds)
    total_inference_time: float = 0.0     # Total inference time (seconds)

    # Token metrics
    total_tokens: int = 0
    tokens_per_second: float = 0.0        # Tokens generated per second

    # Preprocessing metrics
    preprocessing_time: float = 0.0       # Time to extract frames/audio (seconds)

    # Response metrics
    response_length: int = 0              # Length of response text
    error: Optional[str] = None           # Error message if inference failed

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "time_to_first_token": self.time_to_first_token,
            "time_to_last_token": self.time_to_last_token,
            "total_inference_time": self.total_inference_time,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "preprocessing_time": self.preprocessing_time,
            "response_length": self.response_length,
            "error": self.error
        }


@dataclass
class BenchmarkSummary:
    """Summary statistics across all clips."""
    total_clips: int = 0
    successful_clips: int = 0
    failed_clips: int = 0

    # Average metrics
    avg_ttft: float = 0.0
    avg_ttlt: float = 0.0
    avg_tokens_per_sec: float = 0.0
    avg_preprocessing_time: float = 0.0
    avg_total_time: float = 0.0

    # Percentiles
    percentiles: Dict[str, float] = field(default_factory=dict)

    # Total time
    total_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert summary to dictionary."""
        return {
            "total_clips": self.total_clips,
            "successful_clips": self.successful_clips,
            "failed_clips": self.failed_clips,
            "avg_ttft": self.avg_ttft,
            "avg_ttlt": self.avg_ttlt,
            "avg_tokens_per_sec": self.avg_tokens_per_sec,
            "avg_preprocessing_time": self.avg_preprocessing_time,
            "avg_total_time": self.avg_total_time,
            "percentiles": self.percentiles,
            "total_time": self.total_time
        }


class MetricsCalculator:
    """Calculate statistics from collected metrics."""

    @staticmethod
    def calculate_summary(metrics_list: List[ClipMetrics], percentiles: List[float] = None) -> BenchmarkSummary:
        """
        Calculate summary statistics from a list of clip metrics.

        Args:
            metrics_list: List of clip metrics
            percentiles: Percentiles to calculate (e.g., [0.5, 0.9, 0.95, 0.99])

        Returns:
            BenchmarkSummary with aggregated statistics
        """
        if percentiles is None:
            percentiles = [0.5, 0.9, 0.95, 0.99]

        # Filter successful metrics
        successful = [m for m in metrics_list if m.error is None]
        failed = [m for m in metrics_list if m.error is not None]

        summary = BenchmarkSummary(
            total_clips=len(metrics_list),
            successful_clips=len(successful),
            failed_clips=len(failed)
        )

        if not successful:
            return summary

        # Calculate averages
        summary.avg_ttft = np.mean([m.time_to_first_token for m in successful])
        summary.avg_ttlt = np.mean([m.time_to_last_token for m in successful])
        summary.avg_tokens_per_sec = np.mean([m.tokens_per_second for m in successful])
        summary.avg_preprocessing_time = np.mean([m.preprocessing_time for m in successful])
        summary.avg_total_time = np.mean([m.total_inference_time for m in successful])

        # Calculate percentiles
        ttft_values = [m.time_to_first_token for m in successful]
        ttlt_values = [m.time_to_last_token for m in successful]
        tps_values = [m.tokens_per_second for m in successful]

        summary.percentiles = {}
        for p in percentiles:
            summary.percentiles[f"ttft_p{int(p*100)}"] = float(np.percentile(ttft_values, p * 100))
            summary.percentiles[f"ttlt_p{int(p*100)}"] = float(np.percentile(ttlt_values, p * 100))
            summary.percentiles[f"tokens_per_sec_p{int(p*100)}"] = float(np.percentile(tps_values, p * 100))

        return summary


class MetricsCollector:
    """Collect metrics during benchmark execution."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.preprocessing_start: Optional[float] = None
        self.token_timestamps: List[float] = []

    def start_preprocessing(self) -> None:
        """Mark the start of preprocessing."""
        self.preprocessing_start = time.time()

    def end_preprocessing(self) -> float:
        """Mark the end of preprocessing and return duration."""
        if self.preprocessing_start is None:
            return 0.0
        duration = time.time() - self.preprocessing_start
        self.preprocessing_start = None
        return duration

    def start_inference(self) -> None:
        """Mark the start of inference."""
        self.start_time = time.time()
        self.token_timestamps = []

    def record_token(self) -> None:
        """Record a token emission timestamp."""
        if self.start_time is not None:
            self.token_timestamps.append(time.time())

    def get_metrics(self, response_text: str, error: Optional[str] = None) -> ClipMetrics:
        """
        Get collected metrics.

        Args:
            response_text: The response text
            error: Error message if inference failed

        Returns:
            ClipMetrics with collected data
        """
        if self.start_time is None:
            return ClipMetrics(error=error or "No timing data collected")

        end_time = time.time()

        metrics = ClipMetrics(
            total_inference_time=end_time - self.start_time,
            response_length=len(response_text),
            error=error
        )

        if self.token_timestamps:
            metrics.time_to_first_token = self.token_timestamps[0] - self.start_time
            metrics.time_to_last_token = self.token_timestamps[-1] - self.start_time
            metrics.total_tokens = len(self.token_timestamps)

            generation_time = self.token_timestamps[-1] - self.token_timestamps[0]
            if generation_time > 0:
                metrics.tokens_per_second = metrics.total_tokens / generation_time

        return metrics
