"""Export benchmark results to various formats."""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List

from benchmark_framework.storage.results import BenchmarkResult, ClipResult


class ResultsExporter:
    """Export benchmark results to different formats."""

    def export_json(self, result: BenchmarkResult, output_path: Path) -> None:
        """
        Export result as JSON.

        Args:
            result: Benchmark result to export
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def export_csv_summary(self, results: List[BenchmarkResult], output_path: Path) -> None:
        """
        Export summary of multiple benchmark runs as CSV.

        Args:
            results: List of benchmark results
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                "run_id", "timestamp", "video_path",
                "clip_duration", "frames_per_clip",
                "total_clips", "successful_clips", "failed_clips",
                "avg_ttft", "avg_ttlt", "avg_tokens_per_sec",
                "avg_preprocessing_time", "avg_total_time"
            ]
            writer.writerow(header)

            # Rows
            for result in results:
                row = [
                    result.run_id,
                    result.timestamp.isoformat(),
                    result.video_path,
                    result.config.clip_duration,
                    result.config.frames_per_clip,
                    result.summary.total_clips,
                    result.summary.successful_clips,
                    result.summary.failed_clips,
                    f"{result.summary.avg_ttft:.4f}",
                    f"{result.summary.avg_ttlt:.4f}",
                    f"{result.summary.avg_tokens_per_sec:.2f}",
                    f"{result.summary.avg_preprocessing_time:.4f}",
                    f"{result.summary.avg_total_time:.4f}"
                ]
                writer.writerow(row)

    def export_csv_clips(self, result: BenchmarkResult, output_path: Path) -> None:
        """
        Export per-clip details as CSV.

        Args:
            result: Benchmark result to export
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                "run_id", "clip_index", "start_time", "end_time", "duration",
                "ttft", "ttlt", "tokens_per_sec", "total_tokens",
                "preprocessing_time", "total_inference_time",
                "response_length", "frame_count", "error"
            ]
            writer.writerow(header)

            # Rows
            for clip_result in result.clip_results:
                row = [
                    result.run_id,
                    clip_result.clip_index,
                    f"{clip_result.start_time:.2f}",
                    f"{clip_result.end_time:.2f}",
                    f"{clip_result.duration:.2f}",
                    f"{clip_result.metrics.time_to_first_token:.4f}",
                    f"{clip_result.metrics.time_to_last_token:.4f}",
                    f"{clip_result.metrics.tokens_per_second:.2f}",
                    clip_result.metrics.total_tokens,
                    f"{clip_result.metrics.preprocessing_time:.4f}",
                    f"{clip_result.metrics.total_inference_time:.4f}",
                    clip_result.metrics.response_length,
                    clip_result.frame_count,
                    clip_result.metrics.error or ""
                ]
                writer.writerow(row)

    def export_markdown_report(self, result: BenchmarkResult, output_path: Path) -> None:
        """
        Export a markdown report.

        Args:
            result: Benchmark result to export
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Benchmark Report - {result.run_id}",
            "",
            f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Video:** {result.video_path}",
            "",
            "## Configuration",
            "",
            f"- **Clip Duration:** {result.config.clip_duration}s",
            f"- **Frames per Clip:** {result.config.frames_per_clip}",
            f"- **Model:** {result.config.model_name}",
            f"- **Max Tokens:** {result.config.max_tokens}",
            f"- **Temperature:** {result.config.temperature}",
            "",
            "## Performance Summary",
            "",
            f"- **Total Clips:** {result.summary.total_clips}",
            f"- **Successful:** {result.summary.successful_clips}",
            f"- **Failed:** {result.summary.failed_clips}",
            "",
            "### Average Metrics",
            "",
            f"- **Time to First Token (TTFT):** {result.summary.avg_ttft:.4f}s",
            f"- **Time to Last Token (TTLT):** {result.summary.avg_ttlt:.4f}s",
            f"- **Tokens Per Second:** {result.summary.avg_tokens_per_sec:.2f}",
            f"- **Preprocessing Time:** {result.summary.avg_preprocessing_time:.4f}s",
            "",
            "### Percentiles",
            ""
        ]

        for key, value in result.summary.percentiles.items():
            lines.append(f"- **{key}:** {value:.4f}s" if "ttft" in key or "ttlt" in key else f"- **{key}:** {value:.2f}")

        lines.extend([
            "",
            "## Per-Clip Details",
            "",
            "| Clip | Start | End | TTFT (s) | TTLT (s) | Tokens/sec | Tokens | Preproc (s) |",
            "|------|-------|-----|----------|----------|------------|--------|-------------|"
        ])

        for clip in result.clip_results:
            lines.append(
                f"| {clip.clip_index} | {clip.start_time:.1f}s | {clip.end_time:.1f}s | "
                f"{clip.metrics.time_to_first_token:.4f} | {clip.metrics.time_to_last_token:.4f} | "
                f"{clip.metrics.tokens_per_second:.2f} | {clip.metrics.total_tokens} | "
                f"{clip.metrics.preprocessing_time:.4f} |"
            )

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
