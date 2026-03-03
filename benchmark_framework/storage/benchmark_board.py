"""Benchmark board generator for comparing multiple test runs."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from benchmark_framework.storage.results import BenchmarkResult


class BenchmarkBoardGenerator:
    """Generate comprehensive comparison boards for benchmark suites."""

    def generate_board(
        self,
        results: List[tuple[str, BenchmarkResult]],
        output_path: Path,
        video_path: str = None
    ) -> Path:
        """
        Generate a benchmark board comparing multiple test results.

        Args:
            results: List of (test_name, BenchmarkResult) tuples
            output_path: Path to save the board
            video_path: Path to the video file (optional, for metadata)

        Returns:
            Path to the generated board file
        """
        lines = []

        # Header
        lines.append("# Benchmark Comparison Board")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if video_path:
            lines.append(f"**Video:** `{video_path}`")
        lines.append(f"**Tests:** {len(results)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Configuration matrix
        lines.append("## Test Configurations")
        lines.append("")
        lines.append("| Test | Modality | Frames | Clip Duration | Model |")
        lines.append("|------|----------|--------|---------------|-------|")

        for test_name, result in results:
            modality = self._get_modality_label(result.config.enable_audio, result.config.enable_video)
            frames = result.config.frames_per_clip if result.config.enable_video else "N/A"
            lines.append(
                f"| {test_name} | {modality} | {frames} | "
                f"{result.config.clip_duration}s | {result.config.model_name} |"
            )

        lines.append("")

        # Performance comparison
        lines.append("## Performance Comparison")
        lines.append("")
        lines.append("| Test | TTFT (s) | TTLT (s) | Tokens/sec | Total Clips | Success Rate |")
        lines.append("|------|----------|----------|------------|-------------|--------------|")

        for test_name, result in results:
            ttft = f"{result.summary.avg_ttft:.4f}" if result.summary.avg_ttft else "N/A"
            ttlt = f"{result.summary.avg_ttlt:.4f}" if result.summary.avg_ttlt else "N/A"
            tps = f"{result.summary.avg_tokens_per_sec:.2f}" if result.summary.avg_tokens_per_sec else "N/A"
            total = result.summary.total_clips
            success = result.summary.successful_clips
            success_rate = f"{(success/total*100):.1f}%" if total > 0 else "N/A"

            lines.append(f"| {test_name} | {ttft} | {ttlt} | {tps} | {total} | {success_rate} |")

        lines.append("")

        # Summary insights
        lines.append("## Summary Insights")
        lines.append("")

        # Best performers
        lines.append("### Best Performers")
        lines.append("")

        # Find best TTFT
        valid_results = [(n, r) for n, r in results if r.summary.avg_ttft and r.summary.avg_ttft > 0]
        if valid_results:
            best_ttft = min(valid_results, key=lambda x: x[1].summary.avg_ttft)
            lines.append(f"- **Fastest First Token:** {best_ttft[0]} ({best_ttft[1].summary.avg_ttft:.4f}s)")

        # Find best TTLT
        valid_results = [(n, r) for n, r in results if r.summary.avg_ttlt and r.summary.avg_ttlt > 0]
        if valid_results:
            best_ttlt = min(valid_results, key=lambda x: x[1].summary.avg_ttlt)
            lines.append(f"- **Fastest Complete Response:** {best_ttlt[0]} ({best_ttlt[1].summary.avg_ttlt:.4f}s)")

        # Find best tokens/sec
        valid_results = [(n, r) for n, r in results if r.summary.avg_tokens_per_sec and r.summary.avg_tokens_per_sec > 0]
        if valid_results:
            best_tps = max(valid_results, key=lambda x: x[1].summary.avg_tokens_per_sec)
            lines.append(f"- **Highest Throughput:** {best_tps[0]} ({best_tps[1].summary.avg_tokens_per_sec:.2f} tokens/sec)")

        # Best success rate
        valid_results = [(n, r) for n, r in results if r.summary.total_clips > 0]
        if valid_results:
            best_success = max(valid_results, key=lambda x: x[1].summary.successful_clips / x[1].summary.total_clips)
            success_rate = (best_success[1].summary.successful_clips / best_success[1].summary.total_clips) * 100
            lines.append(f"- **Best Success Rate:** {best_success[0]} ({success_rate:.1f}%)")

        lines.append("")

        # Per-test details
        lines.append("## Per-Test Details")
        lines.append("")

        for test_name, result in results:
            lines.append(f"### {test_name}")
            lines.append("")
            modality = self._get_modality_label(result.config.enable_audio, result.config.enable_video)
            lines.append(f"**Configuration:** {modality}, {result.config.frames_per_clip} frames/clip")
            lines.append("")

            lines.append("**Metrics:**")
            lines.append(f"- Total Clips: {result.summary.total_clips}")
            lines.append(f"- Successful: {result.summary.successful_clips}")
            lines.append(f"- Failed: {result.summary.failed_clips}")
            lines.append(f"- Avg TTFT: {result.summary.avg_ttft:.4f}s" if result.summary.avg_ttft else "- Avg TTFT: N/A")
            lines.append(f"- Avg TTLT: {result.summary.avg_ttlt:.4f}s" if result.summary.avg_ttlt else "- Avg TTLT: N/A")
            lines.append(f"- Avg Tokens/sec: {result.summary.avg_tokens_per_sec:.2f}" if result.summary.avg_tokens_per_sec else "- Avg Tokens/sec: N/A")
            lines.append(f"- Total Time: {result.summary.total_time:.2f}s")
            lines.append("")

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return output_path

    def _get_modality_label(self, enable_audio: bool, enable_video: bool) -> str:
        """Get a human-readable modality label."""
        if enable_audio and enable_video:
            return "A+V"
        elif enable_audio:
            return "Audio"
        elif enable_video:
            return "Video"
        else:
            return "None"
