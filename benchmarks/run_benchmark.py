#!/usr/bin/env python3
"""
Main benchmark runner for Qwen3-Omni video analysis.

This script runs benchmarks on video files using the Qwen3-Omni model,
collecting performance metrics like time to first token (TTFT), tokens per second,
and time to last token (TTLT).
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_framework.config.settings import BenchmarkConfig, VideoConfig
from benchmark_framework.core.benchmark import VideoBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Omni video benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Video input
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Path to video file"
    )

    # Benchmark hyperparameters
    parser.add_argument(
        "--clip-duration",
        type=int,
        default=5,
        help="Clip duration in seconds"
    )
    parser.add_argument(
        "--frames",
        "--frames-per-clip",
        type=int,
        default=5,
        dest="frames_per_clip",
        help="Number of frames per clip"
    )

    # API settings
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8078/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="qwen3-omni-api-key",
        help="API authentication key"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name"
    )

    # Inference settings
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    # Benchmark settings
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )

    # Video processing settings
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz"
    )
    parser.add_argument(
        "--audio-channels",
        type=int,
        default=1,
        help="Audio channels (1=mono, 2=stereo)"
    )

    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please analyze this video content. Describe what you observe in both the visual and audio elements. What is happening in this clip?",
        help="Prompt to use for inference"
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="Output format for results"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate video file exists
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Create benchmark configuration
    config = BenchmarkConfig(
        clip_duration=args.clip_duration,
        frames_per_clip=args.frames_per_clip,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output,
        num_runs=args.runs
    )

    # Create video configuration
    video_config = VideoConfig(
        input_path=args.video,
        audio_sample_rate=args.audio_sample_rate,
        audio_channels=args.audio_channels,
        temp_dir=args.output / "temp"
    )

    # Create output directories
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw").mkdir(exist_ok=True)
    (args.output / "processed").mkdir(exist_ok=True)
    (args.output / "reports").mkdir(exist_ok=True)

    # Initialize benchmark
    benchmark = VideoBenchmark(config, video_config)

    # Print configuration
    print("=" * 60)
    print("Qwen3-Omni Video Benchmark")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Clip Duration: {config.clip_duration}s")
    print(f"Frames per Clip: {config.frames_per_clip}")
    print(f"Model: {config.model_name}")
    print(f"API: {config.api_url}")
    print(f"Runs: {config.num_runs}")
    print("=" * 60)
    print()

    # Run benchmarks
    all_results = []
    for run in range(config.num_runs):
        print(f"\n=== Run {run + 1}/{config.num_runs} ===")

        result = benchmark.run_benchmark(args.video, prompt=args.prompt)
        all_results.append(result)

        # Save results
        formats = ["json", "csv", "markdown"] if args.format == "both" else [args.format]
        saved_files = benchmark.save_results(result, args.output, formats)

        print(f"\nResults saved to:")
        for file_path in saved_files:
            print(f"  - {file_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    for i, result in enumerate(all_results):
        print(f"\nRun {i + 1} Summary:")
        print(f"  Total Clips: {result.summary.total_clips}")
        print(f"  Successful: {result.summary.successful_clips}")
        print(f"  Failed: {result.summary.failed_clips}")
        print(f"  Avg TTFT: {result.summary.avg_ttft:.4f}s")
        print(f"  Avg TTLT: {result.summary.avg_ttlt:.4f}s")
        print(f"  Avg Tokens/sec: {result.summary.avg_tokens_per_sec:.2f}")
        print(f"  Total Time: {result.summary.total_time:.2f}s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
