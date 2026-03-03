#!/usr/bin/env python3
"""
Benchmark suite wrapper for running multiple tests with different configurations.

This script runs benchmarks with various modality combinations and hyperparameters,
then generates a comprehensive comparison board.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_framework.config.settings import BenchmarkConfig, VideoConfig
from benchmark_framework.core.benchmark import VideoBenchmark
from benchmark_framework.storage.benchmark_board import BenchmarkBoardGenerator


# Predefined test configurations
PREDEFINED_TESTS = [
    {
        "name": "audio_only",
        "enable_audio": True,
        "enable_video": False,
        "frames_per_clip": 0,
        "description": "Audio only, no video frames"
    },
    {
        "name": "video_only_6frames",
        "enable_audio": False,
        "enable_video": True,
        "frames_per_clip": 6,
        "description": "Video only with 6 frames per clip"
    },
    {
        "name": "av_2frames",
        "enable_audio": True,
        "enable_video": True,
        "frames_per_clip": 2,
        "description": "Audio + Video with 2 frames per clip"
    },
    {
        "name": "av_6frames",
        "enable_audio": True,
        "enable_video": True,
        "frames_per_clip": 6,
        "description": "Audio + Video with 6 frames per clip"
    },
    {
        "name": "av_10frames",
        "enable_audio": True,
        "enable_video": True,
        "frames_per_clip": 10,
        "description": "Audio + Video with 10 frames per clip"
    },
    {
        "name": "av_15frames",
        "enable_audio": True,
        "enable_video": True,
        "frames_per_clip": 15,
        "description": "Audio + Video with 15 frames per clip"
    },
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Omni benchmark suite with multiple configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Video input
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Path to video file"
    )

    # Test selection
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated list of test names to run (default: all predefined tests)"
    )

    # Custom configurations
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated list of frames_per_clip values to test with all modalities"
    )

    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        choices=["audio", "video", "audio_video", "all"],
        help="Modality combinations to test (for use with --frames)"
    )

    # Benchmark hyperparameters
    parser.add_argument(
        "--clip-duration",
        type=int,
        default=5,
        help="Clip duration in seconds"
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

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: results/benchmark_suite_<timestamp>)"
    )

    return parser.parse_args()


def build_test_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Build list of test configurations based on CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of test configuration dictionaries
    """
    configs = []

    # If custom frames specified, build custom tests
    if args.frames:
        frames_list = [int(f.strip()) for f in args.frames.split(",")]

        # Determine which modalities to test
        if args.modalities == "audio":
            modality_configs = [
                {"enable_audio": True, "enable_video": False, "frames_per_clip": 0}
            ]
        elif args.modalities == "video":
            modality_configs = [
                {"enable_audio": False, "enable_video": True}
            ]
        elif args.modalities == "audio_video":
            modality_configs = [
                {"enable_audio": True, "enable_video": True}
            ]
        else:  # all or None
            modality_configs = [
                {"enable_audio": True, "enable_video": False, "frames_per_clip": 0},
                {"enable_audio": False, "enable_video": True},
                {"enable_audio": True, "enable_video": True}
            ]

        # Build test configs
        for i, mod_cfg in enumerate(modality_configs):
            for frames in frames_list:
                if not mod_cfg.get("enable_video", True):
                    # Audio-only: frames is always 0
                    frames_val = 0
                else:
                    frames_val = frames

                name_parts = []
                if mod_cfg.get("enable_audio") and mod_cfg.get("enable_video"):
                    name_parts.append("av")
                elif mod_cfg.get("enable_audio"):
                    name_parts.append("audio_only")
                else:
                    name_parts.append("video_only")

                if mod_cfg.get("enable_video", True):
                    name_parts.append(f"{frames}frames")

                configs.append({
                    "name": "_".join(name_parts),
                    **mod_cfg,
                    "frames_per_clip": frames_val,
                    "description": f"Custom test"
                })

    # Use predefined tests (optionally filtered)
    else:
        if args.tests:
            # Filter predefined tests
            selected_tests = [t.strip() for t in args.tests.split(",")]
            for test in PREDEFINED_TESTS:
                if test["name"] in selected_tests:
                    configs.append(test)
        else:
            # Use all predefined tests
            configs = PREDEFINED_TESTS.copy()

    return configs


def run_single_test(
    test_config: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path
):
    """
    Run a single benchmark test.

    Args:
        test_config: Test configuration dictionary
        args: Parsed command line arguments
        output_dir: Output directory for this test

    Returns:
        BenchmarkResult or None if failed
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {test_config['name']}")
    print(f"Description: {test_config['description']}")
    print(f"{'=' * 60}")

    # Create benchmark configuration
    config = BenchmarkConfig(
        clip_duration=args.clip_duration,
        frames_per_clip=test_config["frames_per_clip"],
        enable_audio=test_config["enable_audio"],
        enable_video=test_config["enable_video"],
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=output_dir,
        num_runs=1
    )

    # Create video configuration
    video_config = VideoConfig(
        input_path=args.video,
        audio_sample_rate=args.audio_sample_rate,
        audio_channels=args.audio_channels,
        temp_dir=output_dir / "temp"
    )

    # Initialize benchmark
    benchmark = VideoBenchmark(config, video_config)

    # Print configuration
    print(f"Modality: ", end="")
    if config.enable_audio and config.enable_video:
        print("Audio + Video")
    elif config.enable_audio:
        print("Audio only")
    else:
        print("Video only")
    print(f"Frames per clip: {config.frames_per_clip}")
    print(f"Model: {config.model_name}")
    print()

    # Run benchmark
    try:
        result = benchmark.run_benchmark(args.video, prompt=args.prompt)

        # Save individual test results
        formats = ["json", "markdown"]
        test_output_dir = output_dir / "raw"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = benchmark.save_results(result, output_dir, formats)

        print(f"\nTest '{test_config['name']}' completed successfully")
        print(f"Avg TTFT: {result.summary.avg_ttft:.4f}s")
        print(f"Avg TTLT: {result.summary.avg_ttlt:.4f}s")
        print(f"Avg Tokens/sec: {result.summary.avg_tokens_per_sec:.2f}")

        return result

    except Exception as e:
        print(f"\nTest '{test_config['name']}' failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    args = parse_args()

    # Validate video file exists
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Build test configurations
    test_configs = build_test_configs(args)

    if not test_configs:
        print("Error: No test configurations selected")
        sys.exit(1)

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path("results") / f"benchmark_suite_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    # Print suite info
    print("=" * 60)
    print("Qwen3-Omni Benchmark Suite")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Tests to run: {len(test_configs)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print("\nTest configurations:")
    for cfg in test_configs:
        print(f"  - {cfg['name']}: {cfg['description']}")
    print()

    # Run all tests
    results = []
    for test_config in test_configs:
        result = run_single_test(test_config, args, output_dir)
        if result:
            results.append((test_config['name'], result))

    # Generate benchmark board
    if results:
        print("\n" + "=" * 60)
        print("Generating benchmark board...")
        print("=" * 60)

        board_generator = BenchmarkBoardGenerator()
        board_path = output_dir / "benchmark_board.md"
        board_generator.generate_board(
            results=results,
            output_path=board_path,
            video_path=str(args.video)
        )

        print(f"\nBenchmark board saved to: {board_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("Benchmark Suite Complete!")
    print("=" * 60)
    print(f"Total tests: {len(test_configs)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(test_configs) - len(results)}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
