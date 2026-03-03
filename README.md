# Qwen3-Omni Video Benchmark Framework

Benchmark framework for evaluating Qwen3-Omni multimodal model on video analysis tasks.

## Installation

```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS

# Install Python dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Single Benchmark

```bash
python benchmarks/run_benchmark.py --video videos/test.mp4
```

### Run Benchmark Suite (Multiple Tests)

Run all 6 predefined tests (audio-only, video-only, and audio+video with 2/6/10/15 frames):

```bash
python benchmarks/run_benchmark_suite.py --video videos/test.mp4
```

Run specific tests:

```bash
python benchmarks/run_benchmark_suite.py --video videos/test.mp4 --tests audio_only,av_6frames
```

Custom frames with all modalities:

```bash
python benchmarks/run_benchmark_suite.py --video videos/test.mp4 --frames 4,8,12 --modalities all
```

## Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | Required | Path to video file |
| `--clip-duration` | 5 | Clip duration in seconds |
| `--frames` / `--frames-per-clip` | 5 | Number of frames per clip |
| `--api-url` | `http://localhost:8078/v1` | API endpoint |
| `--model` | `qwen3-omni` | Model name |
| `--output` | `results/` | Output directory |

### Benchmark Suite Options

| Option | Description |
|--------|-------------|
| `--tests` | Comma-separated test names (audio_only, video_only_6frames, av_2frames, etc.) |
| `--frames` | Custom frames_per_clip values |
| `--modalities` | audio, video, audio_video, or all |

## Output Structure

```
results/
├── raw/                    # JSON results
├── processed/              # CSV summaries
├── reports/                # Markdown reports
└── benchmark_suite_*/      # Suite runs
    └── benchmark_board.md  # Comparison report
```

## Metrics

- **TTFT**: Time to First Token (responsiveness)
- **TTLT**: Time to Last Token (total response time)
- **Tokens/sec**: Generation throughput
