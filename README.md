# Qwen3-Omni Video Benchmark Framework

A Python-based benchmark framework for evaluating the Qwen3-Omni multimodal model on video analysis tasks. The framework implements a sliding window approach to extract N-second clips from videos and collects detailed performance metrics including Time to First Token (TTFT), Tokens Per Second, and Time to Last Token (TTLT).

## Features

- **Configurable Hyperparameters:** Adjust clip duration and frames per clip
- **Sliding Window Extraction:** Process entire videos with configurable overlap
- **Accurate Metrics Collection:** Streaming-based measurement for precise TTFT
- **Multiple Export Formats:** JSON, CSV, and Markdown reports
- **Statistical Analysis:** Built-in percentiles and aggregation

## Installation

### Prerequisites

1. **FFmpeg** must be installed on your system:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

2. **Qwen3-Omni vLLM server** running with the model loaded.

### Python Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run a benchmark with default settings (5s clips, 5 frames):

```bash
python benchmarks/run_benchmark.py --video ./videos/test.mp4
```

### Custom Hyperparameters

Adjust clip duration and frame count:

```bash
python benchmarks/run_benchmark.py \
  --video ./videos/test.mp4 \
  --clip-duration 10 \
  --frames 10
```

### Multiple Runs for Statistics

Run multiple times to get statistical significance:

```bash
python benchmarks/run_benchmark.py \
  --video ./videos/test.mp4 \
  --runs 5
```

## Configuration

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | Required | Path to video file |
| `--clip-duration` | 5 | Clip duration in seconds |
| `--frames-per-clip` | 5 | Number of frames per clip |
| `--api-url` | `http://localhost:8078/v1` | API base URL |
| `--api-key` | `qwen3-omni-api-key` | API authentication key |
| `--model` | `qwen3-omni` | Model name |
| `--max-tokens` | 512 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--runs` | 1 | Number of benchmark runs |
| `--output` | `results/` | Output directory |
| `--format` | `both` | Output format (json, csv, both) |

### Environment Variables

You can also set configuration via environment variables:

```bash
export QWEN_OMNI_API_URL="http://localhost:8078/v1"
export QWEN_OMNI_API_KEY="your-api-key"
export QWEN_OMNI_MODEL="qwen3-omni"
```

## Output Structure

Results are saved in the following structure:

```
results/
├── raw/                    # Raw JSON results
│   └── benchmark_20260303_154322.json
├── processed/              # CSV summaries
│   ├── summary.csv         # Aggregated metrics
│   └── clips.csv          # Per-clip details
├── reports/                # Markdown reports
│   └── benchmark_20260303_154322.md
└── temp/                   # Temporary files (auto-cleaned)
```

## Metrics Explained

### Time to First Token (TTFT)
The time from when the request is sent to when the first token is received. Measures model responsiveness and initial processing overhead.

### Time to Last Token (TTLT)
The time from when the request is sent to when the last token is received. Measures total response time.

### Tokens Per Second
The number of tokens generated per second after the first token. Calculated as: `total_tokens / (TTLT - TTFT)`.

### Preprocessing Time
The time taken to extract frames and audio from the video clip.

## Example Results

### Summary CSV

```csv
run_id,timestamp,video_path,clip_duration,frames_per_clip,total_clips,successful_clips,avg_ttft,avg_tokens_per_sec
benchmark_20260303_154322,2026-03-03T15:43:22Z,./videos/test.mp4,5,5,6,6,0.250,44.8
```

### Markdown Report

```markdown
# Benchmark Report - benchmark_20260303_154322

## Configuration
- **Clip Duration:** 5s
- **Frames per Clip:** 5
- **Model:** qwen3-omni

## Performance Summary
- **Total Clips:** 6
- **Avg TTFT:** 0.250s
- **Avg TTLT:** 2.500s
- **Avg Tokens/sec:** 44.8
```

## Architecture

### Core Components

```
benchmark_framework/
├── config/              # Configuration management
│   ├── settings.py     # Pydantic config classes
│   └── defaults.yaml   # Default hyperparameters
├── core/               # Core benchmark logic
│   ├── benchmark.py    # Main orchestrator
│   ├── clip_extractor.py  # Sliding window extraction
│   └── metrics.py      # Metrics collection
├── api/                # API client
│   └── client.py       # OpenAI-compatible client
├── preprocessing/      # Video processing
│   └── video_processor.py  # FFmpeg wrapper
└── storage/            # Results storage
    ├── results.py      # Data models
    └── exporter.py     # Export formats
```

## Advanced Usage

### Python API

Use the benchmark as a Python library:

```python
from pathlib import Path
from benchmark_framework.config.settings import BenchmarkConfig, VideoConfig
from benchmark_framework.core.benchmark import VideoBenchmark

# Configure
config = BenchmarkConfig(
    clip_duration=5,
    frames_per_clip=5,
    api_url="http://localhost:8078/v1"
)

video_config = VideoConfig(
    input_path=Path("video.mp4"),
    audio_sample_rate=16000
)

# Run benchmark
benchmark = VideoBenchmark(config, video_config)
result = benchmark.run_benchmark(Path("video.mp4"))

# Save results
benchmark.save_results(result, Path("results"))
```

## Troubleshooting

### FFmpeg not found
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

### Connection refused
Make sure the vLLM server is running:
```bash
docker compose logs -f
```

### Out of memory errors
Reduce clip duration or frames per clip:
```bash
python benchmarks/run_benchmark.py --video test.mp4 --clip-duration 3 --frames 3
```

## License

MIT License - See LICENSE file for details.
