# Use the NVIDIA vLLM container
FROM nvcr.io/nvidia/vllm:26.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_PORT=8078 \
    VLLM_HOST=0.0.0.0 \
    MODEL_PATH=/models \
    SERVED_MODEL_NAME=qwen3-omni

WORKDIR /workspace

# Install audio support for vLLM (required for Qwen3-Omni)
RUN pip install --no-cache-dir vllm[audio] && \
    pip install --no-cache-dir qwen-omni-utils

# Expose the API port
EXPOSE ${VLLM_PORT}

# Default command (can be overridden by docker-compose)
CMD ["/bin/bash"]
