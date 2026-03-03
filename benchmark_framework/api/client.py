"""Qwen3-Omni API client with streaming support for accurate metrics measurement."""

import asyncio
import os
import time
import base64
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import httpx
import json


@dataclass
class InferenceMetrics:
    """Metrics collected during inference."""
    time_to_first_token: float = 0.0      # Request sent → First token received
    time_to_last_token: float = 0.0       # Request sent → Last token received
    tokens_per_second: float = 0.0        # Tokens generated per second
    total_tokens: int = 0
    total_inference_time: float = 0.0


@dataclass
class InferenceResult:
    """Result from inference with metrics."""
    text: str
    metrics: InferenceMetrics
    raw_response: dict = field(default_factory=dict)


class QwenOmniClient:
    """Client for Qwen3-Omni OpenAI-compatible API with streaming support."""

    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: float = 300.0):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 data URL."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _encode_audio(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 data URL."""
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/wav;base64,{b64}"

    def _build_request(
        self,
        frames: List[bytes],
        audio: bytes,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        enable_audio: bool = True,
        enable_video: bool = True
    ) -> dict:
        """Build API request payload."""
        content = []

        # Add images if video enabled
        if enable_video:
            for frame_bytes in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(frame_bytes)}
                })

        # Add audio if audio enabled
        if enable_audio:
            content.append({
                "type": "audio_url",
                "audio_url": {"url": self._encode_audio(audio)}
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

    async def infer(
        self,
        frames: List[bytes],
        audio: bytes,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        enable_audio: bool = True,
        enable_video: bool = True
    ) -> InferenceResult:
        """
        Run inference with streaming to collect accurate metrics.

        Args:
            frames: List of frame images as bytes
            audio: Audio data as bytes
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_audio: Enable audio modality
            enable_video: Enable video modality

        Returns:
            InferenceResult with text and metrics
        """
        request_payload = self._build_request(frames, audio, prompt, max_tokens, temperature, enable_audio, enable_video)

        request_sent = time.time()
        first_token_time: Optional[float] = None
        all_chunks = []
        chunk_count = 0

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.api_url}/chat/completions",
                    json=request_payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()

                    # Debug: Log content type
                    content_type = response.headers.get("content-type", "")

                    # Stream the response
                    async for line in response.aiter_lines():
                        if not line.strip() or not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            # vLLM returns: {"choices": [{"delta": {"content": "token"}}]}
                            delta = data.get("choices", [{}])[0].get("delta", {})

                            # Capture first token time
                            if first_token_time is None and ("content" in delta or "text" in delta):
                                first_token_time = time.time()

                            # Accumulate content (check both "content" and "text" fields)
                            content_text = delta.get("content") or delta.get("text", "")
                            if content_text:
                                all_chunks.append(content_text)
                                chunk_count += 1

                        except json.JSONDecodeError as e:
                            # Skip malformed JSON chunks
                            continue

            except Exception as e:
                # Return error with elapsed time
                elapsed = time.time() - request_sent
                return InferenceResult(
                    text=f"Error: {type(e).__name__}: {e}",
                    metrics=InferenceMetrics(
                        time_to_first_token=0.0,
                        time_to_last_token=elapsed,
                        total_inference_time=elapsed
                    ),
                    raw_response={"error": str(e)}
                )

        last_token_time = time.time()

        # Calculate metrics
        full_text = "".join(all_chunks)
        total_inference_time = last_token_time - request_sent

        metrics = InferenceMetrics(
            time_to_first_token=first_token_time - request_sent if first_token_time else 0.0,
            time_to_last_token=total_inference_time,
            total_tokens=chunk_count,
            total_inference_time=total_inference_time
        )

        # Calculate tokens per second
        if first_token_time and chunk_count > 0:
            generation_time = last_token_time - first_token_time
            metrics.tokens_per_second = chunk_count / generation_time if generation_time > 0 else 0.0

        return InferenceResult(
            text=full_text,
            metrics=metrics,
            raw_response={"text": full_text}
        )

    def infer_sync(
        self,
        frames: List[bytes],
        audio: bytes,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        enable_audio: bool = True,
        enable_video: bool = True
    ) -> InferenceResult:
        """
        Synchronous wrapper for infer.

        Args:
            frames: List of frame images as bytes
            audio: Audio data as bytes
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_audio: Enable audio modality
            enable_video: Enable video modality

        Returns:
            InferenceResult with text and metrics
        """
        return asyncio.run(self.infer(frames, audio, prompt, max_tokens, temperature, enable_audio, enable_video))
