#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VIDEO_PATH="${SCRIPT_DIR}/videos/test.mp4"
API_URL="${VLLM_API_URL:-http://localhost:8078/v1}"
API_KEY="${VLLM_API_KEY:-qwen3-omni-api-key}"
MODEL_NAME="${MODEL_NAME:-qwen3-omni}"

# Create temporary directory for extracted content and request files
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

echo "=========================================="
echo "Qwen3-Omni Multimodal Inference Test"
echo "=========================================="
echo "Video: ${VIDEO_PATH}"
echo "API: ${API_URL}"
echo "Model: ${MODEL_NAME}"
echo "=========================================="

# Check if video exists
if [[ ! -f "${VIDEO_PATH}" ]]; then
    echo "ERROR: Video not found at ${VIDEO_PATH}"
    exit 1
fi

# Check if vLLM server is running
echo ""
echo "Checking if vLLM server is running..."
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server is not responding at ${API_URL}"
    echo "Please start the server first with: ./start.sh"
    exit 1
fi
echo "Server is running!"

# Extract 5 frames evenly distributed throughout the video
echo ""
echo "Extracting 5 frames from video..."
ffmpeg -i "${VIDEO_PATH}" -frames:v 5 "${TEMP_DIR}/frame_%d.png" -y 2>/dev/null

# Extract audio as WAV (16kHz, mono for optimal compatibility)
echo "Extracting audio from video..."
ffmpeg -i "${VIDEO_PATH}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${TEMP_DIR}/audio.wav" -y 2>/dev/null

# List extracted files
echo ""
echo "Extracted files:"
ls -lh "${TEMP_DIR}"/ 2>/dev/null | tail -n +2

# Helper function to create JSON request file
create_request_file() {
    local output_file="$1"
    local content="$2"
    cat > "${output_file}" <<JSON
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": [${content}]
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
JSON
}

# Helper function to create JSON request file with custom max_tokens
create_request_file_max_tokens() {
    local output_file="$1"
    local content="$2"
    local max_tokens="$3"
    cat > "${output_file}" <<JSON
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": [${content}]
    }
  ],
  "max_tokens": ${max_tokens},
  "temperature": 0.7
}
JSON
}

# Build content array for multimodal request
echo ""
echo "Building multimodal request..."
echo "Modalities: 5 images + audio + text"

# Start building the JSON content using jq for proper JSON escaping
CONTENT_JSON="["

# Add images (using OpenAI-compatible format)
for i in {1..5}; do
    if [[ -f "${TEMP_DIR}/frame_${i}.png" ]]; then
        BASE64_IMG=$(base64 -w 0 "${TEMP_DIR}/frame_${i}.png")
        if [[ "${CONTENT_JSON}" != "[" ]]; then
            CONTENT_JSON+=","
        fi
        CONTENT_JSON+="{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${BASE64_IMG}\"}}"
    fi
done

# Add audio (using OpenAI-compatible format)
if [[ -f "${TEMP_DIR}/audio.wav" ]]; then
    BASE64_AUDIO=$(base64 -w 0 "${TEMP_DIR}/audio.wav")
    if [[ "${CONTENT_JSON}" != "[" ]]; then
        CONTENT_JSON+=","
    fi
    CONTENT_JSON+="{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,${BASE64_AUDIO}\"}}"
fi

# Add text prompt
if [[ "${CONTENT_JSON}" != "[" ]]; then
    CONTENT_JSON+=","
fi
CONTENT_JSON+="{\"type\":\"text\",\"text\":\"Please analyze this video content. I'm providing you with 5 frames from the video and the audio track. Describe what you observe in both the visual and audio elements. What is happening in this video?\"}"
CONTENT_JSON+="]"

# Create the complete multimodal request file
cat > "${TEMP_DIR}/request_multimodal.json" <<JSON
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": ${CONTENT_JSON}
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
JSON

# Make the multimodal API request
echo ""
echo "=========================================="
echo "Test 1: All Modalities (Images + Audio + Text)"
echo "=========================================="

RESPONSE=$(curl -s -X POST "${API_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  --data @"${TEMP_DIR}/request_multimodal.json")

echo "Response:"
echo "${RESPONSE}" | jq -r '.choices[0].message.content // .error.message // .error // .' 2>/dev/null || echo "${RESPONSE}"

echo ""
echo "=========================================="

# Test 2: Images only
echo ""
echo "Test 2: Images Only"
echo "=========================================="

IMAGES_CONTENT="["
for i in {1..5}; do
    if [[ -f "${TEMP_DIR}/frame_${i}.png" ]]; then
        BASE64_IMG=$(base64 -w 0 "${TEMP_DIR}/frame_${i}.png")
        if [[ "${IMAGES_CONTENT}" != "[" ]]; then
            IMAGES_CONTENT+=","
        fi
        IMAGES_CONTENT+="{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${BASE64_IMG}\"}}"
    fi
done
IMAGES_CONTENT+=",{\"type\":\"text\",\"text\":\"Describe what you see in these 5 images from a video.\"}"
IMAGES_CONTENT+="]"

cat > "${TEMP_DIR}/request_images.json" <<JSON
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": ${IMAGES_CONTENT}
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
JSON

IMAGES_RESPONSE=$(curl -s -X POST "${API_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  --data @"${TEMP_DIR}/request_images.json")

echo "Images Response:"
echo "${IMAGES_RESPONSE}" | jq -r '.choices[0].message.content // .error.message // .error // .' 2>/dev/null || echo "${IMAGES_RESPONSE}"

echo ""
echo "=========================================="

# Test 3: Audio only
echo ""
echo "Test 3: Audio Only"
echo "=========================================="

AUDIO_CONTENT="[{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,${BASE64_AUDIO}\"}},"
AUDIO_CONTENT+="{\"type\":\"text\",\"text\":\"Describe what you hear in this audio clip.\"}]"

cat > "${TEMP_DIR}/request_audio.json" <<JSON
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": ${AUDIO_CONTENT}
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
JSON

AUDIO_RESPONSE=$(curl -s -X POST "${API_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  --data @"${TEMP_DIR}/request_audio.json")

echo "Audio Response:"
echo "${AUDIO_RESPONSE}" | jq -r '.choices[0].message.content // .error.message // .error // .' 2>/dev/null || echo "${AUDIO_RESPONSE}"

echo ""
echo "=========================================="

# Test 4: Text only
echo ""
echo "Test 4: Text Only"
echo "=========================================="

cat > "${TEMP_DIR}/request_text.json" <<JSON
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": "Hello! Can you tell me about your capabilities as an audio-visual-language model?"
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
JSON

TEXT_RESPONSE=$(curl -s -X POST "${API_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  --data @"${TEMP_DIR}/request_text.json")

echo "Text Response:"
echo "${TEXT_RESPONSE}" | jq -r '.choices[0].message.content // .error.message // .error // .' 2>/dev/null || echo "${TEXT_RESPONSE}"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
