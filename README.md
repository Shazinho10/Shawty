# Shortie - YouTube Shorts Generator

A local Python application that processes video files to generate YouTube shorts by performing ASR (Automatic Speech Recognition) with timestamps and speaker diarization, then using Langchain with multiple LLM providers to analyze transcripts and select up to 5 short segments with titles and reasoning.

## Features

- **Fast ASR**: Uses Faster Whisper for quick local transcription with word-level timestamps
- **Speaker Diarization**: Identifies different speakers in the video using WhisperX
- **Multi-LLM Support**: Supports OpenAI, Anthropic, Grok, or Ollama (DeepSeek 1.5B default)
- **Brand Context**: Optional brand information for better short selection and titling
- **Structured Output**: Pydantic-validated JSON output with timestamps, titles, and reasoning

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for video processing):
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`
   - Windows: Download from https://ffmpeg.org/

4. (Optional) Set up Ollama and pull DeepSeek model:
```bash
ollama pull deepseek-r1:1.5b
```

5. (Optional) Copy `.env.example` to `.env` and add your API keys if using cloud LLMs

**Note:** Speaker diarization is optional. The app works perfectly without it - you just won't get speaker labels in the transcript. If you want diarization:
   - Get a HuggingFace token from https://huggingface.co/settings/tokens
   - Accept terms for pyannote models: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Set `HF_TOKEN` environment variable or use `--hf-token` flag

## Usage

```bash
python -m src.main <video_path> [options]
```

### Options

- `--openai-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--anthropic-key`: Anthropic API key (or set ANTHROPIC_API_KEY env var)
- `--grok-key`: Grok API key (or set GROK_API_KEY env var)
- `--brand-file`: Path to JSON file with brand information
- `--output`: Output JSON file path (default: shorts_output.json)
- `--skip-diarization`: Skip speaker diarization (faster, but no speaker labels)
- `--model-size`: Whisper model size (tiny, base, small, medium, large-v2, large-v3, default: base)
- `--hf-token`: HuggingFace token for diarization (or set HF_TOKEN env var)

### Example

```bash
# Using default Ollama
python -m src.main video.mp4

# Using OpenAI with brand info
python -m src.main video.mp4 --openai-key sk-... --brand-file brand.json

# Using Anthropic
python -m src.main video.mp4 --anthropic-key sk-ant-...
```

## Brand Information Format

Create a JSON file with brand information:

```json
{
  "name": "My Brand",
  "description": "A tech company focused on AI",
  "target_audience": "Developers and tech enthusiasts",
  "tone": "Professional yet approachable"
}
```

## Output Format

The application generates a JSON file with structured output:

```json
{
  "shorts": [
    {
      "title": "Short title",
      "start_time": 10.5,
      "end_time": 45.2,
      "reason": "Explanation of why this segment was selected"
    }
  ],
  "total_shorts": 3
}
```

## Requirements

- Python 3.9+
- FFmpeg
- (Optional) Ollama for local LLM inference (default)

## Notes

- **Speaker diarization is optional** - the app works great without it! You just won't get speaker labels in transcripts.
- The default LLM is Ollama with DeepSeek 1.5B - make sure Ollama is running: `ollama serve`
- For better transcription accuracy, use larger Whisper models (medium, large-v2, large-v3) but they're slower
- All features work locally - no external services required (unless using cloud LLM APIs)
