"""Video processing utilities"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def validate_video_file(video_path: str) -> bool:
    """
    Validate that the video file exists and is accessible.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    path = Path(video_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    
    # Check if ffmpeg/ffprobe can read the file header (fast check)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True,
            timeout=30  # 30 second timeout for metadata read
        )
        if result.returncode != 0:
            # Try ffmpeg as fallback if ffprobe fails
            result = subprocess.run(
                ["ffmpeg", "-i", str(path), "-t", "1", "-f", "null", "-"],
                capture_output=True,
                timeout=30
            )
        # Check for file not found errors
        stderr = result.stderr.decode() if result.stderr else ""
        if "No such file" in stderr or "does not exist" in stderr.lower():
            raise ValueError(f"Cannot read video file: {video_path}")
    except subprocess.TimeoutExpired:
        raise ValueError(f"Video file validation timed out: {video_path}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg/FFprobe not found. Please install FFmpeg to use this application.")
    
    return True


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to the video file
        output_path: Optional output path for audio file. If None, creates temp file.
        
    Returns:
        Path to the extracted audio file
    """
    validate_video_file(video_path)
    
    if output_path is None:
        # Create temporary audio file
        temp_dir = tempfile.gettempdir()
        video_name = Path(video_path).stem
        output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
    
    # Extract audio as WAV (16kHz mono for Whisper)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-f", "wav",      # Force WAV format
                "-acodec", "pcm_s16le", # PCM 16-bit little endian (fastest)
                "-ar", "16000",  # Sample rate 16kHz (for Whisper)
                "-ac", "1",      # Mono channel
                "-y",            # Overwrite output file
                str(output_path)
            ],
            check=True,
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out")
    
    if not os.path.exists(output_path):
        raise RuntimeError("Audio extraction failed: output file not created")
    
    return output_path


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out while getting video duration")
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output")


def extract_audio_chunks(
    video_path: str,
    chunk_duration: float = 600.0,
    on_progress=None
) -> List[Tuple[str, float]]:
    """
    Extract audio from video in time-based chunks.
    
    Args:
        video_path: Path to the video file
        chunk_duration: Duration of each chunk in seconds (default 600 = 10 min)
        on_progress: Optional callback(chunk_index, total_chunks) for progress
        
    Returns:
        List of (chunk_audio_path, offset_seconds) tuples
    """
    validate_video_file(video_path)
    total_duration = get_video_duration(video_path)
    
    # Calculate number of chunks
    import math
    num_chunks = math.ceil(total_duration / chunk_duration)
    
    if num_chunks <= 1:
        # Video fits in a single chunk, use regular extraction
        audio_path = extract_audio(video_path)
        return [(audio_path, 0.0)]
    
    temp_dir = tempfile.gettempdir()
    video_name = Path(video_path).stem
    chunks = []
    
    for i in range(num_chunks):
        offset = i * chunk_duration
        chunk_path = os.path.join(temp_dir, f"{video_name}_chunk_{i}.wav")
        
        if on_progress:
            on_progress(i + 1, num_chunks)
        
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss", str(offset),         # Seek to offset (before -i for fast seek)
                    "-i", str(video_path),
                    "-t", str(chunk_duration),  # Duration of this chunk
                    "-f", "wav",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    str(chunk_path)
                ],
                check=True,
                capture_output=True,
                timeout=300  # 5 min timeout per chunk
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio chunk {i + 1}: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Audio chunk {i + 1} extraction timed out")
        
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
            chunks.append((chunk_path, offset))
    
    if not chunks:
        raise RuntimeError("No audio chunks were produced")
    
    return chunks
