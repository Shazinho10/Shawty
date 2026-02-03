"""Video processing utilities"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


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
    
    # Check if ffmpeg can read the file
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
            timeout=10
        )
        # ffmpeg returns non-zero exit code even for valid files when using null output
        # Check if error is about codec/format (which is fine) vs file not found
        if result.stderr and "No such file" in result.stderr.decode():
            raise ValueError(f"Cannot read video file: {video_path}")
    except subprocess.TimeoutExpired:
        raise ValueError(f"Video file validation timed out: {video_path}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to use this application.")
    
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
                "-ar", "16000",  # Sample rate 16kHz
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
