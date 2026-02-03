"""Speaker diarization using WhisperX"""

import os
from typing import Dict, Any, Optional

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    whisperx = None


class Diarizer:
    """Speaker diarization using WhisperX"""
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cpu",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        """
        Initialize the diarizer.
        
        Args:
            hf_token: HuggingFace token for accessing pyannote models (optional - diarization skipped if not provided)
            device: Device to use (cuda, cpu)
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
    
    def diarize(
        self,
        audio_path: str,
        transcription_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Perform speaker diarization on transcribed audio.
        
        Args:
            audio_path: Path to audio file
            transcription_result: Result from Transcriber.transcribe()
            
        Returns:
            Enhanced transcription result with speaker labels, or None if diarization unavailable
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Skip diarization if no HF token or whisperx not available (graceful degradation)
        if not self.hf_token:
            return None
        
        if not WHISPERX_AVAILABLE:
            return None
        
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diarization model. "
                f"Make sure HF_TOKEN is set and you have access to pyannote models. Error: {e}"
            )
        
        # Perform diarization
        try:
            diarize_segments = diarize_model(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
        except Exception as e:
            raise RuntimeError(f"Speaker diarization failed: {e}")
        
        # Align diarization with transcription segments
        try:
            result = whisperx.assign_word_speakers(
                diarize_segments,
                transcription_result
            )
        except Exception as e:
            # If alignment fails, try to merge manually
            print(f"Warning: Could not align speakers automatically: {e}")
            result = transcription_result.copy()
            # Try to add speaker info from diarization segments
            if hasattr(diarize_segments, 'segments'):
                result["diarization_segments"] = diarize_segments
            else:
                result["diarization_segments"] = list(diarize_segments) if diarize_segments else []
        
        return result
    
    def merge_with_transcription(
        self,
        transcription_result: Dict[str, Any],
        diarization_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge transcription and diarization results into structured format.
        
        Args:
            transcription_result: Result from Transcriber.transcribe()
            diarization_result: Result from Diarizer.diarize() or None if unavailable
            
        Returns:
            Merged result with speaker labels (or original if diarization unavailable)
        """
        merged = transcription_result.copy()
        
        # If no diarization result, return transcription as-is
        if not diarization_result:
            return merged
        
        # Add speaker information to segments
        if "segments" in diarization_result:
            for i, segment in enumerate(diarization_result["segments"]):
                if i < len(merged["segments"]):
                    merged["segments"][i]["speaker"] = segment.get("speaker", "SPEAKER_00")
                    
                    # Add speaker info to words if available
                    if "words" in segment and "words" in merged["segments"][i]:
                        word_speaker_map = {
                            word.get("word"): word.get("speaker", "SPEAKER_00")
                            for word in segment["words"]
                        }
                        for word in merged["segments"][i]["words"]:
                            word["speaker"] = word_speaker_map.get(word["word"], "SPEAKER_00")
        
        # Extract unique speakers
        speakers = set()
        for segment in merged.get("segments", []):
            if "speaker" in segment:
                speakers.add(segment["speaker"])
        
        merged["speakers"] = sorted(list(speakers))
        merged["num_speakers"] = len(speakers)
        
        return merged
