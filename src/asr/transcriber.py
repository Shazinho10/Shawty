"""ASR transcription using Faster Whisper"""

import os
from typing import List, Dict, Any, Optional
from faster_whisper import WhisperModel


class Transcriber:
    """Transcriber using Faster Whisper for fast local transcription"""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use (cuda, cpu, auto)
            compute_type: Compute type (int8, int8_float16, float16, float32, auto)
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model: Optional[WhisperModel] = None
    
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
    
    def transcribe(
        self,
        audio_path: str,
        word_timestamps: bool = True,
        vad_filter: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to audio file
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to use Voice Activity Detection filter
            
        Returns:
            Dictionary containing:
                - text: Full transcript text
                - segments: List of segments with timestamps
                - language: Detected language
                - words: List of words with timestamps (if word_timestamps=True)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self._load_model()
        
        # Transcribe with word-level timestamps
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None
        )
        
        # Convert segments to list and extract data
        segment_list = []
        word_list = []
        full_text_parts = []
        
        for segment in segments:
            segment_data = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            }
            
            if word_timestamps and segment.words:
                segment_words = []
                for word in segment.words:
                    word_data = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    segment_words.append(word_data)
                    word_list.append(word_data)
                
                segment_data["words"] = segment_words
            
            segment_list.append(segment_data)
            full_text_parts.append(segment_data["text"])
        
        result = {
            "text": " ".join(full_text_parts),
            "segments": segment_list,
            "language": info.language,
            "language_probability": info.language_probability
        }
        
        if word_timestamps:
            result["words"] = word_list
        
        return result
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Faster Whisper models don't need explicit cleanup
        pass
