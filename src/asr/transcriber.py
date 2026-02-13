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
        language: Optional[str] = None,
        cpu_threads: int = 4
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size
            device: Device to use (cuda, cpu, auto)
            compute_type: Compute type (int8, int8_float16, float16, float32, auto)
            language: Language code
            cpu_threads: Number of threads for CPU inference
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.cpu_threads = cpu_threads
        self.model: Optional[WhisperModel] = None
    
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.cpu_threads
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
    
    def transcribe_chunks(
        self,
        chunks: list,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        on_progress=None
    ) -> Dict[str, Any]:
        """
        Transcribe multiple audio chunks and merge with offset-corrected timestamps.
        
        Args:
            chunks: List of (chunk_audio_path, offset_seconds) tuples
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to use Voice Activity Detection filter
            on_progress: Optional callback(chunk_index, total_chunks) for progress
            
        Returns:
            Merged transcription result with globally-corrected timestamps
        """
        all_segments = []
        all_words = []
        all_text_parts = []
        language = None
        language_probability = None
        segment_id_counter = 0
        
        for idx, (chunk_path, offset) in enumerate(chunks):
            if on_progress:
                on_progress(idx + 1, len(chunks))
            
            # Transcribe this chunk
            chunk_result = self.transcribe(
                chunk_path,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter
            )
            
            # Use language from first chunk (most reliable detection)
            if language is None:
                language = chunk_result.get("language")
                language_probability = chunk_result.get("language_probability")
            
            # Offset all timestamps by the chunk's global offset
            for segment in chunk_result.get("segments", []):
                segment_id_counter += 1
                segment["id"] = segment_id_counter
                segment["start"] = segment.get("start", 0) + offset
                segment["end"] = segment.get("end", 0) + offset
                
                if "words" in segment:
                    for word in segment["words"]:
                        word["start"] = word.get("start", 0) + offset
                        word["end"] = word.get("end", 0) + offset
                        all_words.append(word)
                
                all_segments.append(segment)
                all_text_parts.append(segment.get("text", ""))
            
            # Also offset standalone words list
            if word_timestamps:
                for word in chunk_result.get("words", []):
                    # Only add if not already added via segments
                    pass  # Words are already collected from segments above
        
        result = {
            "text": " ".join(all_text_parts),
            "segments": all_segments,
            "language": language or "unknown",
            "language_probability": language_probability or 0.0
        }
        
        if word_timestamps:
            result["words"] = all_words
        
        return result
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Faster Whisper models don't need explicit cleanup
        pass
