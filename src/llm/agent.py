"""Langchain agent for short selection"""

import json
import re
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
import re

from ..models.output import ShortsOutput, ShortClient
from ..utils import refine_shorts_output
from .prompts import (
    get_shorts_selection_prompt,
    get_shorts_repair_prompt,
    get_titles_reasons_prompt,
    format_transcript_for_llm,
    format_brand_context,
)


class ShortsAgent:
    """Agent for selecting YouTube shorts from transcripts using LLM"""
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the shorts agent.
        
        Args:
            llm: Langchain LLM instance
        """
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=ShortsOutput)
        self.prompt = get_shorts_selection_prompt()
        self.repair_prompt = get_shorts_repair_prompt()
        self.titles_reasons_prompt = get_titles_reasons_prompt()
    
    def select_shorts(
        self,
        transcription_result: Dict[str, Any],
        brand_info: Optional[Dict[str, Any]] = None,
        target_shorts: int = 5,
        min_gap_seconds: float = 90.0
    ) -> ShortsOutput:
        """
        Select YouTube shorts from transcript using LLM.
        
        Args:
            transcription_result: Result from Transcriber or Diarizer
            brand_info: Optional brand information dict
            
        Returns:
            ShortsOutput with selected shorts
        """
        import re
        
        # Format transcript for LLM
        transcript_text = format_transcript_for_llm(transcription_result)
        
        # Format brand context
        brand_context = format_brand_context(brand_info) if brand_info else ""
        
        # Format the prompt
        formatted_prompt = self.prompt.format_messages(
            transcript=transcript_text,
            brand_context=brand_context,
            target_shorts=target_shorts,
            min_gap_seconds=round(min_gap_seconds, 2)
        )
        
        # Invoke LLM
        response = self.llm.invoke(formatted_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up response - remove thinking tags and markdown
        # Remove <think>...</think> tags (DeepSeek R1 style)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()
        
        # Clean specific DeepSeek artifacts
        content = self._clean_json_content(content)
        try:
            output = self._parse_shorts_output(content)
        except Exception:
            output = self._repair_and_parse(content)

        ranked = []
        if output.shorts:
            ranked = self._rank_and_spread(
                output.shorts,
                transcription_result,
                target_shorts,
                min_gap_seconds
            )
        refined = refine_shorts_output(
            ShortsOutput(shorts=ranked, total_shorts=len(ranked)),
            transcription_result,
            min_len=15.0,
            max_len=60.0,
            max_shorts=max(target_shorts, 5),
            min_shorts=5,
        )
        return self._enrich_shorts(refined, transcription_result)

    def _clean_json_content(self, content: str) -> str:
        """Clean common LLM JSON errors"""
        import re

        # Strip common prefixes/suffixes outside JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            content = content[json_start:json_end]

        # Remove JS-style comments
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Fix invalid timecodes on start/end times (e.g., 10.56.39.32 or 00:10:56.39)
        def _fix_timecode(match: re.Match) -> str:
            key = match.group(1)
            raw = match.group(2)
            seconds = self._parse_time_value(raw)
            if seconds is None:
                return match.group(0)
            return f"\"{key}\": {seconds}"

        content = re.sub(
            r'"((?:start|end)_time)"\s*:\s*([0-9]+(?:[.:][0-9]+){2,3})',
            _fix_timecode,
            content
        )

        # Fix timestamps with units (e.g., "10.5s" -> 10.5)
        # Look for "start_time": "10.5s" or "start_time": "10.5"
        content = re.sub(
            r'"((?:start|end)_time)"\s*:\s*"([\d\.]+)(?:s|sec|secs)?"',
            r'"\1": \2',
            content
        )

        # Fix missing commas between objects in array
        content = re.sub(r'}\s*{', '}, {', content)

        # Fix trailing commas in arrays/objects
        content = re.sub(r',\s*([\]}])', r'\1', content)

        return content

    def _parse_shorts_output(self, content: str) -> ShortsOutput:
        """Parse cleaned LLM response into ShortsOutput."""
        import re

        # Try multiple JSON extraction strategies
        json_data = None

        # Strategy 1: Look for JSON with "shorts" array
        match = re.search(r'\{[^{}]*"shorts"\s*:\s*\[.*?\][^{}]*\}', content, re.DOTALL)
        if match:
            try:
                json_data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: Find outermost braces
        if json_data is None:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    json_data = json.loads(content[json_start:json_end])
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Build minimal valid JSON from regex captures
        if json_data is None:
            # Try to extract individual shorts
            shorts_match = re.search(r'"shorts"\s*:\s*(\[.*?\])', content, re.DOTALL)
            total_match = re.search(r'"total_shorts"\s*:\s*(\d+)', content)

            if shorts_match:
                try:
                    shorts_array = json.loads(shorts_match.group(1))
                    total = int(total_match.group(1)) if total_match else len(shorts_array)
                    json_data = {"shorts": shorts_array, "total_shorts": total}
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategy 4: If content is a raw array of shorts, wrap it
        if json_data is None:
            stripped = content.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                try:
                    shorts_array = json.loads(stripped)
                    json_data = {"shorts": shorts_array, "total_shorts": len(shorts_array)}
                except json.JSONDecodeError:
                    pass

        # Strategy 5: Regex salvage for start/end pairs
        if json_data is None:
            shorts = []
            pattern = re.compile(
                r'"title"\s*:\s*"(?P<title>[^"]+)"[^{}]*?'
                r'"start_time"\s*:\s*(?P<start>[0-9:.]+)[^{}]*?'
                r'"end_time"\s*:\s*(?P<end>[0-9:.]+)',
                re.DOTALL
            )
            for m in pattern.finditer(content):
                start_val = self._parse_time_value(m.group("start"))
                end_val = self._parse_time_value(m.group("end"))
                if start_val is None or end_val is None or end_val <= start_val:
                    continue
                shorts.append({
                    "title": m.group("title"),
                    "start_time": start_val,
                    "end_time": end_val,
                    "reason": "Recovered from malformed output",
                    "score": 0,
                })
            if shorts:
                json_data = {"shorts": shorts, "total_shorts": len(shorts)}

        if json_data is None:
            raise ValueError(f"No valid JSON found in LLM response. Content preview: {content[:500]}")

        # Patch data to ensure compliance
        if not isinstance(json_data, dict):
            json_data = {"shorts": [], "total_shorts": 0}
        json_data = self._patch_shorts_data(json_data)

        return ShortsOutput(**json_data)

    def _repair_and_parse(self, content: str) -> ShortsOutput:
        """Attempt to repair malformed output via LLM, then parse."""
        try:
            formatted = self.repair_prompt.format_messages(content=content)
            response = self.llm.invoke(formatted)
            repaired = response.content if hasattr(response, "content") else str(response)
            repaired = repaired.strip()
            repaired = self._clean_json_content(repaired)
            return self._parse_shorts_output(repaired)
        except Exception:
            return ShortsOutput(shorts=[], total_shorts=0)

    def _patch_shorts_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist and have correct types"""
        # print(f"DEBUG: Patching shorts data input: {data}", flush=True)
        if not isinstance(data, dict):
            return {"shorts": [], "total_shorts": 0}

        if "shorts" not in data:
            data["shorts"] = []

        if isinstance(data.get("shorts"), dict):
            data["shorts"] = [data["shorts"]]
        elif not isinstance(data.get("shorts"), list):
            data["shorts"] = []
            
        if "total_shorts" not in data:
            data["total_shorts"] = len(data["shorts"])
            
        patched_shorts = []
        for short in data["shorts"]:
            short = self._coerce_short_item(short)
            if not isinstance(short, dict):
                continue
            # Ensure required text fields
            if "title" not in short:
                short["title"] = "Untitled Segment"
            if "reason" not in short:
                short["reason"] = "Strong standalone moment"
                
            # Ensure timestamps are valid floats
            try:
                # Handle start_time
                if "start_time" not in short:
                    continue
                start_val = self._parse_time_value(short["start_time"])
                if start_val is None:
                    continue
                short["start_time"] = float(start_val)
                    
                # Handle end_time
                if "end_time" not in short:
                    continue
                end_val = self._parse_time_value(short["end_time"])
                if end_val is None:
                    continue
                short["end_time"] = float(end_val)
                    
            except (ValueError, TypeError):
                continue # Skip shorts with invalid time formats

            # Ensure numeric score
            try:
                short["score"] = int(float(short.get("score", 0)))
            except (ValueError, TypeError):
                short["score"] = 0

            patched_shorts.append(short)
            
        data["shorts"] = patched_shorts
        data["total_shorts"] = len(patched_shorts)
        return data

    def _coerce_short_item(self, item: Any) -> Any:
        """Coerce common malformed short items into a dict."""
        if isinstance(item, dict):
            if "short" in item and isinstance(item["short"], dict):
                return item["short"]
            if "long" in item and isinstance(item["long"], dict):
                return item["long"]
            if "short" in item and isinstance(item["short"], str):
                return {
                    "title": item["short"],
                    "reason": item.get("reason", ""),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                    "score": item.get("score", 0),
                }
            return item
        return item

    def _parse_time_value(self, value: Any) -> Optional[float]:
        """Parse time values in seconds or timecode-like strings into seconds."""
        import re

        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        raw = value.strip().lower().replace("sec", "").replace("secs", "").replace("s", "")
        raw = raw.strip()
        if raw == "":
            return None

        # If it looks like a plain float, parse directly
        if re.fullmatch(r"\d+(\.\d+)?", raw):
            try:
                return float(raw)
            except ValueError:
                return None

        # Timecode patterns: HH:MM:SS(.ms) or MM:SS(.ms) or dot-separated
        if ":" in raw:
            parts = raw.split(":")
        elif raw.count(".") >= 2:
            parts = raw.split(".")
        else:
            return None

        try:
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            if len(parts) == 4:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                millis = int(parts[3])
                return hours * 3600 + minutes * 60 + seconds + (millis / 100.0)
        except ValueError:
            return None

        return None

    def _enrich_shorts(self, output: ShortsOutput, transcription_result: Dict[str, Any]) -> ShortsOutput:
        """Fill missing titles/reasons using nearby transcript context."""
        segments = transcription_result.get("segments", []) if transcription_result else []
        if not segments:
            return output
        language = str(transcription_result.get("language", "") or "").lower()
        force_english = language not in {"en", "english"}

        def window_text(start_time: float, end_time: float) -> str:
            parts = []
            for seg in segments:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                if e < start_time or s > end_time:
                    continue
                text = str(seg.get("text", "")).strip()
                if text:
                    parts.append(text)
            joined = " ".join(parts).strip()
            return joined[:1400]

        def nearest_segment_text(time_sec: float) -> str:
            best = None
            best_dist = None
            for seg in segments:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                mid = (s + e) / 2.0
                dist = abs(mid - time_sec)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = seg
            if not best:
                return ""
            return str(best.get("text", "")).strip()

        def is_generic_reason(reason: str) -> bool:
            r = reason.strip().lower()
            if r in {"", "n/a"}:
                return True
            if len(r.split()) < 5:
                return True
            if r.startswith("auto-generated"):
                return True
            return False
        
        def is_generic_title(title: str) -> bool:
            t = title.strip().lower()
            bad_titles = {
                "compelling title",
                "end time",
                "full transcript",
                "key supporting clip",
                "the core message",
                "longer clips from different sections",
                "the aerial",
                "the end of the first day",
                "a character from a charles dickens novel",
                "untitled segment",
                "untitled",
                "auto clip",
            }
            if t in bad_titles:
                return True
            if t.startswith(("like ", "because ", "so ", "then ", "yeah ")):
                return True
            if len(t.split()) <= 2 and t in {"clip", "segment", "highlight"}:
                return True
            return False

        def extract_sentences(text: str) -> list[str]:
            import re

            cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
            if not cleaned:
                return []
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            return [s.strip(" \t\n.?!") for s in sentences if s.strip(" \t\n.?!")]

        def make_title_from_text(text: str) -> str:
            sentences = extract_sentences(text)
            if not sentences:
                return ""
            sentence = sentences[0]
            lower = sentence.lower()
            for delim in (" because ", " that ", " which ", " so ", " but ", " and ", " if ", " when ", " after "):
                idx = lower.find(delim)
                if idx >= 0:
                    sentence = sentence[:idx]
                    lower = sentence.lower()
            sentence = re.sub(r'^(the|a|an|this|that|these|those|it|they|we|i|he|she)\s+', "", sentence, flags=re.IGNORECASE)
            sentence = sentence.strip(" ,;:.!?")
            if len(sentence) > 80:
                sentence = sentence[:80].rsplit(" ", 1)[0]
            if not sentence:
                return ""
            # Add a little context if the lead sentence is too short/vague
            if len(sentence.split()) < 4 and len(sentences) > 1:
                extra = sentences[1].strip(" ,;:.!?")
                if extra and len(extra.split()) >= 3:
                    sentence = f"{sentence}: {extra}"
            sentence = sentence.strip()
            if len(sentence) > 90:
                sentence = sentence[:90].rsplit(" ", 1)[0]
            return sentence[0].upper() + sentence[1:]

        def make_reason_from_text(text: str) -> str:
            sentences = extract_sentences(text)
            if not sentences:
                return ""
            reason = sentences[0].strip()
            # Build a medium-length reason (1–2 sentences, ~90–160 chars)
            if len(reason) < 60 and len(sentences) > 1:
                second = sentences[1].strip()
                if second:
                    reason = f"{reason}. {second}"
            reason = reason.strip()
            if len(reason) < 90 and len(sentences) > 2:
                third = sentences[2].strip()
                if third:
                    reason = f"{reason}. {third}"
            # Trim to a reasonable length while keeping sentences
            if len(reason) > 180:
                reason = reason[:180].rsplit(" ", 1)[0]
            if not reason.endswith("."):
                reason += "."
            return reason

        def is_non_english_text(text: str) -> bool:
            if not text:
                return False
            # Detect common non-Latin scripts (Arabic, Urdu, Hindi/Devanagari)
            for ch in text:
                code = ord(ch)
                if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
                    return True
                if 0x0900 <= code <= 0x097F:
                    return True
            # Heuristic: if very few ASCII letters, assume non-English
            letters = sum(1 for c in text if c.isalpha())
            ascii_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            if letters > 0 and (ascii_letters / letters) < 0.6:
                return True
            return False

        # Detect repeated titles and treat them as generic
        title_counts: Dict[str, int] = {}
        for s in output.shorts:
            t = (s.title or "").strip().lower()
            if t:
                title_counts[t] = title_counts.get(t, 0) + 1

        repair_targets: List[Dict[str, Any]] = []
        for idx, short in enumerate(output.shorts):
            text = window_text(short.start_time, short.end_time) or nearest_segment_text(short.start_time)
            title_key = (short.title or "").strip().lower()
            repeated_title = title_key and title_counts.get(title_key, 0) > 1
            needs_title = not short.title or is_generic_title(short.title) or repeated_title
            needs_reason = not short.reason or is_generic_reason(short.reason)
            non_english = is_non_english_text(short.title or "") or is_non_english_text(short.reason or "")
            if text:
                repair_targets.append({
                    "index": idx,
                    "start_time": round(float(short.start_time), 2),
                    "end_time": round(float(short.end_time), 2),
                    "text": text,
                })

        if repair_targets:
            self._apply_llm_title_reason_repairs(output, repair_targets)

        enriched: List[ShortClient] = []
        for idx, short in enumerate(output.shorts):
            text = window_text(short.start_time, short.end_time) or nearest_segment_text(short.start_time)
            title_key = (short.title or "").strip().lower()
            repeated_title = title_key and title_counts.get(title_key, 0) > 1
            non_english = is_non_english_text(short.title or "") or is_non_english_text(short.reason or "")
            if not short.title or is_generic_title(short.title) or repeated_title or non_english:
                if text:
                    short.title = make_title_from_text(text)
                if not short.title:
                    short.title = f"Clip {idx + 1}"
            if not short.reason or is_generic_reason(short.reason) or non_english:
                if text:
                    short.reason = make_reason_from_text(text)
                if not short.reason:
                    short.reason = "Highlights a clear, self-contained point from the segment."
            enriched.append(short)

        return ShortsOutput(shorts=enriched, total_shorts=len(enriched))

    def _apply_llm_title_reason_repairs(self, output: ShortsOutput, repair_targets: List[Dict[str, Any]]) -> None:
        """Use LLM to repair titles/reasons for low-quality clips."""
        try:
            items_lines = []
            for item in repair_targets:
                items_lines.append(
                    f"Index: {item['index']}\n"
                    f"Start: {item['start_time']} End: {item['end_time']}\n"
                    f"Excerpt: {item['text']}\n"
                )
            items_blob = "\n".join(items_lines).strip()

            formatted = self.titles_reasons_prompt.format_messages(items=items_blob)
            response = self.llm.invoke(formatted)
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()

            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()

            data = None
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    data = json.loads(content[json_start:json_end])
                except json.JSONDecodeError:
                    data = None

            if data is None:
                arr_start = content.find("[")
                arr_end = content.rfind("]") + 1
                if arr_start >= 0 and arr_end > arr_start:
                    try:
                        data = {"items": json.loads(content[arr_start:arr_end])}
                    except json.JSONDecodeError:
                        data = None

            if not data or "items" not in data or not isinstance(data["items"], list):
                return

            for item in data["items"]:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(output.shorts):
                    continue
                title = str(item.get("title", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if title:
                    output.shorts[idx].title = title
                if reason:
                    output.shorts[idx].reason = reason
        except Exception:
            return

    def _rank_and_spread(
        self,
        candidates: List[ShortClient],
        transcription_result: Dict[str, Any],
        target_shorts: int,
        min_gap_seconds: float
    ) -> List[ShortClient]:
        """Pick top clips by score while spreading them across the timeline."""
        segments = transcription_result.get("segments", [])
        if not candidates:
            return []

        if segments:
            min_time = min(s.get("start", 0) for s in segments)
            max_time = max(s.get("end", 0) for s in segments)
        else:
            min_time = min(c.start_time for c in candidates)
            max_time = max(c.end_time for c in candidates)

        span = max(1.0, max_time - min_time)
        bucket_size = span / max(1, target_shorts)

        # Keep best per bucket
        best_per_bucket = {}
        for c in candidates:
            mid = (c.start_time + c.end_time) / 2.0
            bucket = int((mid - min_time) / bucket_size)
            bucket = max(0, min(target_shorts - 1, bucket))
            prev = best_per_bucket.get(bucket)
            if prev is None or c.score > prev.score:
                best_per_bucket[bucket] = c

        selected = list(best_per_bucket.values())
        selected.sort(key=lambda s: s.start_time)

        # Fill remaining slots by score while enforcing spacing
        remaining = [c for c in candidates if c not in selected]
        remaining.sort(key=lambda s: s.score, reverse=True)

        def too_similar(a: ShortClient, b: ShortClient) -> bool:
            if abs(a.start_time - b.start_time) < 0.5 and abs(a.end_time - b.end_time) < 0.5:
                return True
            overlap_start = max(a.start_time, b.start_time)
            overlap_end = min(a.end_time, b.end_time)
            overlap = max(0.0, overlap_end - overlap_start)
            dur = max(0.1, min(a.end_time - a.start_time, b.end_time - b.start_time))
            return (overlap / dur) >= 0.85

        def far_enough(candidate: ShortClient) -> bool:
            c_mid = (candidate.start_time + candidate.end_time) / 2.0
            for s in selected:
                s_mid = (s.start_time + s.end_time) / 2.0
                if abs(c_mid - s_mid) < min_gap_seconds:
                    return False
            return True

        for c in remaining:
            if len(selected) >= target_shorts:
                break
            if far_enough(c):
                selected.append(c)

        # Second pass: if still short, allow lower-scored clips even if closer,
        # but avoid near-duplicate windows.
        if len(selected) < target_shorts:
            for c in remaining:
                if len(selected) >= target_shorts:
                    break
                if any(too_similar(c, s) for s in selected):
                    continue
                selected.append(c)

        selected.sort(key=lambda s: s.start_time)
        return selected[:target_shorts]
    
    def _split_transcript_by_time(
        self,
        transcription_result: Dict[str, Any],
        chunk_minutes: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Split a transcription result into time-based chunks.
        
        Args:
            transcription_result: Full transcription result
            chunk_minutes: Duration of each chunk in minutes
            
        Returns:
            List of transcription result dicts, each covering a time window
        """
        segments = transcription_result.get("segments", [])
        if not segments:
            return [transcription_result]
        
        chunk_seconds = chunk_minutes * 60.0
        
        # Find time boundaries
        min_time = min(s.get("start", 0) for s in segments)
        max_time = max(s.get("end", 0) for s in segments)
        
        import math
        num_chunks = max(1, math.ceil((max_time - min_time) / chunk_seconds))
        
        if num_chunks <= 1:
            return [transcription_result]
        
        chunks = []
        for i in range(num_chunks):
            window_start = min_time + i * chunk_seconds
            window_end = window_start + chunk_seconds
            
            # Filter segments that fall within this window
            chunk_segments = [
                s for s in segments
                if s.get("start", 0) >= window_start and s.get("start", 0) < window_end
            ]
            
            if not chunk_segments:
                continue
            
            chunk_text = " ".join(s.get("text", "") for s in chunk_segments)
            chunk_words = []
            for s in chunk_segments:
                chunk_words.extend(s.get("words", []))
            
            chunk_result = {
                "text": chunk_text,
                "segments": chunk_segments,
                "language": transcription_result.get("language", "unknown"),
                "language_probability": transcription_result.get("language_probability", 0.0),
            }
            if "words" in transcription_result:
                chunk_result["words"] = chunk_words
            
            chunks.append(chunk_result)
        
        return chunks if chunks else [transcription_result]
    
    def select_shorts_chunked(
        self,
        transcription_result: Dict[str, Any],
        brand_info: Optional[Dict[str, Any]] = None,
        chunk_minutes: float = 15.0,
        on_progress=None,
        target_shorts: int = 5,
        min_gap_seconds: float = 90.0
    ) -> ShortsOutput:
        """
        Select shorts by processing transcript in chunks, then merging results.
        
        This is used for large transcripts that exceed LLM context/memory limits.
        Each chunk is processed independently, then results are deduplicated.
        
        Args:
            transcription_result: Result from Transcriber or Diarizer
            brand_info: Optional brand information dict
            chunk_minutes: Size of each transcript chunk in minutes
            on_progress: Optional callback(chunk_index, total_chunks)
            target_shorts: Maximum number of shorts to return
            min_gap_seconds: Minimum spacing between clips by midpoint
            
        Returns:
            ShortsOutput with selected shorts
        """
        chunks = self._split_transcript_by_time(transcription_result, chunk_minutes)
        
        if len(chunks) <= 1:
            # Small enough for single call
            return self.select_shorts(
                transcription_result,
                brand_info,
                target_shorts=target_shorts,
                min_gap_seconds=min_gap_seconds
            )
        
        all_shorts: List[ShortClient] = []
        per_chunk_target = max(1, min(3, target_shorts))
        
        for idx, chunk in enumerate(chunks):
            if on_progress:
                on_progress(idx + 1, len(chunks))
            
            try:
                chunk_output = self.select_shorts(
                    chunk,
                    brand_info,
                    target_shorts=per_chunk_target,
                    min_gap_seconds=min_gap_seconds
                )
                all_shorts.extend(chunk_output.shorts)
            except Exception as e:
                print(f"  ? LLM chunk {idx + 1}/{len(chunks)} failed: {e}. Skipping...")
                continue
        
        if not all_shorts:
            refined = refine_shorts_output(
                ShortsOutput(shorts=[], total_shorts=0),
                transcription_result,
                min_len=15.0,
                max_len=60.0,
                max_shorts=max(target_shorts, 5),
                min_shorts=5,
            )
            return self._enrich_shorts(refined, transcription_result)
        
        # Deduplicate: remove shorts that overlap significantly
        all_shorts.sort(key=lambda s: s.start_time)
        deduped: List[ShortClient] = [all_shorts[0]]
        
        for short in all_shorts[1:]:
            prev = deduped[-1]
            # If this short overlaps >50% with the previous one, skip it
            overlap_start = max(prev.start_time, short.start_time)
            overlap_end = min(prev.end_time, short.end_time)
            overlap = max(0, overlap_end - overlap_start)
            short_duration = max(0.1, short.end_time - short.start_time)
            
            if overlap / short_duration < 0.5:
                deduped.append(short)
        
        final_shorts = self._rank_and_spread(
            deduped,
            transcription_result,
            target_shorts,
            min_gap_seconds
        )
        refined = refine_shorts_output(
            ShortsOutput(shorts=final_shorts, total_shorts=len(final_shorts)),
            transcription_result,
            min_len=15.0,
            max_len=60.0,
            max_shorts=max(target_shorts, 5),
            min_shorts=5,
        )
        
        return self._enrich_shorts(refined, transcription_result)

    def select_shorts_with_retry(
        self,
        transcription_result: Dict[str, Any],
        brand_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        use_chunking: bool = False,
        chunk_minutes: float = 15.0,
        on_progress=None,
        target_shorts: int = 5,
        min_gap_seconds: float = 90.0
    ) -> ShortsOutput:
        """
        Select shorts with retry logic. Supports chunked mode for large transcripts.
        
        Args:
            transcription_result: Result from Transcriber or Diarizer
            brand_info: Optional brand information dict
            max_retries: Maximum number of retries
            use_chunking: If True, split transcript into chunks for LLM
            chunk_minutes: Size of each LLM chunk in minutes
            on_progress: Optional callback for chunk progress
            target_shorts: Maximum number of shorts to return
            min_gap_seconds: Minimum spacing between clips by midpoint
            
        Returns:
            ShortsOutput with selected shorts
        """
        last_error = None
        
        select_fn = (
            (lambda: self.select_shorts_chunked(
                transcription_result,
                brand_info,
                chunk_minutes,
                on_progress,
                target_shorts=target_shorts,
                min_gap_seconds=min_gap_seconds
            ))
            if use_chunking
            else (lambda: self.select_shorts(
                transcription_result,
                brand_info,
                target_shorts=target_shorts,
                min_gap_seconds=min_gap_seconds
            ))
        )
        
        for attempt in range(max_retries + 1):
            try:
                return select_fn()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                else:
                    raise last_error
        
        raise last_error
        
        raise last_error
