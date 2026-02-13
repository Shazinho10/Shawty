"""Post-process and refine clip boundaries for better context and coherence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from ..models.output import ShortsOutput, ShortClient


@dataclass
class _Window:
    start: float
    end: float
    title: str
    reason: str
    score: int
    anchor_mid: float
    merged: bool = False


def refine_shorts_output(
    shorts_output: ShortsOutput,
    transcription_result: Dict[str, Any],
    min_len: float = 15.0,
    max_len: float = 60.0,
    pad: float = 1.5,
    merge_gap: float = 0.0,
    max_shorts: int | None = None,
    min_shorts: int | None = None,
) -> ShortsOutput:
    """
    Refine clip windows for better context by snapping to transcript segments,
    padding for coherence, merging near windows, and enforcing min/max duration.
    """
    segments = transcription_result.get("segments") or []
    if not segments:
        return shorts_output
    segments = sorted(segments, key=lambda s: s.get("start", 0.0))

    transcript_start = min(seg.get("start", 0.0) for seg in segments)
    transcript_end = max(seg.get("end", 0.0) for seg in segments)

    def clamp_time(t: float) -> float:
        return max(transcript_start, min(transcript_end, t))

    def snap_start(t: float) -> float:
        # Find segment that overlaps t, else nearest earlier segment start.
        for seg in segments:
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            if s <= t <= e:
                return s
        # nearest earlier
        earlier = [seg.get("start", 0.0) for seg in segments if seg.get("start", 0.0) <= t]
        return max(earlier) if earlier else transcript_start

    def snap_end(t: float) -> float:
        # Find segment that overlaps t, else nearest later segment end.
        for seg in segments:
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            if s <= t <= e:
                return e
        later = [seg.get("end", 0.0) for seg in segments if seg.get("end", 0.0) >= t]
        return min(later) if later else transcript_end

    def expand_and_snap(start: float, end: float) -> _Window:
        start = clamp_time(start - pad)
        end = clamp_time(end + pad)
        start = snap_start(start)
        end = snap_end(end)
        anchor_mid = (start + end) / 2.0
        return _Window(start=start, end=end, title="", reason="", score=0, anchor_mid=anchor_mid)

    windows: List[_Window] = []

    for short in shorts_output.shorts:
        s = float(short.start_time)
        e = float(short.end_time)
        if e <= s:
            continue
        expanded = expand_and_snap(s, e)
        expanded.title = short.title
        expanded.reason = short.reason
        expanded.score = int(short.score)
        expanded.anchor_mid = (s + e) / 2.0
        windows.append(expanded)

    if not windows and not min_shorts:
        return shorts_output

    def enforce_length(win: _Window) -> _Window:
        dur = win.end - win.start
        if dur > max_len:
            new_start = clamp_time(win.anchor_mid - max_len / 2.0)
            new_end = clamp_time(win.anchor_mid + max_len / 2.0)
            new_start = snap_start(new_start)
            new_end = snap_end(new_end)
            win.start = new_start
            win.end = new_end
        elif dur < min_len:
            new_start = clamp_time(win.anchor_mid - min_len / 2.0)
            new_end = clamp_time(win.anchor_mid + min_len / 2.0)
            new_start = snap_start(new_start)
            new_end = snap_end(new_end)
            win.start = new_start
            win.end = new_end
        # Final guard against overshooting max after snapping
        if win.end - win.start > max_len:
            win.end = clamp_time(win.start + max_len)
        # Final guard to ensure min length when transcript is long enough
        if (win.end - win.start) < min_len and (transcript_end - transcript_start) >= min_len:
            win.start = clamp_time(win.anchor_mid - min_len / 2.0)
            win.end = clamp_time(win.anchor_mid + min_len / 2.0)
        return win

    def is_similar(a: _Window, b: _Window) -> bool:
        if abs(a.start - b.start) < 0.5 and abs(a.end - b.end) < 0.5:
            return True
        overlap_start = max(a.start, b.start)
        overlap_end = min(a.end, b.end)
        overlap = max(0.0, overlap_end - overlap_start)
        dur = max(0.1, min(a.end - a.start, b.end - b.start))
        return (overlap / dur) >= 0.85

    if min_shorts is not None and len(windows) < min_shorts:
        span = max(1.0, transcript_end - transcript_start)
        for i in range(min_shorts):
            if len(windows) >= min_shorts:
                break
            base_mid = transcript_start + (i + 0.5) * (span / min_shorts)
            added = False
            for attempt in range(5):
                jitter = (attempt - 2) * (min_len * 0.35)
                mid = clamp_time(base_mid + jitter)
                start = mid - (min_len / 2.0)
                end = mid + (min_len / 2.0)
                expanded = expand_and_snap(start, end)
                expanded = enforce_length(expanded)
                if (expanded.end - expanded.start) < min_len:
                    continue
                if any(is_similar(expanded, w) for w in windows):
                    continue
                expanded.title = "Auto Clip"
                expanded.reason = "Auto-generated to meet minimum clip count."
                expanded.score = 0
                windows.append(expanded)
                added = True
                break
            if not added:
                expanded = expand_and_snap(base_mid - (min_len / 2.0), base_mid + (min_len / 2.0))
                expanded = enforce_length(expanded)
                if (expanded.end - expanded.start) >= min_len and not any(is_similar(expanded, w) for w in windows):
                    expanded.title = "Auto Clip"
                    expanded.reason = "Auto-generated to meet minimum clip count."
                    expanded.score = 0
                    windows.append(expanded)

    if not windows:
        return shorts_output

    # Sort and (optionally) merge overlapping/close windows
    windows.sort(key=lambda w: w.start)
    merged: List[_Window] = [windows[0]]

    for win in windows[1:]:
        cur = merged[-1]
        if merge_gap > 0 and win.start <= cur.end + merge_gap:
            cur.end = max(cur.end, win.end)
            cur.anchor_mid = (cur.anchor_mid + win.anchor_mid) / 2.0
            cur.merged = True
            continue
        merged.append(win)

    refined: List[_Window] = []
    for win in merged:
        win = enforce_length(win)
        if win.end - win.start >= min_len:
            if not any(is_similar(win, r) for r in refined):
                refined.append(win)

    if min_shorts is not None and len(refined) < min_shorts:
        span = max(1.0, transcript_end - transcript_start)
        for i in range(min_shorts):
            if len(refined) >= min_shorts:
                break
            base_mid = transcript_start + (i + 0.5) * (span / min_shorts)
            added = False
            for attempt in range(5):
                jitter = (attempt - 2) * (min_len * 0.35)
                mid = clamp_time(base_mid + jitter)
                start = mid - (min_len / 2.0)
                end = mid + (min_len / 2.0)
                expanded = expand_and_snap(start, end)
                expanded = enforce_length(expanded)
                if (expanded.end - expanded.start) < min_len:
                    continue
                if any(is_similar(expanded, r) for r in refined):
                    continue
                expanded.title = "Auto Clip"
                expanded.reason = "Auto-generated to meet minimum clip count."
                expanded.score = 0
                refined.append(expanded)
                added = True
                break
            if not added:
                expanded = expand_and_snap(base_mid - (min_len / 2.0), base_mid + (min_len / 2.0))
                expanded = enforce_length(expanded)
                if (expanded.end - expanded.start) >= min_len and not any(is_similar(expanded, r) for r in refined):
                    expanded.title = "Auto Clip"
                    expanded.reason = "Auto-generated to meet minimum clip count."
                    expanded.score = 0
                    refined.append(expanded)

    # Cap to requested count, preserve chronological order
    if max_shorts is not None:
        refined = refined[:max_shorts]

    final_shorts: List[ShortClient] = []
    for win in refined:
        final_shorts.append(
            ShortClient(
                title=win.title,
                start_time=round(win.start, 2),
                end_time=round(win.end, 2),
                reason=win.reason,
                score=win.score,
            )
        )

    return ShortsOutput(shorts=final_shorts, total_shorts=len(final_shorts))
