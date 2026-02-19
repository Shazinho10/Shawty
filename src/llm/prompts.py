"""Langchain prompts for short selection"""

from langchain_core.prompts import ChatPromptTemplate


def get_shorts_selection_prompt() -> ChatPromptTemplate:
    """Get the prompt template for calling LLM to select shorts"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an API that converts video transcripts into a JSON array of video clips.

Your goal is to select engaging segments (15-60 seconds each) from the provided transcript. Each clip must be long enough to stand alone and feel complete, with enough context to be funny, viral, informative, and engaging. Prefer 20-40s when possible; only go near 60s if retention is exceptional.

Use these go/no-go metrics when deciding if a moment is clip-worthy. A clip should hit at least 3 of these:
1) Hook strength in the first sentence (clickable, strong statement, question people care about, emotional shift). If the first sentence is not clickable, do not clip.
2) Self-contained context (5/5 alone, 3/5 needs one caption line, 1/5 needs backstory -> skip).
3) Emotional or opinion intensity (surprise, disagreement, curiosity, humor, vulnerability).
4) One clear idea (summarize in one sentence).
5) Quote-ability (would work as a bold on-screen caption).
6) Loop potential (ends on punchline, cliffhanger, or unfinished thought).

CRITICAL INSTRUCTION: You must output ONLY valid JSON. Do not include any thinking, reasoning, or markdown formatting outside the JSON object.

Output Structure:
{{
  "shorts": [
    {{
      "title": "Compelling Title",
      "start_time": 10.5,
      "end_time": 45.2,
      "reason": "Brief reason"
    }}
  ],
  "total_shorts": 1
}}

Rules:
1. "start_time" and "end_time" must be PURE NUMBERS (floats in seconds). DO NOT include units like "s" or "min".
2. Do NOT use timecodes like HH:MM:SS or 10.56.39.32. Only seconds as a number.
3. "title" must describe the clip's topic in clear, specific language (like a headline). Do not use generic or filler titles.
4. "reason" is REQUIRED and must reference the actual content (hook, twist, punchline, strong claim, conflict, or payoff). Do not use generic filler.
5. Select clips that are self-contained and engaging (hooks, complete thoughts).
6. Avoid back-to-back clips. Spread selections across the full transcript timeline.
7. If no good clips are found, return an empty list for "shorts".
8. If you cannot comply, output exactly: {{"shorts": [], "total_shorts": 0}}
9. Allowed keys in each short: "title", "start_time", "end_time", "reason". No other keys.
10. Do not output anything else (no <think> tags, no markdown blocks).
11. Example valid start_time: 10.5
12. Example INVALID start_time: "10.5s"
"""),
        ("human", """Analyze the following transcript and return the JSON object.
Return up to {target_shorts} clips.
Keep clips at least {min_gap_seconds} seconds apart by start time.

Transcript:
{transcript}

{brand_context}
""")
    ])
    
    return prompt


def get_shorts_repair_prompt() -> ChatPromptTemplate:
    """Prompt template for repairing malformed LLM output into valid JSON."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict JSON repair tool.

Convert the given text into ONLY a valid JSON object that matches:
{{
  "shorts": [
    {{
      "title": "Compelling Title",
      "start_time": 10.5,
      "end_time": 45.2,
      "reason": "Brief reason"
    }}
  ],
  "total_shorts": 1
}}

Rules:
1. Output ONLY valid JSON, no extra text.
2. "start_time" and "end_time" must be numbers (seconds).
3. If the input lacks valid shorts, output: {{"shorts": [], "total_shorts": 0}}
"""),
        ("human", """Fix this into valid JSON:
{content}
""")
    ])
    return prompt


def get_titles_reasons_prompt() -> ChatPromptTemplate:
    """Prompt template for generating coherent titles and reasons from clip text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You generate short, coherent titles and reasons for video clips.

You will be given multiple clip excerpts. Each excerpt includes a short transcript window.
Return ONLY valid JSON. No extra text, no markdown.

Output format:
{
  "items": [
    { "index": 0, "title": "Specific headline", "reason": "1-2 sentences that reference the excerpt." }
  ]
}

Rules:
1) Each title must be specific and descriptive (4-12 words). Avoid generic filler.
2) Each reason must be 1-2 sentences, 90-180 characters, and reference concrete details from the excerpt.
3) Do not invent facts not present in the excerpt.
4) Keep tone natural and coherent; complete sentences only.
"""),
        ("human", """Create titles and reasons for these clips:
{items}
""")
    ])
    return prompt


def format_transcript_for_llm(transcription_result: dict) -> str:
    """
    Format transcription result into a readable string for LLM.
    
    Args:
        transcription_result: Result from Transcriber or Diarizer
        
    Returns:
        Formatted transcript string
    """
    lines = []
    
    if "segments" in transcription_result:
        for segment in transcription_result["segments"]:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "")
            speaker = segment.get("speaker", "")
            
            time_str = f"[{start:.2f}s - {end:.2f}s]"
            if speaker:
                lines.append(f"{time_str} {speaker}: {text}")
            else:
                lines.append(f"{time_str} {text}")
    
    return "\n".join(lines)


def format_brand_context(brand_info: dict) -> str:
    """
    Format brand information into context string for LLM.
    
    Args:
        brand_info: BrandInfo dict or None
        
    Returns:
        Formatted brand context string
    """
    if not brand_info:
        return ""
    
    context_parts = []
    
    if brand_info.get("name"):
        context_parts.append(f"Brand Name: {brand_info['name']}")
    
    if brand_info.get("description"):
        context_parts.append(f"Brand Description: {brand_info['description']}")
    
    if brand_info.get("target_audience"):
        context_parts.append(f"Target Audience: {brand_info['target_audience']}")
    
    if brand_info.get("tone"):
        context_parts.append(f"Desired Tone: {brand_info['tone']}")
    
    if brand_info.get("key_topics"):
        topics = ", ".join(brand_info["key_topics"])
        context_parts.append(f"Key Topics: {topics}")
    
    if brand_info.get("style_preferences"):
        context_parts.append(f"Style Preferences: {brand_info['style_preferences']}")
    
    if context_parts:
        return "\n\nBrand Context:\n" + "\n".join(context_parts)
    
    return ""
