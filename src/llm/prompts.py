"""Langchain prompts for short selection"""

from langchain_core.prompts import ChatPromptTemplate


def get_shorts_selection_prompt() -> ChatPromptTemplate:
    """Get the prompt template for selecting YouTube shorts from transcript"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing video transcripts and selecting the best segments for YouTube Shorts.

Your task is to:
1. Analyze the provided transcript with timestamps
2. Identify up to 5 engaging segments that would work well as YouTube Shorts
3. Each segment should be between 15-60 seconds long
4. Create compelling titles for each short
5. Explain why each segment was selected

Consider:
- Hook value (does it grab attention quickly?)
- Completeness (does it tell a complete story or make a complete point?)
- Engagement potential (will viewers want to watch?)
- Visual/audio quality indicators from the transcript
- Brand alignment (if brand information is provided)

Output format: You must return a valid JSON response with this exact structure:
{{
  "shorts": [
    {{
      "title": "Short title here",
      "start_time": 10.5,
      "end_time": 45.2,
      "reason": "Explanation of why this segment was selected"
    }}
  ],
  "total_shorts": 1
}}

Requirements:
- shorts: Array of up to 5 short segments (maximum)
- Each short must have: title (string), start_time (float in seconds), end_time (float in seconds), reason (string)
- total_shorts: Number of shorts selected (must match array length, max 5)
- start_time must be less than end_time
- Each segment should be 15-60 seconds long

Important: Only select segments that are truly engaging and suitable for shorts. Quality over quantity. Return ONLY valid JSON, no additional text."""),
        ("human", """Transcript with timestamps:
{transcript}

{brand_context}

Please analyze this transcript and select the best segments for YouTube Shorts. Return ONLY a valid JSON object matching the specified format.""")
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
