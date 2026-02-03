"""Langchain agent for short selection"""

import json
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from ..models.output import ShortsOutput
from .prompts import get_shorts_selection_prompt, format_transcript_for_llm, format_brand_context


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
    
    def select_shorts(
        self,
        transcription_result: Dict[str, Any],
        brand_info: Optional[Dict[str, Any]] = None
    ) -> ShortsOutput:
        """
        Select YouTube shorts from transcript using LLM.
        
        Args:
            transcription_result: Result from Transcriber or Diarizer
            brand_info: Optional brand information dict
            
        Returns:
            ShortsOutput with selected shorts
        """
        # Format transcript for LLM
        transcript_text = format_transcript_for_llm(transcription_result)
        
        # Format brand context
        brand_context = format_brand_context(brand_info) if brand_info else ""
        
        # Format the prompt
        formatted_prompt = self.prompt.format_messages(
            transcript=transcript_text,
            brand_context=brand_context
        )
        
        try:
            # Invoke LLM with structured output
            if hasattr(self.llm, 'with_structured_output'):
                # Use structured output if available (newer Langchain versions)
                structured_llm = self.llm.with_structured_output(ShortsOutput)
                result = structured_llm.invoke(formatted_prompt)
                return result
            else:
                # Fallback: use output parser
                response = self.llm.invoke(formatted_prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                result = self.output_parser.parse(content)
                return result
        except Exception as e:
            # If structured output fails, try to parse JSON manually
            try:
                response = self.llm.invoke(formatted_prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Try to find JSON in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    return ShortsOutput(**data)
                else:
                    raise ValueError("No JSON found in LLM response")
            except Exception as parse_error:
                raise RuntimeError(
                    f"Failed to parse LLM output. Original error: {e}, "
                    f"Parse error: {parse_error}"
                )
    
    def select_shorts_with_retry(
        self,
        transcription_result: Dict[str, Any],
        brand_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> ShortsOutput:
        """
        Select shorts with retry logic.
        
        Args:
            transcription_result: Result from Transcriber or Diarizer
            brand_info: Optional brand information dict
            max_retries: Maximum number of retries
            
        Returns:
            ShortsOutput with selected shorts
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.select_shorts(transcription_result, brand_info)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                else:
                    raise last_error
        
        raise last_error
