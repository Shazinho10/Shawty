"""LLM provider factory for multiple LLM backends"""

import os
from enum import Enum
from typing import Optional, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"
    OLLAMA = "ollama"


def get_llm_provider(
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    grok_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_model: str = "deepseek-r1:1.5b"
) -> tuple[BaseChatModel, LLMProvider]:
    """
    Get LLM provider based on available API keys.
    Priority: OpenAI > Anthropic > Grok > Ollama (default)
    
    Args:
        openai_key: OpenAI API key
        anthropic_key: Anthropic API key
        grok_key: Grok API key
        ollama_base_url: Ollama base URL (default: http://localhost:11434)
        ollama_model: Ollama model name (default: deepseek-r1:1.5b)
        
    Returns:
        Tuple of (LLM instance, provider type)
    """
    # Check environment variables if keys not provided
    openai_key = openai_key or os.getenv("OPENAI_API_KEY")
    anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    grok_key = grok_key or os.getenv("GROK_API_KEY")
    ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Priority order: OpenAI > Anthropic > Grok > Ollama
    if openai_key:
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Using cost-effective model
                api_key=openai_key,
                temperature=0.7
            )
            return llm, LLMProvider.OPENAI
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI: {e}")
    
    if anthropic_key:
        try:
            llm = ChatAnthropic(
                model="claude-3-haiku-20240307",  # Using cost-effective model
                api_key=anthropic_key,
                temperature=0.7
            )
            return llm, LLMProvider.ANTHROPIC
        except Exception as e:
            print(f"Warning: Failed to initialize Anthropic: {e}")
    
    if grok_key:
        try:
            # Grok via OpenAI-compatible API (xAI)
            llm = ChatOpenAI(
                model="grok-beta",
                api_key=grok_key,
                base_url="https://api.x.ai/v1",
                temperature=0.7
            )
            return llm, LLMProvider.GROK
        except Exception as e:
            print(f"Warning: Failed to initialize Grok: {e}")
    
    # Default to Ollama with DeepSeek
    try:
        llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7
        )
        return llm, LLMProvider.OLLAMA
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Ollama. Make sure Ollama is running and "
            f"the model '{ollama_model}' is available. Error: {e}"
        )
