import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from openai import OpenAI
import json

# Load environment variables
load_dotenv()

def load_config() -> Dict[str, str]:
    """Load API keys from environment variables."""
    return {
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_CHAT_MODEL": "deepseek-chat",
        "DEEPSEEK_REASONING_MODEL": "deepseek-reasoner"
    }

def call_deepseek(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] = None,
    config: Dict[str, str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    use_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Wrapper for DeepSeek API calls.
    """
    if config is None:
        config = load_config()
    
    client = OpenAI(
        api_key=config["DEEPSEEK_API_KEY"],
        base_url=config["DEEPSEEK_BASE_URL"]
    )
    
    kwargs = {
        "model": config["DEEPSEEK_CHAT_MODEL"] if not use_reasoning else config["DEEPSEEK_REASONING_MODEL"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if tools:
        kwargs["tools"] = tools
    
    response = client.chat.completions.create(**kwargs)
    return response

