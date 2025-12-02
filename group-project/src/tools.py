from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
from utils import load_config
"""
Tool definitions and execution functions.
========================================
Tools:
- search: Search the web using Google Search via Serper API.
- browse: Browse a web page using BeautifulSoup.
- answer: Information gathered are enough to answer the user query. 
"""
def get_tools_schema() -> Dict[str, Any]:
    """Get schema for search tool for DeepSeek function calling format."""
    return [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web using Google Search via Serper API. Use this to find current information, facts, or recent events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {"type": "function",
        "function": {
            "name": "browse",
            "description": "Fetch and extract text content from a web page URL. Use this when search results only provide snippets and you need full page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to browse"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Information gathered are enough to answer the user query. Use this when you have enough information to answer the user query.",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            }
        }
    }
    ]


def search_tool(query: str) -> str:
    """
    Search the web using Google Search via Serper API.
    """
    config = load_config()
    api_key = config.get("SERPER_API_KEY")
    if not api_key:
        return "[search] Missing SERPER_API_KEY"
    try:
        payload = {"q": query}
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        resp = requests.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            return f"[search] HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        organic = data.get("organic", [])
        lines = []
        for i, item in enumerate(organic[:5], start=1):
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            link = item.get("link") or ""
            lines.append(f"{i}. {title}\n{snippet}\nURL: {link}")
        return "\n\n".join(lines) if lines else "[search] No results"
    except Exception as e:
        return f"[search] Error: {e}"


def browse_tool(url: str) -> str:
    """
    Browse a web page using BeautifulSoup.
    """
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return f"[browse] HTTP {resp.status_code}: {resp.text[:200]}"
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs)
        return text[:5000] if text else "[browse] No textual content"
    except Exception as e:
        return f"[browse] Error: {e}"

def answer_tool() -> bool:
    """
    Information gathered are enough to answer the user query.
    """
    return True

"""
Extra tools for Part II --  Realistic agent with multiple tools.

"""