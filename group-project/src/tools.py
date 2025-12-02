from typing import Dict, Any
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
    return None


def browse_tool(url: str) -> str:
    """
    Browse a web page using BeautifulSoup.
    """
    return None

def answer_tool() -> bool:
    """
    Information gathered are enough to answer the user query.
    """
    return True

"""
Extra tools for Part II --  Realistic agent with multiple tools.

"""