import json
from .tools import search_tool, browse_tool, answer_tool
from typing import Dict, Any, List, Optional
from .utils import call_deepseek, load_config

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: Dict[str, str] = None):
        self.config = config or load_config()

    
def tools_execution(tool_name: str, tool_args: Dict[str, Any]) -> Any:
    if tool_name == "search":
        return search_tool(tool_args["query"])
    elif tool_name == "browse":
        return browse_tool(tool_args["url"])
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def agent_loop(question: str) -> str:
    return None