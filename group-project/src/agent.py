import json
from tools import search_tool, browse_tool, answer_tool, get_tools_schema
from typing import Dict, Any, List, Optional
from utils import call_deepseek, load_config
from prompts import SEARCH_AGENT_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: Dict[str, str] = None):
        self.config = config or load_config()

    
def tools_execution(tool_name: str, tool_args: Dict[str, Any]) -> Any:
    if tool_name == "search":
        return search_tool(tool_args["query"])
    elif tool_name == "browse":
        return browse_tool(tool_args["url"])
    elif tool_name == "answer":
        return True
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def agent_loop(question: str, max_steps: int = 6, config: Optional[Dict[str, str]] = None) -> str:
    cfg = config or load_config()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SEARCH_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = get_tools_schema()
    for _ in range(max_steps):
        resp = call_deepseek(messages=messages, tools=tools, config=cfg)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            terminate = False
            assistant_tool_calls = []
            for tc in tool_calls:
                name = tc.function.name
                args_str = tc.function.arguments or "{}"
                assistant_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                })
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": assistant_tool_calls,
            })
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                
                # Execute tool to get result
                result = tools_execution(name, args)
                
                # Append tool output to messages (CRITICAL: Must occur for ALL tools, including 'answer')
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result if isinstance(result, str) else json.dumps(result),
                })

                if name == "answer":
                    terminate = True

            if terminate:

                final = call_deepseek(messages=messages, config=cfg)
                return final.choices[0].message.content
            continue
        content = msg.content or ""
        if content.strip():
            return content
    messages.append({"role": "assistant", "content": "Summarize findings and answer the question succinctly."})
    final = call_deepseek(messages=messages, config=cfg)
    return final.choices[0].message.content

def generate_no_search(question: str, config: Optional[Dict[str, str]] = None, temperature: float = 0.3, max_tokens: int = 512) -> str:
    cfg = config or load_config()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Answer concisely with a short factual string when possible.\nQuestion: {question}"},
    ]
    resp = call_deepseek(messages=messages, config=cfg, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content