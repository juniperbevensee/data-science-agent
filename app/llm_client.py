import requests
import config


def chat(messages: list[dict], tools: list[dict] = None) -> dict:
    """
    Send a chat completion request to LM Studio.
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        tools: Optional list of tool definitions for function calling
    
    Returns:
        The response dict from the API
    """
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    response = requests.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def get_response_content(response: dict) -> str:
    """Extract the text content from an LLM response."""
    return response["choices"][0]["message"]["content"] or ""


def get_tool_calls(response: dict) -> list[dict]:
    """Extract tool calls from an LLM response, if any."""
    message = response["choices"][0]["message"]
    return message.get("tool_calls", [])


# Simple test
if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
    
    print("Testing LLM connection...")
    try:
        result = chat(test_messages)
        print(f"Response: {get_response_content(result)}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to LM Studio. Is it running on port 1234?")
    except Exception as e:
        print(f"ERROR: {e}")

