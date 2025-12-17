import requests
import config


def chat(messages: list[dict], tools: list[dict] = None) -> dict:
    """
    Send a chat completion request to the configured LLM provider.
    
    Provider is determined by LLM_PROVIDER env var:
    - "local" (default): Uses LM Studio at localhost:1234
    - "openai": Uses OpenAI API (requires OPENAI_API_KEY)
    - "anthropic": Uses Anthropic API (requires ANTHROPIC_API_KEY)
    - "bedrock": Uses AWS Bedrock (requires AWS credentials)
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        tools: Optional list of tool definitions for function calling
    
    Returns:
        The response dict from the API
    """
    provider = config.LLM_PROVIDER.lower()
    
    if provider == "anthropic":
        return _chat_anthropic(messages, tools)
    elif provider == "bedrock":
        return _chat_bedrock(messages, tools)
    else:
        # "local" or "openai" - both use OpenAI-compatible API
        return _chat_openai(messages, tools)


def _chat_openai(messages: list[dict], tools: list[dict] = None) -> dict:
    """OpenAI-compatible API (works with LM Studio, OpenAI, etc.)"""
    headers = {"Content-Type": "application/json"}
    
    if config.OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {config.OPENAI_API_KEY}"
    
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
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def _chat_anthropic(messages: list[dict], tools: list[dict] = None) -> dict:
    """Anthropic Claude API with response converted to OpenAI format."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    # Extract system message
    system = ""
    claude_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            claude_messages.append(msg)
    
    payload = {
        "model": config.LLM_MODEL,
        "max_tokens": 4096,
        "messages": claude_messages,
    }
    if system:
        payload["system"] = system
    
    if tools:
        # Convert OpenAI tool format to Anthropic format
        payload["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"]
            }
            for t in tools
        ]
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    result = response.json()
    
    # Convert Anthropic response to OpenAI format
    return _anthropic_to_openai_format(result)


def _chat_bedrock(messages: list[dict], tools: list[dict] = None) -> dict:
    """AWS Bedrock API using boto3."""
    import boto3
    import json
    
    # Priority: explicit credentials > named profile > default chain
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": config.AWS_REGION,
            "aws_access_key_id": config.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": config.AWS_SECRET_ACCESS_KEY,
        }
        if config.AWS_SESSION_TOKEN:
            client_kwargs["aws_session_token"] = config.AWS_SESSION_TOKEN
        client = boto3.client(**client_kwargs)
    elif config.AWS_PROFILE:
        session = boto3.Session(profile_name=config.AWS_PROFILE)
        client = session.client("bedrock-runtime", region_name=config.AWS_REGION)
    else:
        client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
    
    # Extract system message
    system = []
    bedrock_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system.append({"text": msg["content"]})
        else:
            bedrock_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
    
    params = {
        "modelId": config.LLM_MODEL,
        "messages": bedrock_messages,
        "inferenceConfig": {"maxTokens": 4096, "temperature": 0.7}
    }
    if system:
        params["system"] = system
    
    if tools:
        params["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "inputSchema": {"json": t["function"]["parameters"]}
                    }
                }
                for t in tools
            ]
        }
    
    response = client.converse(**params)
    
    # Convert to OpenAI format
    content = ""
    tool_calls = []
    for block in response.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            content = block["text"]
        elif "toolUse" in block:
            tool_calls.append({
                "id": block["toolUse"]["toolUseId"],
                "type": "function",
                "function": {
                    "name": block["toolUse"]["name"],
                    "arguments": json.dumps(block["toolUse"]["input"])
                }
            })
    
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    return {"choices": [{"message": message}]}


def _anthropic_to_openai_format(result: dict) -> dict:
    """Convert Anthropic response to OpenAI format for compatibility."""
    content = ""
    tool_calls = []
    
    for block in result.get("content", []):
        if block["type"] == "text":
            content = block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": __import__("json").dumps(block["input"])
                }
            })
    
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    return {"choices": [{"message": message}]}


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
    
    print(f"Testing LLM connection (provider: {config.LLM_PROVIDER})...")
    try:
        result = chat(test_messages)
        print(f"Response: {get_response_content(result)}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect. Check your LLM_BASE_URL or API key.")
    except Exception as e:
        print(f"ERROR: {e}")
