from flask import Blueprint, request, jsonify, current_app
import re
import json

bp = Blueprint('api', __name__)


@bp.before_request
def log_request():
    current_app.logger.debug(f"Request: {request.method} {request.path}")
    if request.is_json:
        current_app.logger.debug(f"Payload keys: {list(request.get_json().keys())}")


def extract_conversation(messages: list[dict]) -> str:
    """
    Extract the full conversation from all messages.
    Returns a formatted string with context and conversation history.
    """
    conversation_parts = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            # Include system context
            conversation_parts.append(f"[System Context]\n{content}")
        elif role == 'user':
            # Check if this contains the separator
            parts = re.split(r'={5,}', content)
            if len(parts) > 1:
                # Has context header - extract conversation part
                conversation = parts[-1].strip()
                conversation = re.sub(r'^The conversation follows:\s*', '', conversation, flags=re.IGNORECASE)
                if conversation.strip():
                    conversation_parts.append(conversation.strip())
            else:
                # Regular message
                conversation_parts.append(content.strip())
        elif role == 'assistant':
            conversation_parts.append(f"[Assistant]: {content}")
    
    return "\n\n".join(conversation_parts)


@bp.route('/', methods=['POST'])
@bp.route('/query', methods=['POST'])
def query():
    """
    Main endpoint that receives requests and processes them.
    
    Expected format:
    {
      "model": "...",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "=== CONTEXT ===\n...\n=======\n\nActual message here"}
      ],
      ...
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    messages = data.get('messages', [])
    
    # Extract full conversation with context
    extracted_message = extract_conversation(messages)
    current_app.logger.info(f"Extracted conversation:\n{extracted_message[:500]}...")
    
    # Run the agent
    from app.executor import run_agent
    import time
    try:
        result = run_agent(extracted_message)
        current_app.logger.info(f"Agent response: {result['response'][:200]}...")
        current_app.logger.debug(f"Tool calls: {len(result['tool_results'])}")
        for tr in result['tool_results']:
            current_app.logger.debug(f"  Tool: {tr['tool']} -> {json.dumps(tr['result'])[:100]}")
        
        # Return OpenAI-compatible format
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "data-science-agent"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["response"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
    except Exception as e:
        current_app.logger.error(f"Agent error: {e}", exc_info=True)
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "data-science-agent"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                },
                "finish_reason": "stop"
            }]
        }), 500


@bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

