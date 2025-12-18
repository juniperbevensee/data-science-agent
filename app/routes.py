from flask import Blueprint, request, jsonify, current_app
import re
import json
from datetime import datetime

bp = Blueprint('api', __name__)


@bp.before_request
def log_request():
    current_app.logger.debug(f"Request: {request.method} {request.path}")
    if request.is_json:
        current_app.logger.debug(f"Payload keys: {list(request.get_json().keys())}")


def parse_conversation_messages(content: str) -> list[dict]:
    """
    Parse formatted conversation text into individual message objects.

    Recognizes patterns like:
    [username (role) at timestamp]:
    message content

    [Assistant]:
    response content

    Returns list of message dicts with metadata extracted.
    """
    messages = []

    # Pattern to match: [username (role) at ISO-timestamp]:
    user_pattern = r'\[([^\]]+?)\s+\(([^\)]+)\)\s+at\s+([^\]]+)\]:\s*'
    # Pattern to match: [Assistant]:
    assistant_pattern = r'\[Assistant\]:\s*'

    # Split content by message boundaries
    # First, find all user message starts
    user_matches = list(re.finditer(user_pattern, content))
    assistant_matches = list(re.finditer(assistant_pattern, content))

    # Combine and sort all match positions
    all_matches = []
    for match in user_matches:
        all_matches.append(('user', match.start(), match.end(), match))
    for match in assistant_matches:
        all_matches.append(('assistant', match.start(), match.end(), match))

    all_matches.sort(key=lambda x: x[1])  # Sort by start position

    # Extract messages
    for i, (role, start, end, match) in enumerate(all_matches):
        # Find where this message ends (start of next message or end of string)
        if i + 1 < len(all_matches):
            content_end = all_matches[i + 1][1]
        else:
            content_end = len(content)

        # Extract message content
        message_content = content[end:content_end].strip()

        if role == 'user':
            # Extract username, role tag, and timestamp
            username = match.group(1).strip()
            role_tag = match.group(2).strip()
            timestamp_str = match.group(3).strip()

            messages.append({
                'role': 'user',
                'content': message_content,
                'username': username,
                'role_tag': role_tag,
                'original_timestamp': timestamp_str,
                'timestamp': datetime.now().isoformat()
            })
        elif role == 'assistant':
            # Historical assistant messages are TEXT ONLY - no tool_calls
            # (we don't have the tool results, and don't want to re-execute)
            messages.append({
                'role': 'assistant',
                'content': message_content,
                'timestamp': datetime.now().isoformat()
                # Explicitly NO tool_calls - this is just conversational context
            })

    return messages


def parse_conversation_metadata(content: str) -> dict:
    """
    Parse conversation context metadata from the header block.

    Extracts fields like:
    Biome: Example Project - Optional description
    Biome ID: biome-abc123...
    Thread: Conversation Name
    Thread ID: thread-xyz789...
    Participants: 6 members

    Returns dict with extracted metadata fields.
    """
    metadata = {}

    # Pattern to match "Field: Value" lines
    field_pattern = r'^([^:]+):\s*(.+)$'

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        match = re.match(field_pattern, line)
        if match:
            field_name = match.group(1).strip()
            field_value = match.group(2).strip()

            if field_name == 'Biome':
                # Split biome name and description (separated by ' - ')
                if ' - ' in field_value:
                    biome_parts = field_value.split(' - ', 1)
                    metadata['biome_name'] = biome_parts[0].strip()
                    metadata['biome_description'] = biome_parts[1].strip()
                else:
                    metadata['biome_name'] = field_value
            elif field_name == 'Biome ID':
                metadata['biome_id'] = field_value
            elif field_name == 'Thread':
                metadata['thread_name'] = field_value
            elif field_name == 'Thread ID':
                metadata['thread_id'] = field_value
            elif field_name == 'Participants':
                metadata['participants'] = field_value

    return metadata


def extract_conversation(messages: list[dict]) -> tuple[str, list[dict], dict]:
    """
    Extract the full conversation from all messages.
    Returns:
        - A formatted string with context and conversation history (for backward compatibility)
        - A list of parsed message objects with metadata
        - A dict of conversation metadata (biome, thread, participants, etc.)
    """
    conversation_parts = []
    parsed_messages = []
    conversation_metadata = {}

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
                # Has context header - extract metadata and conversation
                context_header = parts[0] if len(parts) > 0 else ""

                # Parse metadata from context header
                if context_header.strip():
                    metadata = parse_conversation_metadata(context_header)
                    if metadata:
                        conversation_metadata.update(metadata)
                        current_app.logger.info(f"📋 Extracted conversation metadata: {metadata}")

                # Extract conversation part
                conversation = parts[-1].strip()
                conversation = re.sub(r'^The conversation follows:\s*', '', conversation, flags=re.IGNORECASE)
                if conversation.strip():
                    conversation_parts.append(conversation.strip())
                    # Try to parse into individual messages
                    parsed_messages.extend(parse_conversation_messages(conversation))
            else:
                # Regular message - try to parse it too
                conversation_parts.append(content.strip())
                parsed_messages.extend(parse_conversation_messages(content))
        elif role == 'assistant':
            conversation_parts.append(f"[Assistant]: {content}")
            # Add assistant messages to parsed history
            parsed_messages.append({
                'role': 'assistant',
                'content': content,
                'timestamp': datetime.now().isoformat()
            })
        elif role == 'tool':
            # Also preserve tool messages in conversation history
            conversation_parts.append(f"[Tool Result]: {content}")
            parsed_messages.append({
                'role': 'tool',
                'content': content,
                'tool_call_id': msg.get('tool_call_id'),
                'timestamp': datetime.now().isoformat()
            })

    formatted_string = "\n\n".join(conversation_parts)
    return formatted_string, parsed_messages, conversation_metadata


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

    # Extract full conversation with context and parse into individual messages
    extracted_message, parsed_messages, conversation_metadata = extract_conversation(messages)
    current_app.logger.info(f"Extracted conversation:\n{extracted_message[:500]}...")
    if parsed_messages:
        current_app.logger.info(f"📝 Parsed {len(parsed_messages)} individual messages from conversation history")
    if conversation_metadata:
        current_app.logger.info(f"📋 Conversation metadata: {conversation_metadata}")

    # Run the agent with full conversation history
    # The SYSTEM_PROMPT instructs the agent to use history for CONTEXT only
    # and to execute tasks ONLY from the latest user message
    from app.executor import run_agent
    import time
    try:
        current_app.logger.info("=" * 60)
        current_app.logger.info("🚀 AGENT REQUEST")
        current_app.logger.info("=" * 60)

        # Pass full conversation history for context
        # System prompt rules 9-13 prevent re-execution of historical tasks
        if parsed_messages:
            current_app.logger.info(f"📜 Sending {len(parsed_messages)} messages as conversation context")
            result = run_agent(
                extracted_message,
                conversation_history=parsed_messages,
                conversation_metadata=conversation_metadata
            )
        else:
            # No parsed history - just send the message
            result = run_agent(
                extracted_message,
                conversation_history=None,
                conversation_metadata=conversation_metadata
            )

        current_app.logger.info("=" * 60)
        current_app.logger.info(f"✅ AGENT COMPLETE - {len(result['tool_results'])} tool(s) executed")
        current_app.logger.info("=" * 60)

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

