import json
import threading
import uuid
from datetime import datetime
from app.llm_client import chat, get_response_content, get_tool_calls
from app.tools import TOOLS, TOOL_SCHEMAS
from app.sandbox import SandboxError

# Thread-local storage for conversation context
# This allows tools to access the current conversation state
conversation_context = threading.local()

SYSTEM_PROMPT = """You are a data science assistant with access to tools for file operations and data analysis.
You work within a sandboxed workspace directory. All file paths are relative to this workspace.

CRITICAL RULES:
1. NEVER analyze raw data directly - ALWAYS use analytics tools (summary_stats, value_counts, word_frequency, etc.)
2. read_csv and read_json return only METADATA and SAMPLES (5 rows max) - they do NOT return full datasets
3. For JSON files: use convert_json_to_csv to convert the FULL file to CSV, then use analytics tools
4. Use specialized tools for every analysis task - you cannot process large datasets directly
5. If you need to analyze text, use text analysis tools (word_frequency, sentiment_analysis, topic_extraction)
6. If you need statistics, use summary_stats, correlation_matrix, or value_counts
7. ALL file paths must be relative to workspace (e.g. "data.csv" not "artefacts/data.csv")
8. You can export your own conversation history using export_conversation, then analyze it as data

When the user asks you to perform data tasks, use the available tools. Always explain what you're doing.
If a task requires multiple steps, execute them one at a time."""

MAX_ITERATIONS = 10


def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a tool by name with given arguments."""
    import logging
    logger = logging.getLogger(__name__)

    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}

    # Log tool execution
    logger.info(f"🔧 Executing tool: {name}")
    logger.info(f"   Arguments: {json.dumps(arguments, indent=2)}")

    try:
        result = TOOLS[name](**arguments)

        # Log result (truncate if too long)
        result_str = json.dumps(result, indent=2)
        if len(result_str) > 500:
            logger.info(f"   Result: {result_str[:500]}... (truncated)")
        else:
            logger.info(f"   Result: {result_str}")

        return result
    except SandboxError as e:
        error_result = {"error": f"Sandbox violation: {e}"}
        logger.error(f"   ❌ Sandbox error: {e}")
        return error_result
    except Exception as e:
        error_result = {"error": f"Tool error: {e}"}
        logger.error(f"   ❌ Tool error: {e}", exc_info=True)
        return error_result


def run_agent(user_message: str, conversation_history: list[dict] = None) -> dict:
    """
    Run the agent loop: LLM decides tools → execute → return results → repeat.

    Args:
        user_message: The current user query (may be a simple message or formatted conversation)
        conversation_history: Optional list of parsed message dicts with metadata
                            (username, role_tag, original_timestamp)

    Returns dict with 'response' (final text) and 'tool_results' (list of tool executions).
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"🤖 Starting agent for query: {user_message[:100]}...")

    # Initialize conversation context for this execution
    conversation_id = str(uuid.uuid4())
    start_time = datetime.now()

    # Build initial messages list
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT, "timestamp": start_time.isoformat()}
    ]

    # If we have parsed conversation history, add those messages first
    if conversation_history:
        logger.info(f"📚 Including {len(conversation_history)} messages from conversation history")
        messages.extend(conversation_history)
    else:
        # No parsed history - use the formatted message string
        messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": start_time.isoformat()
        })

    all_tool_results = []
    tool_usage = {}  # Track tool usage counts

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"📍 Iteration {iteration + 1}/{MAX_ITERATIONS}")

        # Update conversation context (accessible to tools)
        conversation_context.messages = messages
        conversation_context.conversation_id = conversation_id
        conversation_context.start_time = start_time
        conversation_context.iteration = iteration + 1
        conversation_context.tool_usage = tool_usage

        # Call LLM
        response = chat(messages, tools=TOOL_SCHEMAS)
        assistant_message = response["choices"][0]["message"]
        assistant_message["timestamp"] = datetime.now().isoformat()

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            # No tools called, we're done
            final_response = assistant_message.get("content", "")
            logger.info(f"✅ Agent finished. Response: {final_response[:200]}...")

            # Add final assistant message
            messages.append(assistant_message)

            return {
                "response": final_response,
                "tool_results": all_tool_results
            }

        logger.info(f"🔨 LLM requested {len(tool_calls)} tool call(s)")

        # Add assistant message with tool calls to history
        messages.append(assistant_message)

        # Execute each tool call
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            result = execute_tool(func_name, func_args)
            all_tool_results.append({
                "tool": func_name,
                "arguments": func_args,
                "result": result
            })

            # Track tool usage
            tool_usage[func_name] = tool_usage.get(func_name, 0) + 1

            # Add tool result to messages with timestamp
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "tool_name": func_name,  # Add tool name for easier analysis
                "content": json.dumps(result),
                "timestamp": datetime.now().isoformat()
            })

    # Max iterations reached
    logger.warning("⚠️  Max iterations reached. Task may be incomplete.")
    return {
        "response": "Max iterations reached. Task may be incomplete.",
        "tool_results": all_tool_results
    }


# Test
if __name__ == "__main__":
    print("Testing agent loop...")
    print("Query: List the files in the workspace")
    
    result = run_agent("List the files in the workspace")
    
    print(f"\nResponse: {result['response']}")
    print(f"\nTool calls: {len(result['tool_results'])}")
    for tr in result['tool_results']:
        print(f"  - {tr['tool']}: {tr['result']}")

