import json
from app.llm_client import chat, get_response_content, get_tool_calls
from app.tools import TOOLS, TOOL_SCHEMAS
from app.sandbox import SandboxError

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


def run_agent(user_message: str) -> dict:
    """
    Run the agent loop: LLM decides tools → execute → return results → repeat.

    Returns dict with 'response' (final text) and 'tool_results' (list of tool executions).
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"🤖 Starting agent for query: {user_message[:100]}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    all_tool_results = []

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"📍 Iteration {iteration + 1}/{MAX_ITERATIONS}")

        # Call LLM
        response = chat(messages, tools=TOOL_SCHEMAS)
        assistant_message = response["choices"][0]["message"]

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            # No tools called, we're done
            final_response = assistant_message.get("content", "")
            logger.info(f"✅ Agent finished. Response: {final_response[:200]}...")
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

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result)
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

