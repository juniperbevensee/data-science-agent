import os
import json
import shutil
import pandas as pd
from app.sandbox import resolve_path, list_files as sandbox_list_files, WORKSPACE_PATH


def read_csv(path: str, **kwargs) -> dict:
    """
    Read CSV file metadata and sample. Returns summary info, NOT full data.
    Use analytics tools (summary_stats, value_counts, etc.) to analyze the data.
    """
    full_path = resolve_path(path)
    df = pd.read_csv(full_path, **kwargs)

    # Return only metadata and a small sample
    return {
        "success": True,
        "path": path,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(5).to_dict(orient="records"),
        "message": f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns. Use analytics tools to analyze this data."
    }


def read_json(path: str) -> dict:
    """
    Read JSON file metadata and sample. Returns summary info, NOT full data.
    For large files, convert to CSV first then use analytics tools.
    """
    full_path = resolve_path(path)
    with open(full_path, 'r') as f:
        data = json.load(f)

    # Determine structure
    is_array = isinstance(data, list)
    is_object = isinstance(data, dict)

    if is_array:
        # Array of objects - return metadata and sample
        sample_size = min(5, len(data))

        # Get keys from first object if available
        keys = list(data[0].keys()) if len(data) > 0 and isinstance(data[0], dict) else []

        return {
            "success": True,
            "path": path,
            "type": "array",
            "count": len(data),
            "keys": keys,
            "sample": data[:sample_size],
            "message": f"Loaded JSON array with {len(data)} items. Use convert_json_to_csv to convert the FULL file to CSV, then use analytics tools."
        }
    elif is_object:
        # Object - return structure info
        return {
            "success": True,
            "path": path,
            "type": "object",
            "keys": list(data.keys()),
            "sample": {k: (v[:3] if isinstance(v, list) else v) for k, v in list(data.items())[:10]},
            "message": "Loaded JSON object. Extract relevant data and convert to CSV for analysis."
        }
    else:
        # Primitive type
        return {
            "success": True,
            "path": path,
            "type": type(data).__name__,
            "data": data
        }


def write_csv(path: str, data: list[dict], **kwargs) -> dict:
    """Write data to a CSV file."""
    full_path = resolve_path(path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True) if os.path.dirname(full_path) else None
    df = pd.DataFrame(data)
    df.to_csv(full_path, index=False, **kwargs)
    return {"success": True, "path": path, "rows": len(df)}


def write_json(path: str, data) -> dict:
    """Write data to a JSON file."""
    full_path = resolve_path(path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True) if os.path.dirname(full_path) else None
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)
    return {"success": True, "path": path}


def convert_json_to_csv(
    json_path: str,
    csv_path: str,
    extract_path: str = None
) -> dict:
    """
    Convert FULL JSON file to CSV format. Handles arrays and nested objects.

    Use this instead of write_csv when you need to convert a JSON file to CSV.
    This loads and converts the ENTIRE JSON file, not just a sample.

    Args:
        json_path: Path to input JSON file
        csv_path: Path for output CSV file
        extract_path: Optional dot-notation path to extract nested data (e.g. "results")
    """
    full_json_path = resolve_path(json_path)

    with open(full_json_path, 'r') as f:
        data = json.load(f)

    # Extract nested data if path provided
    if extract_path:
        parts = extract_path.split('.')
        for part in parts:
            if isinstance(data, dict):
                data = data.get(part, [])
            elif isinstance(data, list) and part.isdigit():
                data = data[int(part)]
            else:
                return {"success": False, "error": f"Cannot extract '{part}' from {type(data).__name__}"}

    # Convert to DataFrame
    if isinstance(data, list):
        if len(data) == 0:
            return {"success": False, "error": "Empty data array"}

        if isinstance(data[0], dict):
            # List of objects - flatten nested structures
            df = pd.json_normalize(data)
        else:
            # List of primitives
            df = pd.DataFrame({'value': data})
    elif isinstance(data, dict):
        # Single object or nested object - normalize it
        df = pd.json_normalize([data])
    else:
        return {"success": False, "error": f"Cannot convert {type(data).__name__} to CSV"}

    # Write to CSV
    full_csv_path = resolve_path(csv_path)
    os.makedirs(os.path.dirname(full_csv_path), exist_ok=True) if os.path.dirname(full_csv_path) else None
    df.to_csv(full_csv_path, index=False)

    return {
        "success": True,
        "input": json_path,
        "output": csv_path,
        "rows": len(df),
        "columns": list(df.columns)[:10],  # Show first 10 columns
        "total_columns": len(df.columns),
        "message": f"Converted {len(df)} rows to CSV. Use analytics tools to analyze this data."
    }


def copy_file(src: str, dst: str) -> dict:
    """Copy a file within the sandbox."""
    src_path = resolve_path(src)
    dst_path = resolve_path(dst)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True) if os.path.dirname(dst_path) else None
    shutil.copy2(src_path, dst_path)
    return {"success": True, "src": src, "dst": dst}


def list_files(subdir: str = "") -> dict:
    """List files in the workspace."""
    files = sandbox_list_files(subdir)
    return {"success": True, "files": files, "directory": subdir or "."}


def file_info(path: str) -> dict:
    """Get info about a file."""
    full_path = resolve_path(path)
    if not os.path.exists(full_path):
        return {"success": False, "error": f"File not found: {path}"}

    stat = os.stat(full_path)
    return {
        "success": True,
        "path": path,
        "size_bytes": stat.st_size,
        "is_file": os.path.isfile(full_path),
        "is_dir": os.path.isdir(full_path)
    }


def export_conversation(output_path: str, include_system: bool = False) -> dict:
    """
    Export the current conversation history to JSON file.

    Saves all messages from the current agent execution including:
    - User messages
    - Assistant responses with tool calls
    - Tool execution results
    - Timestamps and metadata

    Once exported, you can analyze the conversation using existing tools:
    - convert_json_to_csv to convert messages to CSV
    - value_counts to analyze message roles or tool usage
    - text_analysis tools to analyze message content
    - plot_bar to visualize tool usage patterns

    Args:
        output_path: Path for JSON file (e.g. "conversation.json")
        include_system: Whether to include system prompt (default False)
    """
    from app.executor import conversation_context
    from datetime import datetime

    # Check if conversation context is available
    if not hasattr(conversation_context, 'messages'):
        return {
            "success": False,
            "error": "No active conversation context. This tool must be called during an agent execution."
        }

    messages = conversation_context.messages

    # Filter out system message if requested
    if not include_system:
        messages = [msg for msg in messages if msg.get("role") != "system"]

    # Build conversation export with metadata
    conversation_data = {
        "conversation_id": conversation_context.conversation_id,
        "start_time": conversation_context.start_time.isoformat(),
        "export_time": datetime.now().isoformat(),
        "iteration": conversation_context.iteration,
        "message_count": len(messages),
        "tool_usage": conversation_context.tool_usage,
        "messages": messages
    }

    # Include conversation context metadata if available
    if hasattr(conversation_context, 'metadata') and conversation_context.metadata:
        conversation_data["metadata"] = conversation_context.metadata

    # Write to JSON file
    full_path = resolve_path(output_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True) if os.path.dirname(full_path) else None
    with open(full_path, 'w') as f:
        json.dump(conversation_data, f, indent=2)

    # Generate summary
    role_counts = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "success": True,
        "path": output_path,
        "conversation_id": conversation_context.conversation_id,
        "messages_exported": len(messages),
        "role_breakdown": role_counts,
        "tools_used": conversation_context.tool_usage,
        "message": f"Exported {len(messages)} messages. Use convert_json_to_csv with extract_path='messages' to analyze the conversation data."
    }


# Tool definitions for LLM
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_csv",
            "description": "Get CSV file metadata and small sample (5 rows). Returns row count, columns, data types, and sample. Does NOT return full data - use analytics tools for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to CSV file relative to workspace"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_json",
            "description": "Get JSON file metadata and small sample. Returns structure info and sample only. To analyze JSON data, use convert_json_to_csv to convert the FULL file to CSV first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to JSON file relative to workspace"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_csv",
            "description": "Write data to a CSV file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path for output CSV file"},
                    "data": {"type": "array", "items": {"type": "object"}, "description": "Array of row objects"}
                },
                "required": ["path", "data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_json",
            "description": "Write data to a JSON file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path for output JSON file"},
                    "data": {"description": "Data to write"}
                },
                "required": ["path", "data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_json_to_csv",
            "description": "Convert FULL JSON file to CSV. Use this to load and convert entire JSON files for analysis. Handles nested objects and arrays. This loads ALL data, not just a sample.",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_path": {"type": "string", "description": "Path to input JSON file"},
                    "csv_path": {"type": "string", "description": "Path for output CSV file"},
                    "extract_path": {"type": "string", "description": "Optional dot-notation path to extract nested data (e.g. 'results' or 'data.items')"}
                },
                "required": ["json_path", "csv_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file within the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "Source file path"},
                    "dst": {"type": "string", "description": "Destination file path"}
                },
                "required": ["src", "dst"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Subdirectory to list (optional)", "default": ""}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_info",
            "description": "Get information about a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_conversation",
            "description": "Export the current conversation history to JSON file with full metadata. Includes all messages (user, assistant, tool results), timestamps, tool usage stats, and conversation ID. Once exported, use convert_json_to_csv and analytics tools to analyze conversation patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Path for JSON file (e.g. 'conversation.json')"},
                    "include_system": {"type": "boolean", "description": "Whether to include system prompt in export (default false)", "default": False}
                },
                "required": ["output_path"]
            }
        }
    }
]

# Map function names to implementations
TOOLS = {
    "read_csv": read_csv,
    "read_json": read_json,
    "write_csv": write_csv,
    "write_json": write_json,
    "convert_json_to_csv": convert_json_to_csv,
    "copy_file": copy_file,
    "list_files": list_files,
    "file_info": file_info,
    "export_conversation": export_conversation,
}


# Test
if __name__ == "__main__":
    # Create test data
    test_data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"},
    ]
    
    print("Testing file tools...")
    
    # Write CSV
    result = write_csv("test.csv", test_data)
    print(f"write_csv: {result}")
    
    # Read CSV
    result = read_csv("test.csv")
    print(f"read_csv: {result}")
    
    # Write JSON
    result = write_json("test.json", test_data)
    print(f"write_json: {result}")
    
    # Read JSON
    result = read_json("test.json")
    print(f"read_json: {result}")
    
    # Copy file
    result = copy_file("test.csv", "test_copy.csv")
    print(f"copy_file: {result}")
    
    # List files
    result = list_files()
    print(f"list_files: {result}")
    
    # File info
    result = file_info("test.csv")
    print(f"file_info: {result}")

