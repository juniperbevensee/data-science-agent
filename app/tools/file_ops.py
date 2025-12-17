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
            "message": f"Loaded JSON array with {len(data)} items. Consider converting to CSV for analysis using write_csv, then use analytics tools."
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
            "description": "Get JSON file metadata and small sample. Returns structure info and sample only. For analysis, convert to CSV first using write_csv, then use analytics tools.",
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
    }
]

# Map function names to implementations
TOOLS = {
    "read_csv": read_csv,
    "read_json": read_json,
    "write_csv": write_csv,
    "write_json": write_json,
    "copy_file": copy_file,
    "list_files": list_files,
    "file_info": file_info,
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

