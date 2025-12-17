import pandas as pd
from app.sandbox import resolve_path


def fill_missing(path: str, column: str = None, value=None, method: str = None, output_path: str = None) -> dict:
    """
    Fill missing values.
    - value: fill with specific value
    - method: 'mean', 'median', 'mode', 'ffill', 'bfill'
    """
    df = pd.read_csv(resolve_path(path))
    cols = [column] if column else df.columns
    
    for col in cols:
        if col not in df.columns:
            continue
        if value is not None:
            df[col] = df[col].fillna(value)
        elif method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else None)
        elif method in ("ffill", "bfill"):
            df[col] = df[col].fillna(method=method)
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "output": output_path}
    return {"success": True, "data": df.to_dict(orient="records")}


def drop_missing(path: str, columns: list[str] = None, how: str = "any", output_path: str = None) -> dict:
    """Drop rows with missing values."""
    df = pd.read_csv(resolve_path(path))
    original_len = len(df)
    
    df = df.dropna(subset=columns, how=how)
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "dropped": original_len - len(df), "output": output_path}
    return {"success": True, "dropped": original_len - len(df), "data": df.to_dict(orient="records")}


def convert_types(path: str, conversions: dict, output_path: str = None) -> dict:
    """
    Convert column types.
    conversions: {"column": "int|float|str|datetime"}
    """
    df = pd.read_csv(resolve_path(path))
    
    for col, dtype in conversions.items():
        if col not in df.columns:
            continue
        if dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif dtype == "int":
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == "str":
            df[col] = df[col].astype(str)
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "output": output_path}
    return {"success": True, "data": df.to_dict(orient="records")}


def deduplicate(path: str, columns: list[str] = None, keep: str = "first", output_path: str = None) -> dict:
    """Remove duplicate rows."""
    df = pd.read_csv(resolve_path(path))
    original_len = len(df)
    
    df = df.drop_duplicates(subset=columns, keep=keep)
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "removed": original_len - len(df), "output": output_path}
    return {"success": True, "removed": original_len - len(df), "data": df.to_dict(orient="records")}


def rename_columns(path: str, mapping: dict, output_path: str = None) -> dict:
    """Rename columns."""
    df = pd.read_csv(resolve_path(path))
    df = df.rename(columns=mapping)
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "columns": list(df.columns), "output": output_path}
    return {"success": True, "columns": list(df.columns), "data": df.to_dict(orient="records")}


def add_column(path: str, name: str, expression: str = None, source_column: str = None, 
               transform: str = None, pattern: str = None, replacement: str = "", 
               output_path: str = None) -> dict:
    """
    Add a computed column.
    - expression: pandas eval expression for arithmetic (e.g. 'price * quantity')
    - source_column + transform: apply a transform to a column
      transforms: 'regex_replace', 'lower', 'upper', 'strip', 'extract'
    - pattern/replacement: for regex operations
    """
    import re
    df = pd.read_csv(resolve_path(path))
    
    if expression:
        df[name] = df.eval(expression)
    elif source_column and transform:
        col = df[source_column].astype(str)
        if transform == "regex_replace":
            df[name] = col.str.replace(pattern, replacement, regex=True)
        elif transform == "extract":
            df[name] = col.str.extract(pattern, expand=False)
        elif transform == "lower":
            df[name] = col.str.lower()
        elif transform == "upper":
            df[name] = col.str.upper()
        elif transform == "strip":
            df[name] = col.str.strip()
        else:
            return {"success": False, "error": f"Unknown transform: {transform}"}
    else:
        return {"success": False, "error": "Provide either 'expression' or 'source_column' + 'transform'"}
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "output": output_path}
    return {"success": True, "data": df.to_dict(orient="records")}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fill_missing",
            "description": "Fill missing values with a value or method (mean, median, mode, ffill, bfill)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string", "description": "Column to fill (or all if omitted)"},
                    "value": {"description": "Value to fill with"},
                    "method": {"type": "string", "enum": ["mean", "median", "mode", "ffill", "bfill"]},
                    "output_path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop_missing",
            "description": "Drop rows with missing values",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "how": {"type": "string", "enum": ["any", "all"], "default": "any"},
                    "output_path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_types",
            "description": "Convert column data types",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "conversions": {"type": "object", "description": "{\"column\": \"int|float|str|datetime\"}"},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "conversions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deduplicate",
            "description": "Remove duplicate rows",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to check for duplicates"},
                    "keep": {"type": "string", "enum": ["first", "last", "false"], "default": "first"},
                    "output_path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rename_columns",
            "description": "Rename columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mapping": {"type": "object", "description": "{\"old_name\": \"new_name\"}"},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "mapping"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_column",
            "description": "Add a computed column. Use 'expression' for arithmetic (e.g. 'price * quantity'), or use 'source_column' + 'transform' for string operations like regex_replace, lower, upper, strip, extract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string", "description": "New column name"},
                    "expression": {"type": "string", "description": "Pandas eval expression for arithmetic"},
                    "source_column": {"type": "string", "description": "Column to transform"},
                    "transform": {"type": "string", "enum": ["regex_replace", "extract", "lower", "upper", "strip"]},
                    "pattern": {"type": "string", "description": "Regex pattern for regex_replace or extract"},
                    "replacement": {"type": "string", "description": "Replacement string for regex_replace"},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "name"]
            }
        }
    }
]

TOOLS = {
    "fill_missing": fill_missing,
    "drop_missing": drop_missing,
    "convert_types": convert_types,
    "deduplicate": deduplicate,
    "rename_columns": rename_columns,
    "add_column": add_column,
}


if __name__ == "__main__":
    from app.tools.file_ops import write_csv
    
    # Test data with issues
    data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": None, "score": 90},
        {"name": "Alice", "age": 30, "score": 85},  # duplicate
        {"name": "Carol", "age": 25, "score": None},
    ]
    write_csv("dirty.csv", data)
    
    print("fill_missing (age with mean):")
    print(fill_missing("dirty.csv", column="age", method="mean"))
    
    print("\ndrop_missing:")
    print(drop_missing("dirty.csv"))
    
    print("\ndeduplicate:")
    print(deduplicate("dirty.csv"))
    
    print("\nrename_columns:")
    print(rename_columns("dirty.csv", {"name": "full_name", "score": "test_score"}))

