import pandas as pd
from app.sandbox import resolve_path


def filter_rows(path: str, column: str, operator: str, value, output_path: str = None) -> dict:
    """Filter rows based on a condition."""
    df = pd.read_csv(resolve_path(path))
    
    ops = {
        "==": lambda x, v: x == v,
        "!=": lambda x, v: x != v,
        ">": lambda x, v: x > v,
        ">=": lambda x, v: x >= v,
        "<": lambda x, v: x < v,
        "<=": lambda x, v: x <= v,
        "contains": lambda x, v: x.astype(str).str.contains(str(v), na=False),
        "isnull": lambda x, v: x.isnull(),
        "notnull": lambda x, v: x.notnull(),
    }
    
    if operator not in ops:
        return {"error": f"Unknown operator: {operator}"}
    
    filtered = df[ops[operator](df[column], value)]
    
    if output_path:
        filtered.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "rows": len(filtered), "output": output_path}
    
    return {"success": True, "rows": len(filtered), "data": filtered.to_dict(orient="records")}


def select_columns(path: str, columns: list[str], output_path: str = None) -> dict:
    """Select specific columns from a CSV."""
    df = pd.read_csv(resolve_path(path))
    selected = df[columns]
    
    if output_path:
        selected.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "columns": columns, "output": output_path}
    
    return {"success": True, "columns": columns, "data": selected.to_dict(orient="records")}


def sort_data(path: str, by: list[str], ascending: bool = True, output_path: str = None) -> dict:
    """Sort data by one or more columns."""
    df = pd.read_csv(resolve_path(path))
    sorted_df = df.sort_values(by=by, ascending=ascending)
    
    if output_path:
        sorted_df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "sorted_by": by, "output": output_path}
    
    return {"success": True, "sorted_by": by, "data": sorted_df.to_dict(orient="records")}


def join_data(left_path: str, right_path: str, on: str, how: str = "inner", output_path: str = None) -> dict:
    """Join two CSV files."""
    left = pd.read_csv(resolve_path(left_path))
    right = pd.read_csv(resolve_path(right_path))
    
    merged = pd.merge(left, right, on=on, how=how)
    
    if output_path:
        merged.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "rows": len(merged), "output": output_path}
    
    return {"success": True, "rows": len(merged), "data": merged.to_dict(orient="records")}


def group_aggregate(path: str, group_by: list[str], aggregations: dict, output_path: str = None) -> dict:
    """
    Group by columns and aggregate.
    aggregations: {"column": "mean|sum|count|min|max"}
    """
    df = pd.read_csv(resolve_path(path))
    grouped = df.groupby(group_by).agg(aggregations).reset_index()
    
    # Flatten column names if multi-level
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    
    if output_path:
        grouped.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "rows": len(grouped), "output": output_path}
    
    return {"success": True, "rows": len(grouped), "data": grouped.to_dict(orient="records")}


def pivot_data(path: str, index: str, columns: str, values: str, output_path: str = None) -> dict:
    """Pivot a table."""
    df = pd.read_csv(resolve_path(path))
    pivoted = df.pivot_table(index=index, columns=columns, values=values, aggfunc='sum').reset_index()
    
    if output_path:
        pivoted.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "output": output_path}
    
    return {"success": True, "data": pivoted.to_dict(orient="records")}


def melt_data(path: str, id_vars: list[str], value_vars: list[str] = None, output_path: str = None) -> dict:
    """Unpivot/melt a table."""
    df = pd.read_csv(resolve_path(path))
    melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
    
    if output_path:
        melted.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "rows": len(melted), "output": output_path}
    
    return {"success": True, "rows": len(melted), "data": melted.to_dict(orient="records")}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": "Filter rows in a CSV based on a condition",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Input CSV path"},
                    "column": {"type": "string", "description": "Column to filter on"},
                    "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "isnull", "notnull"]},
                    "value": {"description": "Value to compare against"},
                    "output_path": {"type": "string", "description": "Optional output CSV path"}
                },
                "required": ["path", "column", "operator", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_columns",
            "description": "Select specific columns from a CSV",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_data",
            "description": "Sort CSV data by columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "by": {"type": "array", "items": {"type": "string"}},
                    "ascending": {"type": "boolean", "default": True},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "by"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "join_data",
            "description": "Join two CSV files on a column",
            "parameters": {
                "type": "object",
                "properties": {
                    "left_path": {"type": "string"},
                    "right_path": {"type": "string"},
                    "on": {"type": "string", "description": "Column to join on"},
                    "how": {"type": "string", "enum": ["inner", "left", "right", "outer"], "default": "inner"},
                    "output_path": {"type": "string"}
                },
                "required": ["left_path", "right_path", "on"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "group_aggregate",
            "description": "Group by columns and aggregate (sum, mean, count, min, max)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "group_by": {"type": "array", "items": {"type": "string"}},
                    "aggregations": {"type": "object", "description": "e.g. {\"revenue\": \"sum\", \"orders\": \"count\"}"},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "group_by", "aggregations"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pivot_data",
            "description": "Pivot a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "index": {"type": "string"},
                    "columns": {"type": "string"},
                    "values": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "index", "columns", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "melt_data",
            "description": "Unpivot/melt a table from wide to long format",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "id_vars": {"type": "array", "items": {"type": "string"}},
                    "value_vars": {"type": "array", "items": {"type": "string"}},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "id_vars"]
            }
        }
    }
]

TOOLS = {
    "filter_rows": filter_rows,
    "select_columns": select_columns,
    "sort_data": sort_data,
    "join_data": join_data,
    "group_aggregate": group_aggregate,
    "pivot_data": pivot_data,
    "melt_data": melt_data,
}


if __name__ == "__main__":
    from app.tools.file_ops import write_csv
    
    # Create test data
    sales = [
        {"region": "North", "product": "A", "revenue": 100},
        {"region": "North", "product": "B", "revenue": 150},
        {"region": "South", "product": "A", "revenue": 200},
        {"region": "South", "product": "B", "revenue": 120},
    ]
    write_csv("sales.csv", sales)
    
    print("filter_rows (revenue > 100):")
    print(filter_rows("sales.csv", "revenue", ">", 100))
    
    print("\nselect_columns:")
    print(select_columns("sales.csv", ["region", "revenue"]))
    
    print("\nsort_data:")
    print(sort_data("sales.csv", ["revenue"], ascending=False))
    
    print("\ngroup_aggregate:")
    print(group_aggregate("sales.csv", ["region"], {"revenue": "sum"}))

