import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Suppress matplotlib's verbose font logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import seaborn as sns
from app.sandbox import resolve_path


def _get_unique_filename(base_path: str) -> str:
    """
    Generate a unique filename to avoid overwriting existing files.
    If the file exists, append a number (e.g., plot_1.png, plot_2.png).
    """
    if not os.path.exists(base_path):
        return base_path

    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename) if directory else new_filename
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def plot_line(
    path: str,
    x_column: str,
    y_columns: list[str],
    output_path: str = "line_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a line plot using matplotlib."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    for y_col in y_columns:
        plt.plot(df[x_column], df[y_col], marker='o', label=y_col)

    plt.xlabel(xlabel or x_column)
    plt.ylabel(ylabel or ', '.join(y_columns))
    plt.title(title or f"Line Plot: {', '.join(y_columns)} vs {x_column}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_scatter(
    path: str,
    x_column: str,
    y_column: str,
    hue_column: str = None,
    output_path: str = "scatter_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a scatter plot using matplotlib/seaborn."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    if hue_column:
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, s=100, alpha=0.6)
    else:
        plt.scatter(df[x_column], df[y_column], s=100, alpha=0.6)

    plt.xlabel(xlabel or x_column)
    plt.ylabel(ylabel or y_column)
    plt.title(title or f"Scatter Plot: {y_column} vs {x_column}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_bar(
    path: str,
    x_column: str,
    y_column: str,
    output_path: str = "bar_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    horizontal: bool = False,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a bar plot using matplotlib."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    if horizontal:
        plt.barh(df[x_column], df[y_column])
    else:
        plt.bar(df[x_column], df[y_column])

    plt.xlabel(xlabel or x_column)
    plt.ylabel(ylabel or y_column)
    plt.title(title or f"Bar Plot: {y_column} by {x_column}")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_histogram(
    path: str,
    column: str,
    bins: int = 30,
    output_path: str = "histogram.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Frequency",
    kde: bool = True,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a histogram using seaborn."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=column, bins=bins, kde=kde)

    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.title(title or f"Distribution of {column}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_box(
    path: str,
    columns: list[str] = None,
    x_column: str = None,
    y_column: str = None,
    output_path: str = "box_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a box plot using seaborn."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)

    if x_column and y_column:
        sns.boxplot(data=df, x=x_column, y=y_column)
        plt.xticks(rotation=45, ha='right')
    elif columns:
        sns.boxplot(data=df[columns])
        plt.xticks(rotation=45, ha='right')
    else:
        # Plot all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sns.boxplot(data=df[numeric_cols])
        plt.xticks(rotation=45, ha='right')

    plt.xlabel(xlabel or "")
    plt.ylabel(ylabel or "Value")
    plt.title(title or "Box Plot")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_heatmap(
    path: str,
    columns: list[str] = None,
    output_path: str = "heatmap.png",
    title: str = None,
    annot: bool = True,
    cmap: str = "coolwarm",
    figsize: tuple[int, int] = (10, 8),
    overwrite: bool = False
) -> dict:
    """Create a correlation heatmap using seaborn."""
    df = pd.read_csv(resolve_path(path))

    if columns:
        df = df[columns]
    else:
        df = df.select_dtypes(include=[np.number])

    corr = df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title(title or "Correlation Heatmap")
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_pairplot(
    path: str,
    columns: list[str] = None,
    hue_column: str = None,
    output_path: str = "pairplot.png",
    figsize: tuple[int, int] = (12, 12),
    overwrite: bool = False
) -> dict:
    """Create a pair plot using seaborn."""
    df = pd.read_csv(resolve_path(path))

    if columns:
        plot_cols = columns + ([hue_column] if hue_column and hue_column not in columns else [])
        df = df[plot_cols]

    pairplot = sns.pairplot(df, hue=hue_column, diag_kind='kde', corner=True)
    pairplot.fig.set_size_inches(figsize)

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    pairplot.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_violin(
    path: str,
    x_column: str,
    y_column: str,
    output_path: str = "violin_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a violin plot using seaborn."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=x_column, y=y_column)

    plt.xlabel(xlabel or x_column)
    plt.ylabel(ylabel or y_column)
    plt.title(title or f"Violin Plot: {y_column} by {x_column}")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_pie(
    path: str,
    column: str,
    top_n: int = 10,
    output_path: str = "pie_chart.png",
    title: str = None,
    figsize: tuple[int, int] = (10, 8),
    overwrite: bool = False
) -> dict:
    """Create a pie chart using matplotlib."""
    df = pd.read_csv(resolve_path(path))

    value_counts = df[column].value_counts().head(top_n)

    plt.figure(figsize=figsize)
    plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(title or f"Distribution of {column} (Top {top_n})")
    plt.axis('equal')
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_count(
    path: str,
    column: str,
    hue_column: str = None,
    output_path: str = "count_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Count",
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a count plot using seaborn."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=column, hue=hue_column)

    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.title(title or f"Count of {column}")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


def plot_regression(
    path: str,
    x_column: str,
    y_column: str,
    output_path: str = "regression_plot.png",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple[int, int] = (10, 6),
    overwrite: bool = False
) -> dict:
    """Create a regression plot with fitted line using seaborn."""
    df = pd.read_csv(resolve_path(path))

    plt.figure(figsize=figsize)
    sns.regplot(data=df, x=x_column, y=y_column, scatter_kws={'alpha': 0.5})

    plt.xlabel(xlabel or x_column)
    plt.ylabel(ylabel or y_column)
    plt.title(title or f"Regression: {y_column} vs {x_column}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    full_output = resolve_path(output_path)
    if not overwrite:
        full_output = _get_unique_filename(full_output)

    plt.savefig(full_output, dpi=300, bbox_inches='tight')
    plt.close()

    relative_path = os.path.relpath(full_output, os.path.dirname(resolve_path("")))
    return {"success": True, "output": relative_path}


# Tool schemas for LLM
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "plot_line",
            "description": "Create a line plot from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to CSV file"},
                    "x_column": {"type": "string", "description": "Column for x-axis"},
                    "y_columns": {"type": "array", "items": {"type": "string"}, "description": "Columns for y-axis"},
                    "output_path": {"type": "string", "default": "line_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False, "description": "Overwrite existing file"}
                },
                "required": ["path", "x_column", "y_columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_scatter",
            "description": "Create a scatter plot from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "hue_column": {"type": "string", "description": "Column for color grouping"},
                    "output_path": {"type": "string", "default": "scatter_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "x_column", "y_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_bar",
            "description": "Create a bar plot from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "output_path": {"type": "string", "default": "bar_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string"},
                    "horizontal": {"type": "boolean", "default": False},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "x_column", "y_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_histogram",
            "description": "Create a histogram with optional KDE from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "bins": {"type": "integer", "default": 30},
                    "output_path": {"type": "string", "default": "histogram.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string", "default": "Frequency"},
                    "kde": {"type": "boolean", "default": True, "description": "Show kernel density estimate"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_box",
            "description": "Create a box plot from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "output_path": {"type": "string", "default": "box_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_heatmap",
            "description": "Create a correlation heatmap from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "output_path": {"type": "string", "default": "heatmap.png"},
                    "title": {"type": "string"},
                    "annot": {"type": "boolean", "default": True, "description": "Annotate cells with values"},
                    "cmap": {"type": "string", "default": "coolwarm"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 8]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_pairplot",
            "description": "Create a pair plot (scatter plot matrix) from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "hue_column": {"type": "string"},
                    "output_path": {"type": "string", "default": "pairplot.png"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [12, 12]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_violin",
            "description": "Create a violin plot from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "output_path": {"type": "string", "default": "violin_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "x_column", "y_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_pie",
            "description": "Create a pie chart from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10},
                    "output_path": {"type": "string", "default": "pie_chart.png"},
                    "title": {"type": "string"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 8]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_count",
            "description": "Create a count plot (categorical bar plot) from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "hue_column": {"type": "string"},
                    "output_path": {"type": "string", "default": "count_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string", "default": "Count"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_regression",
            "description": "Create a regression plot with fitted line from CSV data",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "output_path": {"type": "string", "default": "regression_plot.png"},
                    "title": {"type": "string"},
                    "xlabel": {"type": "string"},
                    "ylabel": {"type": "string"},
                    "figsize": {"type": "array", "items": {"type": "integer"}, "default": [10, 6]},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["path", "x_column", "y_column"]
            }
        }
    }
]

# Map function names to implementations
TOOLS = {
    "plot_line": plot_line,
    "plot_scatter": plot_scatter,
    "plot_bar": plot_bar,
    "plot_histogram": plot_histogram,
    "plot_box": plot_box,
    "plot_heatmap": plot_heatmap,
    "plot_pairplot": plot_pairplot,
    "plot_violin": plot_violin,
    "plot_pie": plot_pie,
    "plot_count": plot_count,
    "plot_regression": plot_regression,
}


if __name__ == "__main__":
    from app.tools.file_ops import write_csv
    import numpy as np

    # Create test data
    np.random.seed(42)
    data = [
        {
            "x": i,
            "y": i * 2 + np.random.randn() * 5,
            "category": "A" if i % 2 == 0 else "B",
            "value": np.random.randint(10, 100)
        }
        for i in range(50)
    ]
    write_csv("test_viz.csv", data)

    print("Testing visualization tools...")

    # Test line plot
    result = plot_line("test_viz.csv", x_column="x", y_columns=["y", "value"])
    print(f"plot_line: {result}")

    # Test scatter plot
    result = plot_scatter("test_viz.csv", x_column="x", y_column="y", hue_column="category")
    print(f"plot_scatter: {result}")

    # Test histogram
    result = plot_histogram("test_viz.csv", column="value")
    print(f"plot_histogram: {result}")

    # Test bar plot
    result = plot_bar("test_viz.csv", x_column="category", y_column="value")
    print(f"plot_bar: {result}")

    print("All visualizations created successfully!")
