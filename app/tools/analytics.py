import pandas as pd
import numpy as np
from app.sandbox import resolve_path


def summary_stats(path: str, columns: list[str] = None) -> dict:
    """Get summary statistics for numeric columns."""
    df = pd.read_csv(resolve_path(path))
    if columns:
        df = df[columns]
    
    stats = df.describe().to_dict()
    return {"success": True, "statistics": stats}


def correlation_matrix(path: str, columns: list[str] = None, output_path: str = None) -> dict:
    """Compute correlation matrix for numeric columns."""
    df = pd.read_csv(resolve_path(path))
    if columns:
        df = df[columns]
    
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().round(3).to_dict()
    
    if output_path:
        numeric.corr().to_csv(resolve_path(output_path))
        return {"success": True, "output": output_path, "correlation": corr}
    return {"success": True, "correlation": corr}


def value_counts(path: str, column: str, top_n: int = 10) -> dict:
    """Get value counts for a column."""
    df = pd.read_csv(resolve_path(path))
    counts = df[column].value_counts().head(top_n).to_dict()
    return {"success": True, "column": column, "counts": counts}


def linear_regression(path: str, target: str, features: list[str], output_path: str = None) -> dict:
    """Fit a linear regression model."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    df = pd.read_csv(resolve_path(path)).dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    result = {
        "success": True,
        "r2_score": round(r2_score(y, predictions), 4),
        "rmse": round(np.sqrt(mean_squared_error(y, predictions)), 4),
        "coefficients": dict(zip(features, [round(c, 4) for c in model.coef_])),
        "intercept": round(model.intercept_, 4)
    }
    
    if output_path:
        df["predicted"] = predictions
        df.to_csv(resolve_path(output_path), index=False)
        result["output"] = output_path
    
    return result


def kmeans_cluster(path: str, features: list[str], n_clusters: int = 3, output_path: str = None) -> dict:
    """Perform K-means clustering."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(resolve_path(path)).dropna(subset=features)
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    cluster_sizes = df["cluster"].value_counts().to_dict()
    
    if output_path:
        df.to_csv(resolve_path(output_path), index=False)
        return {"success": True, "n_clusters": n_clusters, "cluster_sizes": cluster_sizes, "output": output_path}
    return {"success": True, "n_clusters": n_clusters, "cluster_sizes": cluster_sizes, "data": df.to_dict(orient="records")}


def train_test_split_data(path: str, test_size: float = 0.2, train_output: str = None, test_output: str = None) -> dict:
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(resolve_path(path))
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    
    result = {"success": True, "train_size": len(train), "test_size": len(test)}
    
    if train_output:
        train.to_csv(resolve_path(train_output), index=False)
        result["train_output"] = train_output
    if test_output:
        test.to_csv(resolve_path(test_output), index=False)
        result["test_output"] = test_output
    
    return result


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "summary_stats",
            "description": "Get summary statistics (mean, std, min, max, quartiles) for numeric columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to analyze (all numeric if omitted)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_matrix",
            "description": "Compute correlation matrix for numeric columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "output_path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "value_counts",
            "description": "Get frequency counts for values in a column",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "column": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10}
                },
                "required": ["path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regression",
            "description": "Fit a linear regression model and return coefficients and metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string", "description": "Target variable column"},
                    "features": {"type": "array", "items": {"type": "string"}, "description": "Feature columns"},
                    "output_path": {"type": "string", "description": "Save data with predictions"}
                },
                "required": ["path", "target", "features"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kmeans_cluster",
            "description": "Perform K-means clustering",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "n_clusters": {"type": "integer", "default": 3},
                    "output_path": {"type": "string"}
                },
                "required": ["path", "features"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_test_split_data",
            "description": "Split data into training and test sets",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "test_size": {"type": "number", "default": 0.2},
                    "train_output": {"type": "string"},
                    "test_output": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    }
]

TOOLS = {
    "summary_stats": summary_stats,
    "correlation_matrix": correlation_matrix,
    "value_counts": value_counts,
    "linear_regression": linear_regression,
    "kmeans_cluster": kmeans_cluster,
    "train_test_split_data": train_test_split_data,
}


if __name__ == "__main__":
    from app.tools.file_ops import write_csv
    
    # Create test data
    np.random.seed(42)
    data = [
        {"x1": x, "x2": x * 2 + np.random.randn(), "y": 3 * x + 5 + np.random.randn() * 2}
        for x in range(1, 51)
    ]
    write_csv("regression_data.csv", data)
    
    print("summary_stats:")
    print(summary_stats("regression_data.csv"))
    
    print("\ncorrelation_matrix:")
    print(correlation_matrix("regression_data.csv"))
    
    print("\nlinear_regression:")
    print(linear_regression("regression_data.csv", target="y", features=["x1", "x2"]))
    
    print("\nkmeans_cluster:")
    print(kmeans_cluster("regression_data.csv", features=["x1", "x2"], n_clusters=3))
    
    print("\ntrain_test_split:")
    print(train_test_split_data("regression_data.csv", test_size=0.2, train_output="train.csv", test_output="test.csv"))

