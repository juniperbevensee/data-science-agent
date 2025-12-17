# Data Science Agent

A powerful suite of data science, analytics, and visualization capabilities that responds to natural language requests using LLM interpretation. This agent provides tools for data manipulation, statistical analysis, machine learning, and comprehensive data visualization.

## Features

### 📊 Data Visualization
Create publication-quality visualizations using matplotlib and seaborn:
- **Line plots** - Track trends over time or ordered data
- **Scatter plots** - Explore relationships with optional color grouping
- **Bar charts** - Compare categorical data (vertical or horizontal)
- **Histograms** - Visualize distributions with optional KDE
- **Box plots** - Display statistical summaries and outliers
- **Heatmaps** - Correlation matrices with customizable color schemes
- **Pair plots** - Scatter plot matrices for multi-variable analysis
- **Violin plots** - Distribution shapes by category
- **Pie charts** - Show proportions and percentages
- **Count plots** - Frequency of categorical values
- **Regression plots** - Linear relationships with fitted lines

All visualizations are saved as high-resolution PNGs (300 DPI) in the workspace directory.

### 📈 Analytics & Statistics
- **Summary statistics** - Mean, std, min, max, quartiles for numeric columns
- **Correlation analysis** - Compute correlation matrices
- **Value counts** - Frequency analysis for categorical data
- **Linear regression** - Fit models with R², RMSE, coefficients
- **K-means clustering** - Unsupervised learning with customizable clusters
- **Train/test splitting** - Prepare data for machine learning

### 🔧 Data Transformation
- Filter, sort, and select data
- Create new calculated columns
- Aggregate and group operations
- Pivot and reshape data

### 🧹 Data Cleaning
- Handle missing values (drop, fill, impute)
- Remove duplicates
- Normalize and standardize data
- Type conversion and validation

### 📁 File Operations
- Read/write CSV and JSON files
- Copy and manage files
- Browse workspace directory
- Get file information

## Installation

### Prerequisites
- Python 3.8 or higher
- LM Studio (or OpenAI-compatible API server)

### Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd data-science-agent
```

2. Start the server (recommended):
```bash
./start.sh
```

The `start.sh` script will automatically:
- Create a virtual environment
- Install all dependencies from requirements.txt
- Create the workspace directory
- Check if LM Studio is running
- Start the Flask server on `http://0.0.0.0:5000`

### Manual Setup

If you prefer to set up manually:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the LLM backend in `config.py` (if needed):
```python
LLM_BASE_URL = "http://localhost:1234/v1"  # Your LM Studio URL
LLM_MODEL = "local-model"
```

3. Run the server:
```bash
python run.py
```

## Usage

### API Endpoint

Send POST requests to `/query` with natural language instructions:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create a scatter plot of price vs size from sales.csv with colors by category"
  }'
```

### Example Queries

**Visualization:**
- "Create a line plot showing sales over time from quarterly_data.csv"
- "Make a correlation heatmap of all numeric columns in dataset.csv"
- "Show me a histogram of the age distribution"
- "Create a scatter plot of height vs weight colored by gender"

**Analysis:**
- "What are the summary statistics for sales.csv?"
- "Show me value counts for the category column"
- "Fit a linear regression predicting price from size and bedrooms"
- "Cluster the data into 5 groups based on features X, Y, Z"

**Data Operations:**
- "Read the CSV file data.csv and show me the first 10 rows"
- "Remove duplicates from customers.csv and save to clean_customers.csv"
- "Fill missing values in revenue column with the mean"
- "Create a new column 'profit' as revenue minus cost"

## Architecture

### Components

- **Flask Server** (`app/routes.py`) - Handles HTTP requests
- **LLM Client** (`app/llm_client.py`) - Communicates with language model
- **Executor** (`app/executor.py`) - Executes tool calls from LLM
- **Sandbox** (`app/sandbox.py`) - Secure workspace file operations
- **Tools** (`app/tools/`) - Modular tool implementations

### Tool Categories

1. **File Operations** (`file_ops.py`) - File I/O and management
2. **Transforms** (`transforms.py`) - Data manipulation
3. **Cleaning** (`cleaning.py`) - Data quality operations
4. **Analytics** (`analytics.py`) - Statistical analysis and ML
5. **Visualization** (`visualization.py`) - Charts and plots

### Workspace Directory

All data files and visualizations are stored in the `workspace/` directory for security and organization. The agent cannot access files outside this directory.

### File Naming

By default, visualizations use descriptive names and automatically avoid overwriting existing files by appending numbers (e.g., `plot.png`, `plot_1.png`, `plot_2.png`). To force overwriting, set `overwrite=True` in your request.

## Configuration

Edit `config.py` to customize:

```python
# LLM Settings
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "local-model"

# Workspace
WORKSPACE_DIR = "workspace"

# Flask Server
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
```

## Development

### Adding New Tools

1. Create a new function in the appropriate tool file
2. Add the tool schema to `TOOL_SCHEMAS`
3. Register the function in the `TOOLS` dictionary
4. The tool will be automatically available to the LLM

Example:
```python
def my_new_tool(param: str) -> dict:
    """Tool description."""
    # Implementation
    return {"success": True, "result": "..."}

TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "my_new_tool",
        "description": "What this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param"]
        }
    }
})

TOOLS["my_new_tool"] = my_new_tool
```

### Testing Individual Tools

Each tool module can be run standalone for testing:
```bash
python -m app.tools.visualization
python -m app.tools.analytics
```

## Dependencies

- **Flask** - Web server framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **matplotlib** - Plotting library
- **seaborn** - Statistical visualization
- **requests** - HTTP client

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tools return `{"success": True/False}` dictionaries
- All file paths use `resolve_path()` for security
- Visualizations are saved as PNG with 300 DPI
- New tools include proper schemas and documentation 
