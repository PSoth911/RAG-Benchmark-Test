# Guide: `analyze_ollama_benchmark.py`

## 1. Purpose of the analysis script

Your benchmark script (`benchmark.py`) sends one fixed RAG-explanation prompt to a local Ollama model several times and appends one row per run to `results.csv`.

`analyze_ollama_benchmark.py` is a **separate** script that:

- reads `results.csv` (read-only, the file is never changed),
- shows **every individual run** (Run 1, Run 2, Run 3, ...),
- groups runs by **model**,
- adds an **`AVG` row** after the runs of each model,
- prints a readable table in the terminal,
- exports the same table to **CSV** and **Excel (`.xlsx`)**.

Averages are calculated from the raw values **before** any rounding. Rounding is applied only when displaying and exporting.

---

## 2. Project folder structure

```
your_project/
├── benchmark.py                    # your original benchmark script (unchanged)
├── analyze_ollama_benchmark.py     # analysis script
├── analyze_ollama_benchmark_guide.md
├── results.csv                     # raw data, created by benchmark.py
├── results_analysis.csv            # created by the analysis script
└── results_analysis.xlsx           # created by the analysis script
```

Keep `analyze_ollama_benchmark.py` in the same folder as `results.csv`. (Your benchmark writes `results.csv` next to itself, and the analysis script looks for it there by default.) If your files are elsewhere, use `--input` (see section 6).

---

## 3. Required dependencies

| Package | Why |
|---|---|
| `pandas` | read the CSV, group and average |
| `openpyxl` | write the `.xlsx` file |

(Your benchmark also needs `requests`, but the analysis script does not.)

```bash
py -m pip install pandas openpyxl
```

Optional: use a virtual environment first.

```bash
py -m venv venv
venv\Scripts\activate           # Windows (macOS/Linux: source venv/bin/activate)
py -m pip install pandas openpyxl
```

---

## 4. Input CSV preparation and expected column names

You do not need to prepare anything. The analysis script reads the file exactly as your benchmark writes it. The header is:

```
timestamp,model,run,latency_seconds,load_time_seconds,prompt_tokens,generated_tokens,tokens_per_second
```

Example raw rows:

```
2026-11-24T10:00:01,llama3.2:3b,1,12.345,2.1,38,310,28.41
2026-11-24T10:00:14,llama3.2:3b,2,10.002,0.05,38,295,30.12
```

If a column is missing or renamed, the script stops and lists what it found and what it expected.

### Important: the benchmark appends

`benchmark.py` **appends** to `results.csv` every time you launch it, and `run` restarts at 1 each launch. So if you benchmark `llama3.2:3b` today and again tomorrow, the file contains two sets of Run 1, 2, 3 for that model.

The analysis script handles this by creating a **`batch`** column. A new batch starts for a model whenever its `run` number does not increase. Each batch gets its own runs and its own `AVG` row, so separate sessions are never mixed into one average. `batch` is calculated by the analysis script; it does not exist in `results.csv`.

Use `--latest-only` to analyze only the most recent batch of each model.

---

## 5. Complete analysis Python code

Save as `analyze_ollama_benchmark.py`:

```python
"""
analyze_ollama_benchmark.py

Reads the raw results.csv written by the Ollama benchmark script,
shows every individual run grouped by model, adds an AVG row after each
group, prints a summary table, and exports CSV + Excel files.

The raw results.csv is only READ. It is never modified.

Usage:
    py analyze_ollama_benchmark.py
    py analyze_ollama_benchmark.py --input results.csv
    py analyze_ollama_benchmark.py --latest-only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings (these match the columns written by the benchmark script)
# ---------------------------------------------------------------------------
SCRIPT_FOLDER = Path(__file__).parent

DEFAULT_INPUT_FILE = SCRIPT_FOLDER / "results.csv"
OUTPUT_CSV_FILE = SCRIPT_FOLDER / "results_analysis.csv"
OUTPUT_EXCEL_FILE = SCRIPT_FOLDER / "results_analysis.xlsx"

# Columns that identify a run
ID_COLUMNS = ["timestamp", "model", "run"]

# Numeric metrics that are measured and averaged
METRIC_COLUMNS = [
    "latency_seconds",
    "load_time_seconds",
    "prompt_tokens",
    "generated_tokens",
    "tokens_per_second",
]

REQUIRED_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

# Decimal places used ONLY when displaying/exporting (never before averaging)
DECIMALS = {
    "latency_seconds": 3,
    "load_time_seconds": 3,
    "prompt_tokens": 2,
    "generated_tokens": 2,
    "tokens_per_second": 2,
}


# ---------------------------------------------------------------------------
# Step 1: Load and validate the raw CSV
# ---------------------------------------------------------------------------
def load_raw_results(csv_path):
    """Read the raw CSV and check that it is usable. Exits with a message if not."""
    if not csv_path.exists():
        sys.exit(
            f"ERROR: Input file not found: {csv_path}\n"
            "Run the benchmark script first so it creates results.csv, "
            "or pass the file location with --input."
        )

    try:
        raw_data = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        sys.exit(f"ERROR: {csv_path} is empty.")
    except Exception as error:
        sys.exit(f"ERROR: Could not read {csv_path}: {error}")

    if raw_data.empty:
        sys.exit(f"ERROR: {csv_path} has a header but no benchmark rows.")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw_data.columns]
    if missing_columns:
        sys.exit(
            f"ERROR: Missing column(s) in {csv_path.name}: {missing_columns}\n"
            f"Columns found: {list(raw_data.columns)}\n"
            f"Columns expected: {REQUIRED_COLUMNS}"
        )

    # Convert metric columns to numbers; bad values become NaN
    number_columns = ["run"] + METRIC_COLUMNS
    for column in number_columns:
        raw_data[column] = pd.to_numeric(raw_data[column], errors="coerce")

    bad_rows = raw_data[number_columns].isna().any(axis=1)
    if bad_rows.any():
        print(f"WARNING: Skipping {bad_rows.sum()} row(s) with missing or "
              "non-numeric values.")
        raw_data = raw_data[~bad_rows]

    if raw_data.empty:
        sys.exit("ERROR: No valid rows left after removing bad rows.")

    raw_data["run"] = raw_data["run"].astype(int)
    return raw_data.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Split rows into batches
# ---------------------------------------------------------------------------
def add_batch_numbers(data):
    """
    The benchmark script APPENDS to results.csv, and 'run' restarts at 1 each
    time you launch it. So one model can appear in several sessions (batches).
    A new batch starts for a model whenever its run number does not increase.
    """
    batch_numbers = []
    last_run = {}      # model -> last run number seen
    current_batch = {} # model -> current batch number

    for _, row in data.iterrows():
        model = row["model"]
        if model not in current_batch:
            current_batch[model] = 1
        elif row["run"] <= last_run[model]:
            current_batch[model] += 1
        last_run[model] = row["run"]
        batch_numbers.append(current_batch[model])

    data = data.copy()
    data["batch"] = batch_numbers
    return data


# ---------------------------------------------------------------------------
# Step 3: Build the detailed table with AVG rows
# ---------------------------------------------------------------------------
def build_detailed_table(data, latest_only):
    """Individual runs per model/batch, each followed by an AVG row."""
    if latest_only:
        latest = data.groupby("model")["batch"].transform("max")
        data = data[data["batch"] == latest]

    table_parts = []

    for model_name in sorted(data["model"].unique()):
        model_data = data[data["model"] == model_name]

        for batch_number in sorted(model_data["batch"].unique()):
            batch_data = model_data[model_data["batch"] == batch_number]

            # Individual runs (raw, unrounded values)
            runs = batch_data.copy()
            runs["run"] = runs["run"].astype(str)

            # AVG row calculated from the raw values BEFORE rounding
            avg_row = {
                "timestamp": "",
                "model": model_name,
                "batch": batch_number,
                "run": "AVG",
            }
            for metric in METRIC_COLUMNS:
                avg_row[metric] = batch_data[metric].mean()

            table_parts.append(runs)
            table_parts.append(pd.DataFrame([avg_row]))

    detailed = pd.concat(table_parts, ignore_index=True)
    column_order = ["model", "batch", "run", "timestamp"] + METRIC_COLUMNS
    return detailed[column_order]


# ---------------------------------------------------------------------------
# Step 4: Rounding (display/export only)
# ---------------------------------------------------------------------------
def round_for_display(detailed):
    rounded = detailed.copy()
    for metric, places in DECIMALS.items():
        rounded[metric] = rounded[metric].round(places)
    # Token counts of individual runs are whole numbers: show them without ".0"
    is_avg = rounded["run"] == "AVG"
    for metric in ["prompt_tokens", "generated_tokens"]:
        rounded[metric] = rounded[metric].astype(object)
        rounded.loc[~is_avg, metric] = detailed.loc[~is_avg, metric].astype(int)
    return rounded


# ---------------------------------------------------------------------------
# Step 5: Print in the terminal
# ---------------------------------------------------------------------------
def print_table(rounded):
    """Print every run and AVG row, one block per model (and batch)."""
    print("\n" + "=" * 100)
    print("BENCHMARK ANALYSIS - individual runs with AVG rows")
    print("=" * 100)

    show_batch = rounded["batch"].nunique() > 1
    columns = ["run"] + METRIC_COLUMNS

    for model_name in rounded["model"].unique():
        model_rows = rounded[rounded["model"] == model_name]

        for batch_number in model_rows["batch"].unique():
            group_rows = model_rows[model_rows["batch"] == batch_number]

            title = f"Model: {model_name}"
            if show_batch:
                title += f"  (batch {batch_number})"
            print(f"\n{title}")
            print(group_rows[columns].to_string(index=False))

    print("\n" + "=" * 100)


def print_summary(detailed_rounded):
    averages = detailed_rounded[detailed_rounded["run"] == "AVG"]
    summary = averages[["model", "batch"] + METRIC_COLUMNS]
    if summary["batch"].nunique() == 1:
        summary = summary.drop(columns=["batch"])
    print("\nSUMMARY (AVG rows only)")
    print(summary.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Step 6: Export CSV and Excel
# ---------------------------------------------------------------------------
def export_csv(rounded, path):
    try:
        rounded.to_csv(path, index=False, encoding="utf-8")
        print(f"CSV saved:   {path}")
    except PermissionError:
        print(f"ERROR: Cannot write {path}. Close it if it is open elsewhere.")


def export_excel(rounded, path):
    try:
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("ERROR: openpyxl is not installed. Run: py -m pip install openpyxl")
        return

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            rounded.to_excel(writer, sheet_name="Analysis", index=False)
            sheet = writer.sheets["Analysis"]

            # Bold header
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            sheet.freeze_panes = "A2"

            # Highlight AVG rows
            run_column_index = list(rounded.columns).index("run") + 1
            avg_fill = PatternFill("solid", fgColor="DDEBF7")
            for row_number in range(2, sheet.max_row + 1):
                if sheet.cell(row_number, run_column_index).value == "AVG":
                    for cell in sheet[row_number]:
                        cell.font = Font(bold=True)
                        cell.fill = avg_fill

            # Column widths
            for index, column in enumerate(rounded.columns, start=1):
                longest = max(len(str(column)),
                              *(len(str(v)) for v in rounded[column]))
                sheet.column_dimensions[get_column_letter(index)].width = longest + 2
        print(f"Excel saved: {path}")
    except PermissionError:
        print(f"ERROR: Cannot write {path}. Close it if it is open in Excel.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze Ollama benchmark results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE,
                        help="Raw benchmark CSV (default: results.csv next to this script)")
    parser.add_argument("--latest-only", action="store_true",
                        help="Only analyze the most recent batch of each model")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if input_path in (OUTPUT_CSV_FILE.resolve(), OUTPUT_EXCEL_FILE.resolve()):
        sys.exit("ERROR: Input file must be the raw benchmark CSV.")

    raw_data = load_raw_results(input_path)
    data = add_batch_numbers(raw_data)
    detailed = build_detailed_table(data, args.latest_only)
    rounded = round_for_display(detailed)

    print_table(rounded)
    print_summary(rounded)

    export_csv(rounded, OUTPUT_CSV_FILE)
    export_excel(rounded, OUTPUT_EXCEL_FILE)
    print(f"Raw file left unchanged: {input_path}")


if __name__ == "__main__":
    main()
```

---

## 6. How to run the analysis script

First run the benchmark (examples):

```bash
py benchmark.py --model llama3.2:3b --runs 3
py benchmark.py --model mistral:7b --runs 3
```

Then analyze:

```bash
py analyze_ollama_benchmark.py
```

Options:

```bash
# Use a raw CSV in another location
py analyze_ollama_benchmark.py --input path/to/results.csv

# Only the latest batch of each model
py analyze_ollama_benchmark.py --latest-only
```

---

## 7. Explanation of every benchmark metric

These are exactly the metrics your benchmark records, taken from Ollama's `/api/generate` response.

| Column | Meaning | Better is | How the AVG is calculated |
|---|---|---|---|
| `timestamp` | Time the result was saved (ISO format). Identifies the run, not a metric. | n/a | left empty in AVG row |
| `model` | Ollama model name. Used for grouping. | n/a | n/a |
| `run` | Run number within one launch of the benchmark (1, 2, 3, ...). | n/a | shown as `AVG` |
| `latency_seconds` | **Total wall-clock time** of the request, measured in Python with `time.perf_counter()`. It includes model loading, prompt processing, generation and network overhead. | Lower | mean of runs |
| `load_time_seconds` | Time Ollama spent **loading the model** into memory (`load_duration` / 1e9). Usually high on the first run (cold start) and near zero afterwards. | Lower | mean of runs |
| `prompt_tokens` | Number of tokens in the prompt (`prompt_eval_count`). The prompt is fixed, so this is normally constant. | n/a (input size) | mean of runs |
| `generated_tokens` | Number of tokens the model produced (`eval_count`). Longer answers mean longer latency, so compare with care. | n/a (output size) | mean of runs |
| `tokens_per_second` | **Generation speed**: `generated_tokens / generation_time`, where generation time is Ollama's `eval_duration`. Load time and prompt processing are excluded. | Higher | mean of the per-run values |

Notes on reading the results:

- **Cold start:** Run 1 often has a high `load_time_seconds` and `latency_seconds`. This pulls the average up. If you want warm-model speed, compare Runs 2 and 3 or run more runs.
- **`tokens_per_second` is the fairest speed comparison** between models because it removes load time and is normalized by output length. `latency_seconds` is the best "what the user waits" figure.
- **Rounding:** your benchmark already rounds `latency_seconds` and `load_time_seconds` to 3 decimals and `tokens_per_second` to 2 decimals when writing the CSV. The analysis script averages those stored values at full precision and rounds only the final display (3 decimals for times, 2 for the others).
- The AVG of `tokens_per_second` is the mean of per-run speeds, not total tokens divided by total time. The two are close when runs are similar.

---

## 8. Expected terminal output

The numbers below are **example values** to show the layout. Your values will differ.

```
====================================================================================================
BENCHMARK ANALYSIS - individual runs with AVG rows
====================================================================================================

Model: llama3.2:3b
run  latency_seconds  load_time_seconds prompt_tokens generated_tokens  tokens_per_second
  1           12.345               2.10            38              310              28.41
  2           10.002               0.05            38              295              30.12
  3           10.511               0.04            38              301              29.87
AVG           10.953               0.73          38.0            302.0              29.47

Model: mistral:7b
run  latency_seconds  load_time_seconds prompt_tokens generated_tokens  tokens_per_second
  1           25.100               3.20            38              350              14.20
  2           22.900               0.06            38              340              15.01
AVG           24.000               1.63          38.0            345.0              14.60

====================================================================================================

SUMMARY (AVG rows only)
      model  latency_seconds  load_time_seconds prompt_tokens generated_tokens  tokens_per_second
llama3.2:3b           10.953               0.73          38.0            302.0              29.47
 mistral:7b           24.000               1.63          38.0            345.0              14.60

CSV saved:   .../results_analysis.csv
Excel saved: .../results_analysis.xlsx
Raw file left unchanged: .../results.csv
```

If a model appears in more than one launch, a `batch` column appears in the summary and each block is labeled, for example `Model: llama3.2:3b  (batch 2)`.

---

## 9. Generated output files

| File | Content |
|---|---|
| `results_analysis.csv` | Every run plus an `AVG` row per model (and batch). |
| `results_analysis.xlsx` | Same table in sheet **Analysis**, with bold header, frozen header row, auto-sized columns and highlighted (bold, light blue) `AVG` rows. |

Columns in both files:

```
model, batch, run, timestamp, latency_seconds, load_time_seconds, prompt_tokens, generated_tokens, tokens_per_second
```

Example CSV rows:

```
model,batch,run,timestamp,latency_seconds,load_time_seconds,prompt_tokens,generated_tokens,tokens_per_second
llama3.2:3b,1,1,2026-11-24T10:00:01,12.345,2.1,38,310,28.41
llama3.2:3b,1,2,2026-11-24T10:00:14,10.002,0.05,38,295,30.12
llama3.2:3b,1,3,2026-11-24T10:00:25,10.511,0.04,38,301,29.87
llama3.2:3b,1,AVG,,10.953,0.73,38.0,302.0,29.47
```

The analysis files are overwritten each time you run the script. `results.csv` is never written to.

---

## 10. Troubleshooting

| Message / problem | Cause | Fix |
|---|---|---|
| `Input file not found` | `results.csv` does not exist where the script looks | Run the benchmark first, or use `--input path/to/results.csv`. |
| `Missing column(s) ...` | CSV header differs from the benchmark's header | Check the header against section 4. Don't edit it by hand; regenerate with the benchmark. |
| `is empty` / `has a header but no benchmark rows` | Every benchmark run failed (for example, Ollama not running), so nothing was saved | Start Ollama (`ollama serve`), confirm the model exists (`ollama list`), run the benchmark again. |
| `Skipping N row(s) with missing or non-numeric values` | Some rows are damaged | Inspect `results.csv`. The script skips those rows and continues. |
| `ModuleNotFoundError: No module named 'pandas'` | Dependency missing | `py -m pip install pandas openpyxl` |
| `openpyxl is not installed` | Excel library missing | `py -m pip install openpyxl` (CSV is still exported). |
| `Cannot write ... Close it if it is open` | Output file is open in Excel | Close it and run again. |
| Run numbers repeat (1, 2, 3, 1, 2, 3) | Benchmark appends on every launch | Expected. Each launch becomes its own batch with its own AVG. Use `--latest-only` for the newest only. |
| Averages look high for one model | Cold-start load time in Run 1 | Compare `tokens_per_second`, or run more runs. |
| Benchmark script itself fails with `KeyError` | Ollama response lacks `load_duration`, `prompt_eval_count`, `eval_count` or `eval_duration` | Update Ollama. Those errors are in the benchmark, not the analysis. |

---

## 11. Recommended workflow

1. Start Ollama and make sure each model is pulled (`ollama list`).
2. Run the benchmark for each model with the same number of runs:
   ```bash
   py benchmark.py --model llama3.2:3b --runs 5
   py benchmark.py --model mistral:7b --runs 5
   ```
3. Run `py analyze_ollama_benchmark.py`.
4. Check the individual runs first (look for the cold-start Run 1 or outliers), then read the `AVG` rows.
5. Compare models on `tokens_per_second` (speed) and `latency_seconds` (user wait). Keep `generated_tokens` in mind, since longer answers take longer.
6. Open `results_analysis.xlsx` for reports or charts.
7. To start a fresh experiment, rename or move the old `results.csv` (for example `results_2026-11-24.csv`), so the next benchmark creates a new file. Do not edit the raw file by hand.
8. Keep the same hardware, the same prompt and no heavy background apps between runs so the comparison stays fair.