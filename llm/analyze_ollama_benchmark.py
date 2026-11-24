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