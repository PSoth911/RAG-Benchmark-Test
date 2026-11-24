"""
analyze_ocr_benchmark.py

Reads the raw results/ocr_results.csv written by the OCR benchmark script
(Tesseract vs EasyOCR), shows every individual run grouped by OCR engine and
image, adds an AVG row after the runs of each image, adds an ENGINE AVG row
for each engine, prints the tables in the terminal, and exports CSV + Excel.

The raw ocr_results.csv is only READ. It is never modified.

Usage:
    py analyze_ocr_benchmark.py
    py analyze_ocr_benchmark.py --input results/ocr_results.csv
    py analyze_ocr_benchmark.py --latest-only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings (these match the columns written by the OCR benchmark script)
# ---------------------------------------------------------------------------
SCRIPT_FOLDER = Path(__file__).parent

DEFAULT_INPUT_FILE = SCRIPT_FOLDER / "results" / "ocr_results.csv"
OUTPUT_CSV_FILE = SCRIPT_FOLDER / "results" / "ocr_results_analysis.csv"
OUTPUT_EXCEL_FILE = SCRIPT_FOLDER / "results" / "ocr_results_analysis.xlsx"

ID_COLUMNS = ["timestamp", "image", "engine", "run"]

# Metrics that must always have a number
REQUIRED_METRICS = ["processing_time_seconds", "extracted_characters"]

# Metrics that can be EMPTY in the raw CSV (when no ground-truth .txt file
# existed, the benchmark leaves cer_percent and wer_percent blank)
OPTIONAL_METRICS = ["cer_percent", "wer_percent"]

METRIC_COLUMNS = REQUIRED_METRICS + OPTIONAL_METRICS
REQUIRED_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

# Decimal places used ONLY for display/export (never before averaging)
DECIMALS = {
    "processing_time_seconds": 4,
    "extracted_characters": 2,
    "cer_percent": 2,
    "wer_percent": 2,
}

AVG_LABELS = ["AVG", "ENGINE AVG"]


# ---------------------------------------------------------------------------
# Step 1: Load and validate the raw CSV
# ---------------------------------------------------------------------------
def load_raw_results(csv_path):
    """Read the raw CSV and check that it is usable. Exits with a message if not."""
    if not csv_path.exists():
        sys.exit(
            f"ERROR: Input file not found: {csv_path}\n"
            "Run the OCR benchmark first so it creates results/ocr_results.csv, "
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

    # Convert numbers. Bad or empty values become NaN.
    for column in ["run"] + METRIC_COLUMNS:
        raw_data[column] = pd.to_numeric(raw_data[column], errors="coerce")

    # Rows missing a REQUIRED number are unusable. Empty CER/WER is allowed.
    bad_rows = raw_data[["run"] + REQUIRED_METRICS].isna().any(axis=1)
    if bad_rows.any():
        print(f"WARNING: Skipping {bad_rows.sum()} row(s) with missing or "
              "non-numeric run/time/character values.")
        raw_data = raw_data[~bad_rows]

    if raw_data.empty:
        sys.exit("ERROR: No valid rows left after removing bad rows.")

    missing_accuracy = raw_data[OPTIONAL_METRICS].isna().any(axis=1).sum()
    if missing_accuracy:
        print(f"NOTE: {missing_accuracy} row(s) have no CER/WER "
              "(no ground-truth .txt file). They are kept; CER/WER averages "
              "ignore them.")

    raw_data["run"] = raw_data["run"].astype(int)
    return raw_data.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Split rows into batches
# ---------------------------------------------------------------------------
def add_batch_numbers(data):
    """
    The benchmark script APPENDS to ocr_results.csv, and 'run' restarts at 1
    every time you launch it. So one engine + image can appear in several
    sessions (batches). A new batch starts for an (engine, image) pair
    whenever its run number does not increase.
    """
    batch_numbers = []
    last_run = {}
    current_batch = {}

    for _, row in data.iterrows():
        key = (row["engine"], row["image"])
        if key not in current_batch:
            current_batch[key] = 1
        elif row["run"] <= last_run[key]:
            current_batch[key] += 1
        last_run[key] = row["run"]
        batch_numbers.append(current_batch[key])

    data = data.copy()
    data["batch"] = batch_numbers
    return data


# ---------------------------------------------------------------------------
# Step 3: Build the detailed table with AVG rows
# ---------------------------------------------------------------------------
def build_detailed_table(data, latest_only):
    """
    For each engine:
      - for each image (and batch): individual runs, then an AVG row
      - then one ENGINE AVG row (mean of all raw runs of that engine)
    All averages use raw, unrounded values. Empty CER/WER are ignored.
    """
    if latest_only:
        latest = data.groupby(["engine", "image"])["batch"].transform("max")
        data = data[data["batch"] == latest]

    table_parts = []

    for engine_name in sorted(data["engine"].unique()):
        engine_data = data[data["engine"] == engine_name]

        for image_name in sorted(engine_data["image"].unique()):
            image_data = engine_data[engine_data["image"] == image_name]

            for batch_number in sorted(image_data["batch"].unique()):
                batch_data = image_data[image_data["batch"] == batch_number]

                runs = batch_data.copy()
                runs["run"] = runs["run"].astype(str)

                avg_row = {
                    "timestamp": "",
                    "image": image_name,
                    "engine": engine_name,
                    "batch": batch_number,
                    "run": "AVG",
                }
                for metric in METRIC_COLUMNS:
                    # mean() skips empty (NaN) values automatically
                    avg_row[metric] = batch_data[metric].mean()

                table_parts.append(runs)
                table_parts.append(pd.DataFrame([avg_row]))

        # Average of every raw run of this engine (all images together)
        engine_row = {
            "timestamp": "",
            "image": "ALL IMAGES",
            "engine": engine_name,
            "batch": "",
            "run": "ENGINE AVG",
        }
        for metric in METRIC_COLUMNS:
            engine_row[metric] = engine_data[metric].mean()
        table_parts.append(pd.DataFrame([engine_row]))

    detailed = pd.concat(table_parts, ignore_index=True)
    column_order = ["engine", "image", "batch", "run", "timestamp"] + METRIC_COLUMNS
    return detailed[column_order]


# ---------------------------------------------------------------------------
# Step 4: Rounding (display/export only)
# ---------------------------------------------------------------------------
def round_for_display(detailed):
    rounded = detailed.copy()
    is_average = rounded["run"].isin(AVG_LABELS)

    for metric, places in DECIMALS.items():
        rounded[metric] = rounded[metric].round(places)

    # Individual character counts are whole numbers: show without ".0"
    rounded["extracted_characters"] = rounded["extracted_characters"].astype(object)
    rounded.loc[~is_average, "extracted_characters"] = (
        detailed.loc[~is_average, "extracted_characters"].astype(int)
    )

    # Empty values (no ground truth) are shown as blanks instead of "NaN"
    for metric in METRIC_COLUMNS:
        rounded[metric] = rounded[metric].astype(object)
        rounded.loc[detailed[metric].isna(), metric] = ""

    return rounded


# ---------------------------------------------------------------------------
# Step 5: Print in the terminal
# ---------------------------------------------------------------------------
def print_table(rounded):
    print("\n" + "=" * 100)
    print("OCR BENCHMARK ANALYSIS - individual runs with AVG rows")
    print("=" * 100)

    show_batch = rounded.loc[rounded["run"] == "AVG", "batch"].nunique() > 1
    columns = ["image"] + (["batch"] if show_batch else []) + \
              ["run"] + METRIC_COLUMNS

    for engine_name in rounded["engine"].unique():
        engine_rows = rounded[rounded["engine"] == engine_name]
        print(f"\nOCR engine: {engine_name}")
        print(engine_rows[columns].to_string(index=False))

    print("\n" + "=" * 100)


def print_summary(rounded):
    engine_averages = rounded[rounded["run"] == "ENGINE AVG"]
    summary = engine_averages[["engine"] + METRIC_COLUMNS]
    print("\nSUMMARY (ENGINE AVG rows only)")
    print(summary.to_string(index=False))
    print("\nNote: engines may have been tested on different images "
          "(for example EasyOCR skips Khmer images),")
    print("so compare engines on the same images using the per-image AVG rows.\n")


# ---------------------------------------------------------------------------
# Step 6: Export CSV and Excel
# ---------------------------------------------------------------------------
def export_csv(rounded, path):
    try:
        path.parent.mkdir(exist_ok=True)
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
        path.parent.mkdir(exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            rounded.to_excel(writer, sheet_name="Analysis", index=False)
            sheet = writer.sheets["Analysis"]

            for cell in sheet[1]:
                cell.font = Font(bold=True)
            sheet.freeze_panes = "A2"

            run_column_index = list(rounded.columns).index("run") + 1
            fills = {
                "AVG": PatternFill("solid", fgColor="DDEBF7"),
                "ENGINE AVG": PatternFill("solid", fgColor="FFF2CC"),
            }
            for row_number in range(2, sheet.max_row + 1):
                label = sheet.cell(row_number, run_column_index).value
                if label in fills:
                    for cell in sheet[row_number]:
                        cell.font = Font(bold=True)
                        cell.fill = fills[label]

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
    parser = argparse.ArgumentParser(description="Analyze OCR benchmark results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE,
                        help="Raw benchmark CSV (default: results/ocr_results.csv)")
    parser.add_argument("--latest-only", action="store_true",
                        help="Only analyze the most recent batch of each engine/image")
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