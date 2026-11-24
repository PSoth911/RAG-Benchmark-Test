# Guide: `analyze_ocr_benchmark.py`

## 1. Purpose of the analysis script

Your OCR benchmark script (`ocr_benchmark.py`) runs **Tesseract** or **EasyOCR** on every image in `test_documents/`, several times per image, and appends one row per run to `results/ocr_results.csv`. Each row records processing time, how many characters were extracted, and (when a ground-truth `.txt` file exists) the Character Error Rate and Word Error Rate.

`analyze_ocr_benchmark.py` is a **separate** script that:

- reads `results/ocr_results.csv` (read-only, the file is never changed),
- shows **every individual run** (Run 1, Run 2, Run 3, ...),
- groups runs by **OCR engine**, then by **image**,
- adds an **`AVG` row** after the runs of each image,
- adds an **`ENGINE AVG` row** at the end of each engine,
- prints readable tables in the terminal,
- exports the same table to **CSV** and **Excel (`.xlsx`)**.

Averages are calculated from the raw values **before** any rounding. Rounding is applied only when displaying and exporting.

---

## 2. Project folder structure

```
your_project/
├── ocr_benchmark.py                  # your original benchmark script (unchanged)
├── analyze_ocr_benchmark.py          # analysis script
├── analyze_ocr_benchmark_guide.md
├── test_documents/
│   ├── invoice_english.png           # test image
│   ├── invoice_english.txt           # ground truth text (same name as the image)
│   ├── sign_khmer.png                # filename contains "khmer" -> EasyOCR skips it
│   └── sign_khmer.txt
└── results/
    ├── ocr_results.csv               # raw data, created by ocr_benchmark.py
    ├── ocr_results_analysis.csv      # created by the analysis script
    └── ocr_results_analysis.xlsx     # created by the analysis script
```

Keep `analyze_ocr_benchmark.py` in the same folder as `ocr_benchmark.py`. By default it looks for `results/ocr_results.csv` next to itself and writes its outputs into the same `results/` folder. If your files are elsewhere, use `--input` (see section 6). (The image names above are examples.)

---

## 3. Required dependencies

**For the analysis script:**

| Package | Why |
|---|---|
| `pandas` | read the CSV, group and average |
| `openpyxl` | write the `.xlsx` file |

```bash
py -m pip install pandas openpyxl
```

**For the benchmark script** (already needed, listed for completeness):

```bash
py -m pip install pytesseract easyocr pillow
```

The benchmark also requires the Tesseract program installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` with the English and Khmer language data (`eng+khm`).

Optional virtual environment:

```bash
py -m venv venv
venv\Scripts\activate           # Windows (macOS/Linux: source venv/bin/activate)
py -m pip install pandas openpyxl
```

---

## 4. Input CSV preparation and expected column names

You do not need to prepare anything. The analysis script reads the file exactly as your benchmark writes it. The header is:

```
timestamp,image,engine,run,processing_time_seconds,extracted_characters,cer_percent,wer_percent
```

Example raw rows:

```
2026-11-24T10:00:01,invoice_english.png,tesseract,1,1.2345,812,4.52,9.1
2026-11-24T10:00:03,invoice_english.png,tesseract,2,0.9876,812,4.52,9.1
2026-11-24T10:00:13,note_noground.png,tesseract,1,0.8,200,,
```

If a column is missing or renamed, the script stops and lists what it found and what it expected.

### Empty `cer_percent` and `wer_percent`

If an image has no matching `.txt` ground-truth file, your benchmark still records the run but leaves `cer_percent` and `wer_percent` **empty**. The analysis script keeps those rows (time and character count are still valid) and **ignores the empty CER/WER values when averaging**. They appear as blanks in the output. It prints a note telling you how many rows had no CER/WER.

Rows where `run`, `processing_time_seconds` or `extracted_characters` is missing or not a number are skipped with a warning.

### Important: the benchmark appends

`ocr_benchmark.py` **appends** to `ocr_results.csv` every time you launch it, and `run` restarts at 1 each launch. If you run Tesseract twice on the same image, the file contains two sets of Run 1, 2, 3.

The analysis script handles this by creating a **`batch`** column. A new batch starts for an engine + image pair whenever its `run` number does not increase. Each batch gets its own runs and its own `AVG` row, so separate sessions are never mixed into one image average. `batch` is calculated by the analysis script; it is not in `ocr_results.csv`. Use `--latest-only` to analyze only the most recent batch of each engine + image.

---

## 5. Complete analysis Python code

Save as `analyze_ocr_benchmark.py`:

```python
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
```

---

## 6. How to run the analysis script

First run the benchmark (examples):

```bash
py ocr_benchmark.py --engine tesseract --runs 3
py ocr_benchmark.py --engine easyocr --runs 3
```

Then analyze:

```bash
py analyze_ocr_benchmark.py
```

Options:

```bash
# Use a raw CSV in another location
py analyze_ocr_benchmark.py --input path/to/ocr_results.csv

# Only the latest batch of each engine + image
py analyze_ocr_benchmark.py --latest-only
```

---

## 7. Explanation of every benchmark metric

These are exactly the metrics your benchmark records.

| Column | Meaning | Better is | How the AVG is calculated |
|---|---|---|---|
| `timestamp` | Time the result was saved. Identifies the run, not a metric. | n/a | left empty in AVG rows |
| `image` | Test image file name. Used for grouping. | n/a | n/a |
| `engine` | OCR engine: `tesseract` or `easyocr`. Used for grouping. | n/a | n/a |
| `run` | Run number for that image within one launch (1, 2, 3, ...). | n/a | shown as `AVG` or `ENGINE AVG` |
| `processing_time_seconds` | Time to run OCR on the image, measured with `time.perf_counter()`. It covers the OCR call only (for Tesseract this includes opening the image). It does **not** include the CER/WER calculation or EasyOCR's start-up (`easyocr.Reader`), which happens once before the loop. | Lower | mean of runs |
| `extracted_characters` | Length of the text the engine returned (including spaces and line breaks). It is **not** an accuracy measure: a wrong but long result scores high. Use it to spot empty or truncated output. | n/a (compare with the ground-truth length) | mean of runs |
| `cer_percent` | **Character Error Rate** in %. The edit distance (insertions + deletions + substitutions) between the OCR text and the ground truth, divided by the ground-truth length in characters. Text is lower-cased and whitespace is normalized first. | Lower (0 = perfect) | mean of runs that have a value |
| `wer_percent` | **Word Error Rate** in %. Same calculation but on whole words (split by spaces). | Lower (0 = perfect) | mean of runs that have a value |

Notes on reading the results:

- **CER and WER can exceed 100%** when the OCR output has many extra characters or words.
- **WER for Khmer is not reliable.** Khmer is normally written without spaces between words, so splitting on spaces gives very few "words". Rely on CER for Khmer images.
- **Case is ignored**, because both texts are lower-cased before comparison.
- Run 1 may be slower than later runs (caches warming up), so check the individual runs, not only the average.
- **EasyOCR skips images whose file name contains `khmer`**, because your script initializes EasyOCR with English only. So the engines are not tested on identical image sets.
- **Rounding:** the benchmark already rounds `processing_time_seconds` to 4 decimals and CER/WER to 2 decimals when writing the CSV. The analysis script averages those stored values at full precision and rounds only the final display (4 decimals for time, 2 for the others).

### The two kinds of average rows

| Row | What it averages |
|---|---|
| `AVG` | The runs of **one image** for one engine (one batch). Use this to compare engines fairly on the same image. |
| `ENGINE AVG` (image = `ALL IMAGES`) | **Every raw run** of that engine across all its images. Images with more runs count more, and an engine that skipped some images (EasyOCR and Khmer) is averaged over a different set of images, so do not treat this as a perfectly fair engine-vs-engine score. |

---

## 8. Expected terminal output

The numbers below are **example values** to show the layout (two engines, one image without ground truth, and a Khmer image tested only with Tesseract). Your values will differ.

```
NOTE: 3 row(s) have no CER/WER (no ground-truth .txt file). They are kept; CER/WER averages ignore them.

====================================================================================================
OCR BENCHMARK ANALYSIS - individual runs with AVG rows
====================================================================================================

OCR engine: easyocr
              image        run processing_time_seconds extracted_characters cer_percent wer_percent
invoice_english.png          1                     6.5                  830         3.1         7.2
invoice_english.png          2                     4.1                  830         3.1         7.2
invoice_english.png          3                     4.0                  829         3.2         7.3
invoice_english.png        AVG                  4.8667               829.67        3.13        7.23
  note_noground.png          1                     3.0                  210                        
  note_noground.png        AVG                     3.0                210.0                        
         ALL IMAGES ENGINE AVG                     4.4               674.75        3.13        7.23

OCR engine: tesseract
              image        run processing_time_seconds extracted_characters cer_percent wer_percent
invoice_english.png          1                  1.2345                  812        4.52         9.1
invoice_english.png          2                  0.9876                  812        4.52         9.1
invoice_english.png          3                  1.0012                  810         4.6         9.3
invoice_english.png        AVG                  1.0744               811.33        4.55        9.17
  note_noground.png          1                     0.8                  200                        
  note_noground.png          2                     0.7                  200                        
  note_noground.png        AVG                    0.75                200.0                        
     sign_khmer.png          1                     2.5                  150        35.2        80.5
     sign_khmer.png          2                     2.4                  150        35.2        80.5
     sign_khmer.png        AVG                    2.45                150.0        35.2        80.5
         ALL IMAGES ENGINE AVG                  1.3748               447.71       16.81        37.7

====================================================================================================

SUMMARY (ENGINE AVG rows only)
   engine processing_time_seconds extracted_characters cer_percent wer_percent
  easyocr                     4.4               674.75        3.13        7.23
tesseract                  1.3748               447.71       16.81        37.7

Note: engines may have been tested on different images (for example EasyOCR skips Khmer images),
so compare engines on the same images using the per-image AVG rows.

CSV saved:   results/ocr_results_analysis.csv
Excel saved: results/ocr_results_analysis.xlsx
Raw file left unchanged: results/ocr_results.csv
```

If an engine + image pair appears in more than one launch, a `batch` column appears in the tables and each batch has its own `AVG` row.

---

## 9. Generated output files

| File | Content |
|---|---|
| `results/ocr_results_analysis.csv` | Every run, an `AVG` row per image, and an `ENGINE AVG` row per engine. |
| `results/ocr_results_analysis.xlsx` | Same table in sheet **Analysis**, with bold header, frozen header row, auto-sized columns, `AVG` rows highlighted light blue and `ENGINE AVG` rows highlighted light yellow. |

Columns in both files:

```
engine, image, batch, run, timestamp, processing_time_seconds, extracted_characters, cer_percent, wer_percent
```

Example CSV rows:

```
engine,image,batch,run,timestamp,processing_time_seconds,extracted_characters,cer_percent,wer_percent
easyocr,invoice_english.png,1,1,2026-11-24T10:10:01,6.5,830,3.1,7.2
easyocr,invoice_english.png,1,2,2026-11-24T10:10:08,4.1,830,3.1,7.2
easyocr,invoice_english.png,1,3,2026-11-24T10:10:12,4.0,829,3.2,7.3
easyocr,invoice_english.png,1,AVG,,4.8667,829.67,3.13,7.23
easyocr,ALL IMAGES,,ENGINE AVG,,4.4,674.75,3.13,7.23
```

The analysis files are overwritten each time you run the script. `ocr_results.csv` is never written to.

---

## 10. Troubleshooting

| Message / problem | Cause | Fix |
|---|---|---|
| `Input file not found` | `results/ocr_results.csv` does not exist where the script looks | Run the benchmark first, or use `--input path/to/ocr_results.csv`. |
| `Missing column(s) ...` | CSV header differs from the benchmark's header | Compare with section 4. Don't edit the header by hand; regenerate with the benchmark. |
| `is empty` / `has a header but no benchmark rows` | Every run failed or no images were found | Check that `test_documents/` exists and contains `.png`, `.jpg`, `.jpeg`, `.tif` or `.tiff` files, and that Tesseract is installed. |
| `Skipping N row(s) with missing or non-numeric ...` | Some rows are damaged | Inspect the raw CSV. The script skips them and continues. |
| `NOTE: N row(s) have no CER/WER` | Some images have no `.txt` ground-truth file | Add a `.txt` file with the same name as the image (for example `invoice.png` and `invoice.txt`) and rerun the benchmark. |
| CER/WER columns are blank for an image | Same as above | Same as above. |
| `ModuleNotFoundError: No module named 'pandas'` | Dependency missing | `py -m pip install pandas openpyxl` |
| `openpyxl is not installed` | Excel library missing | `py -m pip install openpyxl` (CSV is still exported). |
| `Cannot write ... Close it if it is open` | Output file is open in Excel | Close it and run again. |
| Run numbers repeat (1, 2, 3, 1, 2, 3) for an image | Benchmark appends on every launch | Expected. Each launch becomes its own batch with its own AVG. Use `--latest-only` for the newest only. |
| EasyOCR has no Khmer images | Your benchmark skips them on purpose | Expected. Compare engines using only the images both tested. |
| Very high CER/WER | Wrong language, low image quality, or ground truth does not match the image text | Check the ground-truth `.txt` and the OCR language settings (`eng+khm` for Tesseract, English only for EasyOCR). |

---

## 11. Recommended workflow

1. Put test images in `test_documents/` and add a ground-truth `.txt` file with the same name for each image.
2. Run the benchmark for each engine with the same number of runs:
   ```bash
   py ocr_benchmark.py --engine tesseract --runs 5
   py ocr_benchmark.py --engine easyocr --runs 5
   ```
3. Run `py analyze_ocr_benchmark.py`.
4. Check the individual runs first (look for slow first runs or outliers), then read the `AVG` rows.
5. Compare engines **per image** using `AVG` rows: accuracy (`cer_percent`, `wer_percent`) first, then speed (`processing_time_seconds`).
6. Use `ENGINE AVG` only as a rough overview, because the engines may have been tested on different images.
7. Open `results/ocr_results_analysis.xlsx` for reports or charts.
8. To start a fresh experiment, rename or move the old `ocr_results.csv` (for example `ocr_results_2026-11-24.csv`) so the next benchmark creates a new file. Do not edit the raw file by hand.
9. Keep the same computer and no heavy background apps during benchmarking so timing stays fair.