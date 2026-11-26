import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pytesseract
import easyocr
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "test_documents"
OUTPUT_DIR = BASE_DIR / "results"

RAW_OUTPUT = OUTPUT_DIR / "ocr_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "ocr_summary.csv"

TESSERACT_CMD = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSERACT_LANG = "eng+khm"

SUPPORTED_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
]


# ============================================================
# CSV SCHEMAS
# ============================================================

RAW_FIELDS = [
    "timestamp",
    "image",
    "language",
    "engine",
    "run",
    "processing_time_seconds",
    "extracted_characters",
    "cer_percent",
    "wer_percent",
    "initialization_seconds",
    "warmup_seconds"
]

SUMMARY_FIELDS = [
    "engine",
    "image",
    "language",
    "total_runs",
    "avg_processing_time_seconds",
    "min_processing_time_seconds",
    "max_processing_time_seconds",
    "avg_extracted_characters",
    "avg_cer_percent",
    "avg_wer_percent"
]


# ============================================================
# TEXT EVALUATION
# ============================================================

def edit_distance(a, b):
    """Calculate Levenshtein edit distance."""

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            cost = 0 if char_a == char_b else 1

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )

        previous = current

    return previous[-1]


def calculate_accuracy(predicted, actual):
    """Calculate Character Error Rate and Word Error Rate."""

    predicted = " ".join(predicted.lower().split())
    actual = " ".join(actual.lower().split())

    if not actual:
        return None, None

    # Character Error Rate
    cer = (
        edit_distance(predicted, actual)
        / len(actual)
    ) * 100

    # Word Error Rate
    predicted_words = predicted.split()
    actual_words = actual.split()

    wer = (
        edit_distance(predicted_words, actual_words)
        / len(actual_words)
    ) * 100

    return cer, wer


# ============================================================
# OCR ENGINE INITIALIZATION
# ============================================================

def initialize_ocr(engine):

    initialization_start = time.perf_counter()

    reader = None

    if engine == "tesseract":

        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_CMD
        )

    elif engine == "easyocr":

        print("\nInitializing EasyOCR...")

        reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

    initialization_end = time.perf_counter()

    initialization_time = (
        initialization_end - initialization_start
    )

    print(
        f"Initialization time: "
        f"{initialization_time:.4f} seconds"
    )

    return reader, initialization_time


# ============================================================
# OCR EXECUTION
# ============================================================

def run_tesseract(image_path):

    image = Image.open(image_path)

    try:
        return pytesseract.image_to_string(
            image,
            lang=TESSERACT_LANG
        )

    finally:
        image.close()


def run_easyocr(reader, image_path):

    results = reader.readtext(
        str(image_path),
        detail=0,
        paragraph=True
    )

    return "\n".join(results)


def run_ocr(engine, reader, image_path):

    if engine == "tesseract":

        return run_tesseract(image_path)

    elif engine == "easyocr":

        return run_easyocr(
            reader,
            image_path
        )

    raise ValueError(f"Unsupported engine: {engine}")


# ============================================================
# WARM-UP
# ============================================================

def warmup_ocr(engine, reader, image_files):

    if not image_files:
        return 0.0

    print("\nRunning OCR warm-up...")

    warmup_start = time.perf_counter()

    run_ocr(
        engine,
        reader,
        image_files[0]
    )

    warmup_end = time.perf_counter()

    warmup_time = warmup_end - warmup_start

    print(
        f"Warm-up completed: "
        f"{warmup_time:.4f} seconds"
    )

    return warmup_time


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():

    if not IMAGE_DIR.exists():

        print("Test document folder does not exist.")

        return []

    image_files = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:

        print("No test images found.")

        return []

    return image_files


def detect_language(image_path):

    if "khmer" in image_path.stem.lower():

        return "khmer"

    return "english"


def load_ground_truth(image_path):

    ground_truth_path = image_path.with_suffix(".txt")

    if not ground_truth_path.exists():

        print(
            f"Warning: Ground truth missing for "
            f"{image_path.name}"
        )

        return ""

    return ground_truth_path.read_text(
        encoding="utf-8-sig"
    )


# ============================================================
# BENCHMARK EXECUTION
# ============================================================

def benchmark_ocr(engine, runs):

    results = []

    image_files = load_dataset()

    if not image_files:
        return results

    # EasyOCR only supports English in this experiment.
    if engine == "easyocr":

        image_files = [
            image
            for image in image_files
            if detect_language(image) != "khmer"
        ]

        print(
            "\nEasyOCR English-only configuration."
        )

        print(
            "Khmer images will be skipped."
        )

    if not image_files:

        print("No compatible images found.")

        return results

    print(f"\nSelected OCR engine: {engine}")

    print(f"Total images: {len(image_files)}")

    print(f"Benchmark runs: {runs}")

    # Initialize OCR engine outside processing timer.
    reader, initialization_time = initialize_ocr(
        engine
    )

    # Perform warm-up outside benchmark measurements.
    warmup_time = warmup_ocr(
        engine,
        reader,
        image_files
    )

    # Process every image.
    for image_index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\n[{image_index}/{len(image_files)}] "
            f"Processing: {image_path.name}"
        )

        language = detect_language(image_path)

        actual_text = load_ground_truth(image_path)

        for run in range(1, runs + 1):

            print(
                f"{engine} - Run {run}/{runs}"
            )

            try:

                # Measure OCR processing time only.
                start_time = time.perf_counter()

                extracted_text = run_ocr(
                    engine,
                    reader,
                    image_path
                )

                end_time = time.perf_counter()

                processing_time = (
                    end_time - start_time
                )

                # Evaluate OCR accuracy.
                cer, wer = calculate_accuracy(
                    extracted_text,
                    actual_text
                )

                result = {

                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),

                    "image": image_path.name,

                    "language": language,

                    "engine": engine,

                    "run": run,

                    "processing_time_seconds": round(
                        processing_time,
                        4
                    ),

                    "extracted_characters": len(
                        extracted_text
                    ),

                    "cer_percent": (
                        round(cer, 2)
                        if cer is not None
                        else ""
                    ),

                    "wer_percent": (
                        round(wer, 2)
                        if wer is not None
                        else ""
                    ),

                    "initialization_seconds": round(
                        initialization_time,
                        4
                    ),

                    "warmup_seconds": round(
                        warmup_time,
                        4
                    )
                }

                results.append(result)

                print("Completed!")

                print(
                    f"Processing time: "
                    f"{processing_time:.4f} seconds"
                )

                if cer is not None:

                    print(f"CER: {cer:.2f}%")

                    print(f"WER: {wer:.2f}%")

                else:

                    print(
                        "Accuracy unavailable: "
                        "No ground truth."
                    )

            except Exception as error:

                print(
                    f"Error processing "
                    f"{image_path.name}: {error}"
                )

    return results


# ============================================================
# SUMMARY GENERATION
# ============================================================

def calculate_average(values):

    valid_values = [
        value
        for value in values
        if value != "" and value is not None
    ]

    if not valid_values:
        return ""

    return round(
        sum(valid_values) / len(valid_values),
        4
    )


def generate_summary(results):

    grouped_results = defaultdict(list)

    for result in results:

        key = (
            result["engine"],
            result["image"],
            result["language"]
        )

        grouped_results[key].append(result)

    summary = []

    for key, group in grouped_results.items():

        engine, image, language = key

        processing_times = [
            row["processing_time_seconds"]
            for row in group
        ]

        character_counts = [
            row["extracted_characters"]
            for row in group
        ]

        cer_values = [
            row["cer_percent"]
            for row in group
        ]

        wer_values = [
            row["wer_percent"]
            for row in group
        ]

        summary.append({

            "engine": engine,

            "image": image,

            "language": language,

            "total_runs": len(group),

            "avg_processing_time_seconds": (
                calculate_average(processing_times)
            ),

            "min_processing_time_seconds": round(
                min(processing_times),
                4
            ),

            "max_processing_time_seconds": round(
                max(processing_times),
                4
            ),

            "avg_extracted_characters": (
                calculate_average(character_counts)
            ),

            "avg_cer_percent": (
                calculate_average(cer_values)
            ),

            "avg_wer_percent": (
                calculate_average(wer_values)
            )
        })

    return summary


# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(results, output_file, fields):

    if not results:

        print("No results to export.")

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        output_file.exists()
        and output_file.stat().st_size > 0
    )

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nExported: {output_file}")


def export_summary(results):

    if not results:
        return

    summary = generate_summary(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Summary represents the current benchmark execution.
    with open(
        SUMMARY_OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS
        )

        writer.writeheader()

        writer.writerows(summary)

    print(f"Summary exported: {SUMMARY_OUTPUT}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="OCR Engine Benchmark"
    )

    parser.add_argument(
        "--engine",
        required=True,
        choices=[
            "tesseract",
            "easyocr"
        ],
        help="OCR engine to benchmark"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs"
    )

    args = parser.parse_args()

    if args.runs < 1:

        parser.error(
            "--runs must be at least 1"
        )

    results = benchmark_ocr(
        args.engine,
        args.runs
    )

    if results:

        # Export individual benchmark results.
        export_csv(
            results,
            RAW_OUTPUT,
            RAW_FIELDS
        )

        # Export average benchmark results.
        export_summary(results)

        print("\nOCR BENCHMARK COMPLETED!")

    else:

        print("\nNo benchmark results generated.")
