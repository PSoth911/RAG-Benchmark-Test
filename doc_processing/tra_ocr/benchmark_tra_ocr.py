
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

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
OUTPUT_DIR = BASE_DIR / "results"

RAW_OUTPUT = OUTPUT_DIR / "ocr_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "ocr_summary.csv"
EXTRACTED_DIR = OUTPUT_DIR / "extracted"

TESSERACT_CMD = "tesseract"
TESSERACT_LANG = "eng+ben"

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}

# EasyOCR language configuration.
# Change to ["en"] if you only want English.
EASYOCR_LANGUAGES = ["en"]

RAW_FIELDS = [
    "timestamp",
    "engine",
    "image",
    "language",
    "run",
    "processing_time_seconds",
    "extracted_characters",
    "cer_percent",
    "wer_percent",
    "initialization_seconds",
    "warmup_seconds",
    "status"
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
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1

            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    return previous[-1]


def normalize_text(text):
    return " ".join(text.lower().split())


def calculate_accuracy(predicted, actual):
    predicted = normalize_text(predicted)
    actual = normalize_text(actual)

    if not actual:
        return None, None

    cer = edit_distance(predicted, actual) / len(actual) * 100

    predicted_words = predicted.split()
    actual_words = actual.split()

    wer = (
        edit_distance(predicted_words, actual_words)
        / len(actual_words)
    ) * 100

    return cer, wer


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    images = sorted(
        path for path in DATASET_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    return images


def detect_language(image_path):
    name = image_path.stem.lower()

    if "khmer" in name:
        return "khmer"

    return "english"


def load_ground_truth(image_path):
    txt_path = image_path.with_suffix(".txt")

    if not txt_path.exists():
        return ""

    return txt_path.read_text(encoding="utf-8-sig")


# ============================================================
# OCR INITIALIZATION
# ============================================================

def initialize_ocr(engine):
    start = time.perf_counter()
    reader = None

    if engine == "tesseract":
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        # Verify executable and language data.
        pytesseract.get_tesseract_version()

        available = set(
            pytesseract.get_languages(config="")
        )

        required = set(TESSERACT_LANG.split("+"))
        missing = required - available

        if missing:
            raise RuntimeError(
                f"Missing Tesseract language data: {missing}"
            )

    elif engine == "easyocr":
        print("Initializing EasyOCR...")

        reader = easyocr.Reader(
            EASYOCR_LANGUAGES,
            gpu=True
        )

    else:
        raise ValueError(f"Unknown engine: {engine}")

    initialization_time = time.perf_counter() - start

    return reader, initialization_time


# ============================================================
# OCR EXECUTION
# ============================================================

def run_tesseract(image_path):
    with Image.open(image_path) as image:
        return pytesseract.image_to_string(
            image,
            lang=TESSERACT_LANG
        )


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

    if engine == "easyocr":
        return run_easyocr(reader, image_path)

    raise ValueError(f"Unsupported engine: {engine}")


# ============================================================
# WARM-UP
# ============================================================

def warmup_ocr(engine, reader, images):
    if not images:
        return 0.0

    print("Running warm-up...")

    start = time.perf_counter()

    run_ocr(engine, reader, images[0])

    elapsed = time.perf_counter() - start

    print(f"Warm-up: {elapsed:.4f} seconds")

    return elapsed


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_extracted_text(engine, image_path, run, text):
    folder = EXTRACTED_DIR / engine
    folder.mkdir(parents=True, exist_ok=True)

    output_file = folder / f"{image_path.stem}_run{run}.txt"

    output_file.write_text(text, encoding="utf-8")


def export_csv(rows, output_file, fields):
    if not rows:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exists = output_file.exists() and output_file.stat().st_size > 0

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)

    print(f"Exported: {output_file}")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_ocr(engine, runs):
    images = load_dataset()

    if not images:
        print("No images found.")
        return []

    # EasyOCR configuration is English-only in this experiment.
    if engine == "easyocr":
        images = [
            image for image in images
            if detect_language(image) != "khmer"
        ]

    if not images:
        print("No compatible images found.")
        return []

    print(f"\nEngine: {engine}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Images: {len(images)}")
    print(f"Runs per image: {runs}")

    reader, initialization_time = initialize_ocr(engine)

    print(
        f"Initialization: {initialization_time:.4f} seconds"
    )

    warmup_time = warmup_ocr(engine, reader, images)

    results = []

    for index, image_path in enumerate(images, start=1):
        language = detect_language(image_path)
        actual_text = load_ground_truth(image_path)

        print(
            f"\n[{index}/{len(images)}] {image_path.name}"
        )

        if not actual_text:
            print("WARNING: Ground truth missing.")

        for run in range(1, runs + 1):
            print(f"Run {run}/{runs}")

            try:
                start = time.perf_counter()

                extracted = run_ocr(
                    engine,
                    reader,
                    image_path
                )

                processing_time = time.perf_counter() - start

                cer, wer = calculate_accuracy(
                    extracted,
                    actual_text
                )

                save_extracted_text(
                    engine,
                    image_path,
                    run,
                    extracted
                )

                result = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "engine": engine,
                    "image": image_path.name,
                    "language": language,
                    "run": run,
                    "processing_time_seconds": round(
                        processing_time, 4
                    ),
                    "extracted_characters": len(extracted),
                    "cer_percent": (
                        round(cer, 2) if cer is not None else ""
                    ),
                    "wer_percent": (
                        round(wer, 2) if wer is not None else ""
                    ),
                    "initialization_seconds": round(
                        initialization_time, 4
                    ),
                    "warmup_seconds": round(warmup_time, 4),
                    "status": "success"
                }

                results.append(result)

                print(f"Time: {processing_time:.4f}s")

                if cer is not None:
                    print(f"CER: {cer:.2f}%")
                    print(f"WER: {wer:.2f}%")

            except Exception as error:
                print(f"ERROR: {error}")

                results.append({
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "engine": engine,
                    "image": image_path.name,
                    "language": language,
                    "run": run,
                    "processing_time_seconds": "",
                    "extracted_characters": "",
                    "cer_percent": "",
                    "wer_percent": "",
                    "initialization_seconds": round(
                        initialization_time, 4
                    ),
                    "warmup_seconds": round(warmup_time, 4),
                    "status": f"error: {error}"
                })

    return results


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(results):
    grouped = defaultdict(list)

    for row in results:
        if row["status"] != "success":
            continue

        key = (
            row["engine"],
            row["image"],
            row["language"]
        )

        grouped[key].append(row)

    summary = []

    for (engine, image, language), rows in grouped.items():

        def average(field):
            values = [
                row[field] for row in rows
                if row[field] != ""
            ]

            return round(sum(values) / len(values), 4) if values else ""

        times = [
            row["processing_time_seconds"] for row in rows
        ]

        summary.append({
            "engine": engine,
            "image": image,
            "language": language,
            "total_runs": len(rows),
            "avg_processing_time_seconds": average(
                "processing_time_seconds"
            ),
            "min_processing_time_seconds": round(min(times), 4),
            "max_processing_time_seconds": round(max(times), 4),
            "avg_extracted_characters": average(
                "extracted_characters"
            ),
            "avg_cer_percent": average("cer_percent"),
            "avg_wer_percent": average("wer_percent")
        })

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Traditional OCR Benchmark"
    )

    parser.add_argument(
        "--engine",
        required=True,
        choices=["tesseract", "easyocr", "all"]
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    engines = (
        ["tesseract", "easyocr"]
        if args.engine == "all"
        else [args.engine]
    )

    all_results = []

    try:
        for engine in engines:
            results = benchmark_ocr(engine, args.runs)
            all_results.extend(results)

    except Exception as error:
        print(f"\nBenchmark initialization failed: {error}")
        raise SystemExit(1)

    if all_results:
        export_csv(
            all_results,
            RAW_OUTPUT,
            RAW_FIELDS
        )

        summary = generate_summary(all_results)

        if summary:
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

        print("\nTRADITIONAL OCR BENCHMARK COMPLETED!")

    else:
        print("\nNo benchmark results generated.")
