import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

import pytesseract
import easyocr
from PIL import Image

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Directories
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "test_documents"
OUTPUT_DIR = BASE_DIR / "results"

# Tesseract language configuration
TESSERACT_LANG = "eng+khm"


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
    cer = edit_distance(
        predicted, actual
    ) / len(actual)

    # Word Error Rate
    predicted_words = predicted.split()
    actual_words = actual.split()

    wer = (
        edit_distance(predicted_words, actual_words)
        / len(actual_words)
        if actual_words else None
    )

    return cer * 100, wer * 100


def run_tesseract(image_path):

    image = Image.open(image_path)

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


def benchmark_ocr(engine, runs):

    results = []

    if not IMAGE_DIR.exists():
        print("Test document folder does not exist.")
        return results

    image_files = [
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [
            ".png", ".jpg", ".jpeg", ".tif", ".tiff"
        ]
    ]

    if not image_files:
        print("No test images found.")
        return results

    # Initialize selected OCR engine
    reader = None

    if engine == "easyocr":
        print("\nInitializing EasyOCR...")
        reader = easyocr.Reader(["en"], gpu=False)

    elif engine != "tesseract":
        print("Unsupported OCR engine.")
        return results

    print(f"\nSelected OCR engine: {engine}")

    for image_path in image_files:

        print(f"\nProcessing: {image_path.name}")

        # Identify Khmer test images by filename
        is_khmer = "khmer" in image_path.stem.lower()

        if engine == "easyocr" and is_khmer:
            print("Skipping EasyOCR for Khmer image.")
            continue

        # Load ground truth text
        ground_truth_path = image_path.with_suffix(".txt")

        if ground_truth_path.exists():
            actual_text = ground_truth_path.read_text(
                encoding="utf-8"
            )
        else:
            actual_text = ""

            print("Warning: Ground truth file not found.")
            print("Accuracy metrics will be unavailable.")

        # Run benchmark
        for run in range(1, runs + 1):

            print(f"{engine} - Run {run}/{runs}")

            try:

                start_time = time.perf_counter()

                if engine == "tesseract":
                    extracted_text = run_tesseract(
                        image_path
                    )

                else:
                    extracted_text = run_easyocr(
                        reader,
                        image_path
                    )

                end_time = time.perf_counter()

                processing_time = end_time - start_time

                cer, wer = calculate_accuracy(
                    extracted_text,
                    actual_text
                )

                result = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "image": image_path.name,
                    "engine": engine,
                    "run": run,
                    "processing_time_seconds": round(
                        processing_time, 4
                    ),
                    "extracted_characters": len(
                        extracted_text
                    ),
                    "cer_percent": (
                        round(cer, 2)
                        if cer is not None else ""
                    ),
                    "wer_percent": (
                        round(wer, 2)
                        if wer is not None else ""
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

            except Exception as error:
                print(f"Error with {engine}: {error}")

    return results


def export_csv(results):

    if not results:
        print("No results to export.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "ocr_results.csv"

    file_exists = output_file.exists()

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        if not file_exists or output_file.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nResults exported to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="OCR Engine Benchmark"
    )

    parser.add_argument(
        "--engine",
        required=True,
        choices=["tesseract", "easyocr"],
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
        parser.error("--runs must be at least 1")

    results = benchmark_ocr(
        args.engine,
        args.runs
    )

    export_csv(results)
