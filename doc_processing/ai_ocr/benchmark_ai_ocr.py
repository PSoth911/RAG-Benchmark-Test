
import csv
import time
import argparse
import base64
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
OUTPUT_DIR = BASE_DIR / "results"

RAW_OUTPUT = OUTPUT_DIR / "ai_ocr_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "ai_ocr_summary.csv"
EXTRACTED_DIR = OUTPUT_DIR / "extracted"

OLLAMA_URL = "http://localhost:11434"

DEFAULT_MODELS = [
    "qwen2.5vl:7b",
    "qwen3-vl:8b"
]

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}

# OCR instruction sent to vision-language models.
OCR_PROMPT = """
You are an OCR system.

Extract all visible text from the provided document image.

Instructions:
1. Transcribe the text exactly as it appears.
2. Preserve the original language.
3. Support English and Khmer text.
4. Preserve the reading order from top to bottom.
5. Preserve paragraphs and line breaks where possible.
6. Do not summarize or explain the document.
7. Do not translate the text.
8. Do not add information that is not visible.
9. If a word is unreadable, do not guess it.

Return only the extracted document text.
"""

RAW_FIELDS = [
    "timestamp",
    "model",
    "image",
    "language",
    "run",
    "processing_time_seconds",
    "extracted_characters",
    "cer_percent",
    "wer_percent",
    "prompt_tokens",
    "generated_tokens",
    "tokens_per_second",
    "initialization_seconds",
    "warmup_seconds",
    "status"
]

SUMMARY_FIELDS = [
    "model",
    "image",
    "language",
    "total_runs",
    "avg_processing_time_seconds",
    "min_processing_time_seconds",
    "max_processing_time_seconds",
    "avg_extracted_characters",
    "avg_cer_percent",
    "avg_wer_percent",
    "avg_tokens_per_second"
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

    return sorted(
        path for path in DATASET_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def detect_language(image_path):
    if "khmer" in image_path.stem.lower():
        return "khmer"

    return "english"


def load_ground_truth(image_path):
    txt_path = image_path.with_suffix(".txt")

    if not txt_path.exists():
        return ""

    return txt_path.read_text(encoding="utf-8-sig")


# ============================================================
# OLLAMA CONNECTION
# ============================================================

def check_ollama():
    start = time.perf_counter()

    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=15
    )

    response.raise_for_status()

    initialization_time = time.perf_counter() - start

    installed_models = [
        item["name"]
        for item in response.json().get("models", [])
    ]

    print("\nOllama connection successful.")

    print("Installed models:")

    for model in installed_models:
        print(f"  - {model}")

    return initialization_time, installed_models


def resolve_model_name(requested_model, installed_models):
    """
    Match a requested model against the actual Ollama tags.
    """

    if requested_model in installed_models:
        return requested_model

    requested_base = requested_model.split(":")[0]

    for installed in installed_models:
        installed_base = installed.split(":")[0]

        if installed_base == requested_base:
            return installed

    raise ValueError(
        f"Model '{requested_model}' is not installed.\n"
        f"Available models: {installed_models}"
    )


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(
            image.read()
        ).decode("utf-8")


# ============================================================
# AI OCR EXECUTION
# ============================================================

def run_ai_ocr(model, image_path):

    encoded_image = encode_image(image_path)

    payload = {
        "model": model,

        "messages": [
            {
                "role": "user",
                "content": OCR_PROMPT,
                "images": [encoded_image]
            }
        ],

        "stream": False,

        "think": False,

        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    data = response.json()

    extracted_text = data.get(
        "message", {}
    ).get("content", "")

    # Ollama reports durations in nanoseconds.
    prompt_tokens = data.get("prompt_eval_count", 0)
    generated_tokens = data.get("eval_count", 0)

    eval_duration = data.get("eval_duration", 0)

    tokens_per_second = 0

    if eval_duration > 0:
        tokens_per_second = (
            generated_tokens / (eval_duration / 1_000_000_000)
        )

    return {
        "text": extracted_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second
    }


# ============================================================
# WARM-UP
# ============================================================

def warmup_model(model, images):
    if not images:
        return 0.0

    print(f"\nWarming up model: {model}")

    start = time.perf_counter()

    run_ai_ocr(model, images[0])

    warmup_time = time.perf_counter() - start

    print(f"Warm-up: {warmup_time:.4f} seconds")

    return warmup_time


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_extracted_text(model, image_path, run, text):
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)

    folder = EXTRACTED_DIR / safe_model
    folder.mkdir(parents=True, exist_ok=True)

    output_file = folder / f"{image_path.stem}_run{run}.txt"

    output_file.write_text(text, encoding="utf-8")


def export_csv(rows, output_file, fields):
    if not rows:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exists = (
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

        if not exists:
            writer.writeheader()

        writer.writerows(rows)

    print(f"Exported: {output_file}")


# ============================================================
# BENCHMARK EXECUTION
# ============================================================

def benchmark_model(model, runs, images, initialization_time):

    print("\n" + "=" * 55)
    print(f"MODEL: {model}")
    print("=" * 55)

    warmup_time = warmup_model(model, images)

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

                output = run_ai_ocr(
                    model,
                    image_path
                )

                processing_time = time.perf_counter() - start

                extracted_text = output["text"]

                cer, wer = calculate_accuracy(
                    extracted_text,
                    actual_text
                )

                save_extracted_text(
                    model,
                    image_path,
                    run,
                    extracted_text
                )

                result = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),

                    "model": model,

                    "image": image_path.name,

                    "language": language,

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
                    ),

                    "prompt_tokens": output["prompt_tokens"],

                    "generated_tokens": output["generated_tokens"],

                    "tokens_per_second": round(
                        output["tokens_per_second"], 4
                    ),

                    "initialization_seconds": round(
                        initialization_time, 4
                    ),

                    "warmup_seconds": round(
                        warmup_time, 4
                    ),

                    "status": "success"
                }

                results.append(result)

                print(
                    f"Processing time: {processing_time:.4f}s"
                )

                print(
                    f"Generated tokens: "
                    f"{output['generated_tokens']}"
                )

                print(
                    f"Tokens/second: "
                    f"{output['tokens_per_second']:.2f}"
                )

                if cer is not None:
                    print(f"CER: {cer:.2f}%")
                    print(f"WER: {wer:.2f}%")

            except Exception as error:

                print(f"ERROR: {error}")

                results.append({
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "model": model,
                    "image": image_path.name,
                    "language": language,
                    "run": run,
                    "processing_time_seconds": "",
                    "extracted_characters": "",
                    "cer_percent": "",
                    "wer_percent": "",
                    "prompt_tokens": "",
                    "generated_tokens": "",
                    "tokens_per_second": "",
                    "initialization_seconds": round(
                        initialization_time, 4
                    ),
                    "warmup_seconds": round(warmup_time, 4),
                    "status": f"error: {error}"
                })

    return results


# ============================================================
# SUMMARY GENERATION
# ============================================================

def generate_summary(results):

    grouped = defaultdict(list)

    for row in results:
        if row["status"] != "success":
            continue

        key = (
            row["model"],
            row["image"],
            row["language"]
        )

        grouped[key].append(row)

    summary = []

    for (model, image, language), rows in grouped.items():

        def average(field):
            values = [
                row[field]
                for row in rows
                if row[field] != ""
            ]

            return (
                round(sum(values) / len(values), 4)
                if values else ""
            )

        times = [
            row["processing_time_seconds"]
            for row in rows
        ]

        summary.append({
            "model": model,
            "image": image,
            "language": language,
            "total_runs": len(rows),

            "avg_processing_time_seconds": average(
                "processing_time_seconds"
            ),

            "min_processing_time_seconds": round(
                min(times), 4
            ),

            "max_processing_time_seconds": round(
                max(times), 4
            ),

            "avg_extracted_characters": average(
                "extracted_characters"
            ),

            "avg_cer_percent": average("cer_percent"),

            "avg_wer_percent": average("wer_percent"),

            "avg_tokens_per_second": average(
                "tokens_per_second"
            )
        })

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AI-based OCR Benchmark using Ollama"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model names to benchmark"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per image"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    try:

        images = load_dataset()

        if not images:
            raise RuntimeError("No dataset images found.")

        print(f"Dataset: {DATASET_DIR}")
        print(f"Total images: {len(images)}")
        print(f"Runs per image: {args.runs}")

        initialization_time, installed_models = check_ollama()

        all_results = []

        for requested_model in args.models:

            model = resolve_model_name(
                requested_model,
                installed_models
            )

            results = benchmark_model(
                model,
                args.runs,
                images,
                initialization_time
            )

            all_results.extend(results)

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

                print(
                    f"Summary exported: {SUMMARY_OUTPUT}"
                )

            print("\nAI OCR BENCHMARK COMPLETED!")

        else:
            print("\nNo benchmark results generated.")

    except Exception as error:

        print(f"\nBENCHMARK FAILED: {error}")

        raise SystemExit(1)
