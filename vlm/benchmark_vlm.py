
import argparse
import base64
import csv
import json
import time
from pathlib import Path

import requests


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
QUESTIONS_FILE = DATASET_DIR / "questions.json"

OUTPUT_DIR = BASE_DIR / "results"
RESULTS_FILE = OUTPUT_DIR / "vlm_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "vlm_summary.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODELS = [
    "qwen2.5vl:7b",
    "qwen3-vl:8b"
]

TIMEOUT = 600

CSV_FIELDS = [
    "model",
    "image",
    "question",
    "ground_truth",
    "predicted_answer",
    "processing_time_seconds",
    "prompt_tokens",
    "generated_tokens",
    "tokens_per_second",
    "status",
    "error"
]


# ==========================================
# LOAD DATASET
# ==========================================

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON list.")

    for index, item in enumerate(questions):
        for field in ("image", "question", "ground_truth"):
            if field not in item:
                raise ValueError(
                    f"Question {index + 1} is missing '{field}'."
                )

    return questions


# ==========================================
# IMAGE PROCESSING
# ==========================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# ==========================================
# CALL VLM
# ==========================================

def ask_vlm(model, image_path, question):

    image_base64 = encode_image(image_path)

    prompt = f"""
You are analyzing an image for a finance-domain RAG system.

Answer the following question using only information
that is visible in the provided image.

Question:
{question}

Instructions:
- Give a direct and concise answer.
- Preserve financial values and units accurately.
- Do not invent information.
- If the answer cannot be determined from the image,
  explicitly say that the information is not available.
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    elapsed = time.perf_counter() - start_time

    result = response.json()

    answer = result.get("message", {}).get("content", "")

    prompt_tokens = result.get("prompt_eval_count", 0)
    generated_tokens = result.get("eval_count", 0)
    generation_time_ns = result.get("eval_duration", 0)

    tokens_per_second = (
        generated_tokens / (generation_time_ns / 1_000_000_000)
        if generation_time_ns > 0
        else 0
    )

    return {
        "answer": answer.strip(),
        "processing_time": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(tokens_per_second, 2)
    }


# ==========================================
# SAVE RESULTS IMMEDIATELY
# ==========================================

def save_result(result):

    file_exists = RESULTS_FILE.exists()

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


# ==========================================
# GENERATE SUMMARY
# ==========================================

def generate_summary():

    if not RESULTS_FILE.exists():
        return

    with open(
        RESULTS_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        rows = list(csv.DictReader(file))

    summary = {}

    for row in rows:

        if row["status"] != "success":
            continue

        model = row["model"]

        if model not in summary:
            summary[model] = {
                "total_questions": 0,
                "total_time": 0,
                "total_prompt_tokens": 0,
                "total_generated_tokens": 0
            }

        data = summary[model]

        data["total_questions"] += 1
        data["total_time"] += float(
            row["processing_time_seconds"]
        )
        data["total_prompt_tokens"] += int(
            row["prompt_tokens"]
        )
        data["total_generated_tokens"] += int(
            row["generated_tokens"]
        )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fields = [
            "model",
            "successful_questions",
            "avg_processing_time_seconds",
            "total_prompt_tokens",
            "total_generated_tokens"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for model, data in summary.items():

            count = data["total_questions"]

            writer.writerow({
                "model": model,
                "successful_questions": count,
                "avg_processing_time_seconds": round(
                    data["total_time"] / count, 3
                ),
                "total_prompt_tokens": data["total_prompt_tokens"],
                "total_generated_tokens": data["total_generated_tokens"]
            })


# ==========================================
# MAIN BENCHMARK
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="VLM image understanding benchmark"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()

    print("=" * 60)
    print("VLM BENCHMARK")
    print("=" * 60)

    print(f"Questions: {len(questions)}")
    print(f"Models: {args.models}")

    for model in args.models:

        print(f"\nStarting model: {model}")

        for index, item in enumerate(questions, start=1):

            image_path = DATASET_DIR / item["image"]

            print(
                f"\n[{index}/{len(questions)}] "
                f"Image: {item['image']}"
            )

            print(f"Question: {item['question']}")

            result = {
                "model": model,
                "image": item["image"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "predicted_answer": "",
                "processing_time_seconds": "",
                "prompt_tokens": "",
                "generated_tokens": "",
                "tokens_per_second": "",
                "status": "failed",
                "error": ""
            }

            try:

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                output = ask_vlm(
                    model,
                    image_path,
                    item["question"]
                )

                result.update({
                    "predicted_answer": output["answer"],
                    "processing_time_seconds": output[
                        "processing_time"
                    ],
                    "prompt_tokens": output["prompt_tokens"],
                    "generated_tokens": output["generated_tokens"],
                    "tokens_per_second": output["tokens_per_second"],
                    "status": "success"
                })

                print(f"Answer: {output['answer']}")
                print(
                    f"Time: {output['processing_time']} seconds"
                )

            except Exception as error:

                result["error"] = str(error)

                print(f"ERROR: {error}")

            # Save each question immediately.
            save_result(result)

    generate_summary()

    print("\n" + "=" * 60)
    print("VLM BENCHMARK COMPLETED")
    print(f"Results: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
