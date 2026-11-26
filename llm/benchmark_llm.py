
import requests
import csv
import json
import time
import argparse
import statistics

from datetime import datetime
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen3:8b",
    "llama3.1:8b",
    "gemma3:4b",
    "qwen3:4b"
]

BASE_DIR = Path(__file__).parent

DATASET_FILE = BASE_DIR / "dataset" / "finance_questions.json"

OUTPUT_DIR = BASE_DIR / "results"


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list.")

    required_fields = {
        "id", "context", "question", "reference"
    }

    for index, item in enumerate(dataset):

        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Dataset item {index} missing fields: {missing}"
            )

    print(f"Dataset loaded: {len(dataset)} questions")

    return dataset


# ==================================================
# BUILD PROMPT
# ==================================================

def build_prompt(context, question):

    return f"""
You are a financial information assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information,
clearly state that the information is insufficient.

Keep your answer simple, accurate, and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ==================================================
# GENERATE ANSWER USING OLLAMA
# ==================================================

def generate_answer(model_name, prompt):

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    end_time = time.perf_counter()

    data = response.json()

    total_time = end_time - start_time

    generation_time = data.get(
        "eval_duration", 0
    ) / 1e9

    prompt_time = data.get(
        "prompt_eval_duration", 0
    ) / 1e9

    load_time = data.get(
        "load_duration", 0
    ) / 1e9

    generated_tokens = data.get(
        "eval_count", 0
    )

    prompt_tokens = data.get(
        "prompt_eval_count", 0
    )

    tokens_per_second = (
        generated_tokens / generation_time
        if generation_time > 0 else 0
    )

    return {
        "response": data.get("response", ""),
        "latency": total_time,
        "load_time": load_time,
        "prompt_time": prompt_time,
        "generation_time": generation_time,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second
    }


# ==================================================
# MODEL WARMUP
# ==================================================

def warmup_model(model_name):

    print(f"\nWarming up {model_name}...")

    start_time = time.perf_counter()

    generate_answer(
        model_name,
        "Reply with the word READY."
    )

    warmup_time = time.perf_counter() - start_time

    print(
        f"Warmup completed: {warmup_time:.3f} seconds"
    )

    return warmup_time


# ==================================================
# BENCHMARK MODEL
# ==================================================

def benchmark_model(model_name, dataset, runs):

    results = []

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    try:

        warmup_time = warmup_model(model_name)

    except requests.exceptions.RequestException as error:

        print(f"Warmup failed for {model_name}: {error}")
        return results

    for test in dataset:

        prompt = build_prompt(
            test["context"],
            test["question"]
        )

        for run in range(1, runs + 1):

            print(
                f"\nQuestion: {test['id']} "
                f"| Run {run}/{runs}"
            )

            try:

                metrics = generate_answer(
                    model_name,
                    prompt
                )

                result = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),

                    "model": model_name,
                    "warmup_time_seconds": round(
                        warmup_time, 3
                    ),

                    "question_id": test["id"],
                    "run": run,

                    "question": test["question"],
                    "reference_answer": test["reference"],
                    "response": metrics["response"],

                    "latency_seconds": round(
                        metrics["latency"], 3
                    ),

                    "load_time_seconds": round(
                        metrics["load_time"], 3
                    ),

                    "prompt_eval_seconds": round(
                        metrics["prompt_time"], 3
                    ),

                    "generation_seconds": round(
                        metrics["generation_time"], 3
                    ),

                    "prompt_tokens": metrics["prompt_tokens"],
                    "generated_tokens": metrics["generated_tokens"],

                    "tokens_per_second": round(
                        metrics["tokens_per_second"], 2
                    )
                }

                results.append(result)

                print("\nGenerated Answer:")
                print("-" * 50)
                print(metrics["response"])
                print("-" * 50)

                print(
                    f"Latency: {metrics['latency']:.3f}s"
                )

                print(
                    f"Generated tokens: {metrics['generated_tokens']}"
                )

                print(
                    f"Tokens/sec: {metrics['tokens_per_second']:.2f}"
                )

            except requests.exceptions.RequestException as error:

                print(
                    f"Error: {model_name}, "
                    f"{test['id']}, Run {run}: {error}"
                )

    return results


# ==================================================
# EXPORT DETAILED CSV
# ==================================================

def export_detailed_results(results):

    if not results:
        print("No detailed results to export.")
        return

    output_file = OUTPUT_DIR / "llm_detailed_results.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed CSV exported: {output_file}")


# ==================================================
# EXPORT SUMMARY CSV
# ==================================================

def export_summary(results):

    if not results:
        print("No summary results to export.")
        return

    summary = []

    for model in MODELS:

        model_results = [
            result for result in results
            if result["model"] == model
        ]

        if not model_results:
            continue

        summary.append({

            "model": model,

            "total_tests": len(model_results),

            "avg_latency_seconds": round(
                statistics.mean(
                    r["latency_seconds"]
                    for r in model_results
                ), 3
            ),

            "avg_load_time_seconds": round(
                statistics.mean(
                    r["load_time_seconds"]
                    for r in model_results
                ), 3
            ),

            "avg_prompt_eval_seconds": round(
                statistics.mean(
                    r["prompt_eval_seconds"]
                    for r in model_results
                ), 3
            ),

            "avg_generation_seconds": round(
                statistics.mean(
                    r["generation_seconds"]
                    for r in model_results
                ), 3
            ),

            "avg_prompt_tokens": round(
                statistics.mean(
                    r["prompt_tokens"]
                    for r in model_results
                ), 2
            ),

            "avg_generated_tokens": round(
                statistics.mean(
                    r["generated_tokens"]
                    for r in model_results
                ), 2
            ),

            "avg_tokens_per_second": round(
                statistics.mean(
                    r["tokens_per_second"]
                    for r in model_results
                ), 2
            )
        })

    if not summary:
        return

    output_file = OUTPUT_DIR / "llm_summary.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys())
        )

        writer.writeheader()
        writer.writerows(summary)

    print(f"Summary CSV exported: {output_file}")


# ==================================================
# EXPORT GENERATED ANSWERS
# ==================================================

def export_answers(results):

    if not results:
        print("No answers to export.")
        return

    output_file = OUTPUT_DIR / "generated_answers.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 75 + "\n")
        file.write("LLM BENCHMARK - GENERATED ANSWERS\n")
        file.write("=" * 75 + "\n")

        file.write(
            f"Generated at: {datetime.now().isoformat()}\n"
        )

        file.write(
            f"Total responses: {len(results)}\n\n"
        )

        for result in results:

            file.write("\n" + "=" * 75 + "\n")

            file.write(
                f"MODEL: {result['model']}\n"
            )

            file.write(
                f"QUESTION ID: {result['question_id']}\n"
            )

            file.write(
                f"RUN: {result['run']}\n"
            )

            file.write("=" * 75 + "\n\n")

            file.write("QUESTION:\n")
            file.write(
                result["question"] + "\n\n"
            )

            file.write("REFERENCE ANSWER:\n")
            file.write(
                result["reference_answer"] + "\n\n"
            )

            file.write("GENERATED ANSWER:\n")
            file.write(
                result["response"] + "\n\n"
            )

            file.write("PERFORMANCE:\n")

            file.write(
                f"Latency: {result['latency_seconds']} seconds\n"
            )

            file.write(
                f"Model load time: {result['load_time_seconds']} seconds\n"
            )

            file.write(
                f"Prompt evaluation: {result['prompt_eval_seconds']} seconds\n"
            )

            file.write(
                f"Generation time: {result['generation_seconds']} seconds\n"
            )

            file.write(
                f"Prompt tokens: {result['prompt_tokens']}\n"
            )

            file.write(
                f"Generated tokens: {result['generated_tokens']}\n"
            )

            file.write(
                f"Tokens/sec: {result['tokens_per_second']}\n"
            )

            file.write("\n")

    print(f"Answer report exported: {output_file}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ollama LLM Benchmark"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of runs per question"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset
    TEST_CASES = load_dataset()

    print("\n" + "=" * 65)
    print("STARTING LLM BENCHMARK")
    print("=" * 65)

    print(f"Models: {MODELS}")
    print(f"Questions: {len(TEST_CASES)}")
    print(f"Runs per question: {args.runs}")

    all_results = []

    # Benchmark every model
    for model in MODELS:

        results = benchmark_model(
            model,
            TEST_CASES,
            args.runs
        )

        all_results.extend(results)

    # Export results
    export_detailed_results(all_results)

    export_summary(all_results)

    export_answers(all_results)

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print(f"Successful responses: {len(all_results)}")

    print(f"Expected responses: {len(MODELS) * len(TEST_CASES) * args.runs}")

    print(f"Results folder: {OUTPUT_DIR}")
