import requests
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Fixed prompt for fair comparison
PROMPT = """
Explain what Retrieval-Augmented Generation (RAG) is.
Describe its main components and how they work together.
Keep your explanation simple and clear.
"""


def benchmark_model(model_name, runs):

    results = []

    for run in range(1, runs + 1):

        print(f"\nRunning {model_name} - Test {run}/{runs}")

        payload = {
            "model": model_name,
            "prompt": PROMPT,
            "stream": False
        }

        try:
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

            load_time = data["load_duration"] / 1e9
            prompt_tokens = data["prompt_eval_count"]
            generated_tokens = data["eval_count"]
            generation_time = data["eval_duration"] / 1e9

            tokens_per_second = (
                generated_tokens / generation_time
                if generation_time > 0 else 0
            )

            result = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "model": model_name,
                "run": run,
                "latency_seconds": round(total_time, 3),
                "load_time_seconds": round(load_time, 3),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "tokens_per_second": round(
                    tokens_per_second, 2
                )
            }

            results.append(result)

            print("Completed!")
            print(f"Latency: {total_time:.3f} seconds")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Tokens/sec: {tokens_per_second:.2f}")

        except (requests.exceptions.RequestException, KeyError) as error:
            print(f"Error running model: {error}")

    return results


def export_csv(results):

    if not results:
        print("No results to export.")
        return

    output_file = Path(__file__).parent / "results.csv"

    file_exists = output_file.exists()

    with open(output_file, "a", newline="", encoding="utf-8") as file:

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
        description="Ollama Model Benchmark"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs"
    )

    args = parser.parse_args()

    results = benchmark_model(args.model, args.runs)

    export_csv(results)
