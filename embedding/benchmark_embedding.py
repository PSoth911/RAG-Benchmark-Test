import requests
import csv
import json
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Ollama embedding API endpoint
OLLAMA_URL = "http://localhost:11434/api/embed"

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_INPUT = BASE_DIR / "finance_texts.txt"


def load_texts(path):
    """Read one text per line. Skip blank lines and # comments."""
    path = Path(path)

    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        texts = [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]

    if not texts:
        sys.exit(f"No texts found in: {path}")

    return texts


def benchmark_embedding(model_name, runs, texts, input_name):

    results = []
    last_embeddings = None

    # Warm up the model before measuring
    print(f"\nLoading model: {model_name}")

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model_name, "input": texts},
            timeout=600
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as error:
        print(f"Model initialization failed: {error}")
        return results, None

    # Run benchmark
    for run in range(1, runs + 1):

        print(f"\nRunning {model_name} - Test {run}/{runs}")

        payload = {"model": model_name, "input": texts}

        try:
            start_time = time.perf_counter()

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=600
            )
            response.raise_for_status()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            data = response.json()
            embeddings = data["embeddings"]

            if not embeddings:
                print("No embeddings returned.")
                continue

            text_count = len(embeddings)
            dimensions = len(embeddings[0])

            total_duration = data.get("total_duration", 0) / 1e9
            load_duration = data.get("load_duration", 0) / 1e9

            throughput = (
                text_count / elapsed_time if elapsed_time > 0 else 0
            )

            results.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "input_file": input_name,
                "run": run,
                "text_count": text_count,
                "embedding_dimensions": dimensions,
                "latency_seconds": round(elapsed_time, 4),
                "ollama_total_seconds": round(total_duration, 4),
                "load_time_seconds": round(load_duration, 4),
                "texts_per_second": round(throughput, 2)
            })

            # Keep the embeddings from the latest successful run
            last_embeddings = embeddings

            print("Completed!")
            print(f"Dimensions: {dimensions}")
            print(f"Latency: {elapsed_time:.4f} seconds")
            print(f"Throughput: {throughput:.2f} texts/sec")

        except (
            requests.exceptions.RequestException,
            KeyError,
            IndexError,
            ValueError
        ) as error:
            print(f"Benchmark error: {error}")

    return results, last_embeddings


def preview_embeddings(texts, embeddings, count=3, values=8):
    """Print a short look at what the embeddings contain."""

    print("\n--- Embedding preview ---")

    for text, vector in list(zip(texts, embeddings))[:count]:
        norm = sum(v * v for v in vector) ** 0.5
        shown = ", ".join(f"{v:.4f}" for v in vector[:values])

        print(f"\nText: {text}")
        print(f"Dimensions: {len(vector)}")
        print(f"First {values} values: [{shown}, ...]")
        print(f"Vector length (L2 norm): {norm:.4f}")


def export_embeddings(model_name, input_name, texts, embeddings):
    """Save full embeddings to results/embeddings_<model>.json"""

    if not embeddings:
        print("No embeddings to export.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    # Make the model name safe for filenames (e.g. "model:latest")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    output_file = RESULTS_DIR / f"embeddings_{safe_name}.json"

    payload = {
        "model": model_name,
        "input_file": input_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "text_count": len(embeddings),
        "embedding_dimensions": len(embeddings[0]),
        "items": [
            {"id": i, "text": text, "embedding": vector}
            for i, (text, vector) in enumerate(zip(texts, embeddings))
        ]
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Embeddings exported to: {output_file}")


def export_csv(results):

    if not results:
        print("No results to export.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    output_file = RESULTS_DIR / "embedding_results.csv"
    file_exists = output_file.exists()

    with open(output_file, "a", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=results[0].keys())

        if not file_exists or output_file.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(results)

    print(f"Results exported to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ollama Embedding Model Benchmark"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama embedding model name"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs"
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Text file with one test text per line"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    texts = load_texts(args.input)
    input_name = Path(args.input).name

    print(f"Loaded {len(texts)} texts from {input_name}")

    results, embeddings = benchmark_embedding(
        args.model,
        args.runs,
        texts,
        input_name
    )

    if embeddings:
        preview_embeddings(texts, embeddings)
        export_embeddings(args.model, input_name, texts, embeddings)

    export_csv(results)