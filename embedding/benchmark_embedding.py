import requests
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

# Ollama embedding API endpoint
OLLAMA_URL = "http://localhost:11434/api/embed"

# Fixed dataset for fair comparison
TEST_TEXTS = [
    "Retrieval-Augmented Generation combines information retrieval with language generation.",
    "Machine learning allows computers to learn patterns from data.",
    "A vector database stores numerical representations of text.",
    "Python is a popular programming language for artificial intelligence.",
    "Document processing extracts useful information from files.",
    "Natural language processing helps computers understand human language.",
    "Embeddings represent text as numerical vectors.",
    "Semantic search finds information based on meaning.",
    "A chatbot can use retrieved documents to answer questions.",
    "Ollama allows users to run language models locally."
]


def benchmark_embedding(model_name, runs):

    results = []

    # Warm up the model before measuring
    print(f"\nLoading model: {model_name}")

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "input": TEST_TEXTS
            },
            timeout=600
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as error:
        print(f"Model initialization failed: {error}")
        return results

    # Run benchmark
    for run in range(1, runs + 1):

        print(f"\nRunning {model_name} - Test {run}/{runs}")

        payload = {
            "model": model_name,
            "input": TEST_TEXTS
        }

        try:
            # Measure execution time
            start_time = time.perf_counter()

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=600
            )

            response.raise_for_status()

            end_time = time.perf_counter()

            data = response.json()

            # Calculate elapsed time
            elapsed_time = end_time - start_time

            # Extract embeddings
            embeddings = data["embeddings"]

            if not embeddings:
                print("No embeddings returned.")
                continue

            text_count = len(embeddings)
            dimensions = len(embeddings[0])

            # Extract Ollama metrics
            total_duration = data.get(
                "total_duration", 0
            ) / 1e9

            load_duration = data.get(
                "load_duration", 0
            ) / 1e9

            # Calculate throughput
            throughput = (
                text_count / elapsed_time
                if elapsed_time > 0 else 0
            )

            # Store results
            result = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "model": model_name,
                "run": run,
                "text_count": text_count,
                "embedding_dimensions": dimensions,
                "latency_seconds": round(elapsed_time, 4),
                "ollama_total_seconds": round(total_duration, 4),
                "load_time_seconds": round(load_duration, 4),
                "texts_per_second": round(throughput, 2)
            }

            results.append(result)

            # Display results
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

    return results


def export_csv(results):

    if not results:
        print("No results to export.")
        return

    # Create results directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "embedding_results.csv"

    file_exists = output_file.exists()

    # Append results to CSV
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

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    results = benchmark_embedding(
        args.model,
        args.runs
    )

    export_csv(results)
