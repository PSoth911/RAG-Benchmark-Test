"""
Combined Ollama embedding benchmark.

Measures:
1. Embedding speed on a plain-text dataset.
2. Retrieval quality on labeled documents and queries.
3. Embedding dimensions and optional vector export.

Expected files:
    data/finance_texts.txt
    data/finance_documents.json
    data/test_queries.json

Example:
    python benchmark_embedding.py
    python benchmark_embedding.py --models bge-m3 nomic-embed-text qwen3-embedding:0.6b
    python benchmark_embedding.py --runs 5 --k 10
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"

DEFAULT_MODELS = [
    "bge-m3",
    "nomic-embed-text",
    "qwen3-embedding:0.6b",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DEFAULT_SPEED_INPUT = DATA_DIR / "finance_texts.txt"
DEFAULT_DOCUMENTS = DATA_DIR / "finance_documents.json"
DEFAULT_QUERIES = DATA_DIR / "test_queries.json"

RESULTS_CSV = RESULTS_DIR / "embedding_benchmark_results.csv"


# ---------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------

def load_texts(file_path):
    """Load one text per line; ignore blank lines and # comments."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        texts = [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]

    if not texts:
        raise ValueError(f"No usable texts found in: {file_path}")

    return texts


def load_json(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_retrieval_dataset(documents_path, queries_path):
    documents = load_json(documents_path)
    queries = load_json(queries_path)

    if not isinstance(documents, list) or not documents:
        raise ValueError("Documents JSON must contain a non-empty list.")

    if not isinstance(queries, list) or not queries:
        raise ValueError("Queries JSON must contain a non-empty list.")

    for index, document in enumerate(documents):
        if "id" not in document or "text" not in document:
            raise ValueError(
                f"Document at position {index} must contain 'id' and 'text'."
            )
        if not isinstance(document["text"], str) or not document["text"].strip():
            raise ValueError(f"Document {document.get('id')} has empty text.")

    document_ids = [document["id"] for document in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs found.")

    available_ids = set(document_ids)

    for index, query in enumerate(queries):
        if "text" not in query or "relevant_docs" not in query:
            raise ValueError(
                f"Query at position {index} must contain "
                "'text' and 'relevant_docs'."
            )

        if not isinstance(query["text"], str) or not query["text"].strip():
            raise ValueError(f"Query at position {index} has empty text.")

        relevant_ids = query["relevant_docs"]

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"Query {query.get('id', index)} must have relevant_docs."
            )

        missing_ids = set(relevant_ids) - available_ids
        if missing_ids:
            raise ValueError(
                f"Query {query.get('id', index)} references unknown "
                f"document IDs: {sorted(missing_ids, key=str)}"
            )

    print(f"Loaded {len(documents)} retrieval documents.")
    print(f"Loaded {len(queries)} test queries.")

    return documents, queries


# ---------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------

def request_embeddings(model, texts, timeout=600):
    """Call Ollama and return embeddings, response metadata, and wall time."""
    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    response.raise_for_status()

    elapsed = time.perf_counter() - start
    data = response.json()

    embeddings = data.get("embeddings")
    if not embeddings:
        raise ValueError(f"Ollama returned no embeddings for model '{model}'.")

    if len(embeddings) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings but received {len(embeddings)}."
        )

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Ollama returned embeddings with an unexpected shape.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding output contains NaN or infinite values.")

    metadata = {
        "ollama_total_seconds": data.get("total_duration", 0) / 1e9,
        "ollama_load_seconds": data.get("load_duration", 0) / 1e9,
    }

    return vectors, elapsed, metadata


def warm_up_model(model):
    """Warm up the model and measure the first request separately."""
    print(f"\nWarming up model: {model}")
    _, elapsed, metadata = request_embeddings(
        model,
        ["This is a short warm-up text for the embedding benchmark."],
    )
    print(f"Warm-up request: {elapsed:.4f} seconds")
    return elapsed, metadata


# ---------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------

def benchmark_speed(model, texts, runs):
    """Measure repeated batch-embedding requests on the speed dataset."""
    latencies = []
    throughputs = []
    ollama_total_times = []
    ollama_load_times = []
    last_embeddings = None

    print(f"\n--- Speed benchmark: {model} ---")
    print(f"Texts per run: {len(texts)}")

    for run_number in range(1, runs + 1):
        print(f"Run {run_number}/{runs} ...")

        vectors, elapsed, metadata = request_embeddings(model, texts)

        throughput = len(texts) / elapsed if elapsed > 0 else 0.0

        latencies.append(elapsed)
        throughputs.append(throughput)
        ollama_total_times.append(metadata["ollama_total_seconds"])
        ollama_load_times.append(metadata["ollama_load_seconds"])
        last_embeddings = vectors

        print(
            f"  Latency: {elapsed:.4f}s | "
            f"Throughput: {throughput:.2f} texts/sec | "
            f"Dimensions: {vectors.shape[1]}"
        )

    return {
        "speed_text_count": len(texts),
        "embedding_dimensions": int(last_embeddings.shape[1]),
        "speed_runs_completed": len(latencies),
        "avg_speed_latency_seconds": float(np.mean(latencies)),
        "min_speed_latency_seconds": float(np.min(latencies)),
        "max_speed_latency_seconds": float(np.max(latencies)),
        "avg_texts_per_second": float(np.mean(throughputs)),
        "avg_ollama_total_seconds": float(np.mean(ollama_total_times)),
        "avg_ollama_load_seconds": float(np.mean(ollama_load_times)),
    }, last_embeddings


# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

def cosine_similarities(query_vector, document_vectors):
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Encountered a zero-length embedding.")

    return np.dot(document_vectors, query_vector) / (
        document_norms * query_norm
    )


def retrieve_documents(query_vector, document_vectors, documents, k):
    similarities = cosine_similarities(query_vector, document_vectors)
    ranked_indices = np.argsort(similarities)[::-1][:k]

    return [
        {
            "id": documents[index]["id"],
            "score": float(similarities[index]),
        }
        for index in ranked_indices
    ]


def calculate_recall(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids) & relevant) / len(relevant)


def calculate_precision(retrieved_ids, relevant_ids, k):
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for document_id in retrieved_ids if document_id in relevant)
    return hits / min(k, len(retrieved_ids)) if retrieved_ids else 0.0


def calculate_mrr(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevance_scores):
    score = 0.0
    for rank, relevance in enumerate(relevance_scores, start=1):
        gain = (2 ** relevance) - 1
        score += gain / np.log2(rank + 1)
    return score


def calculate_ndcg(retrieved_ids, relevant_ids):
    """Binary-relevance NDCG: relevant=1, non-relevant=0."""
    relevant = set(relevant_ids)
    actual = [1 if document_id in relevant else 0 for document_id in retrieved_ids]
    ideal = sorted(actual, reverse=True)

    dcg = calculate_dcg(actual)
    idcg = calculate_dcg(ideal)

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------
# Retrieval benchmark
# ---------------------------------------------------------------------

def benchmark_retrieval(model, documents, queries, k):
    print(f"\n--- Retrieval benchmark: {model} ---")

    document_texts = [document["text"] for document in documents]

    print("Generating document embeddings...")
    document_vectors, document_elapsed, document_metadata = request_embeddings(
        model, document_texts
    )

    if len(document_vectors) != len(documents):
        raise ValueError("Document embedding count does not match document count.")

    print(
        f"Embedded {len(documents)} documents in {document_elapsed:.4f}s "
        f"({document_vectors.shape[1]} dimensions)."
    )

    effective_k = min(k, len(documents))

    recall_scores = []
    precision_scores = []
    mrr_scores = []
    ndcg_scores = []
    query_times = []

    for index, query in enumerate(queries, start=1):
        query_vector, query_elapsed, _ = request_embeddings(
            model, [query["text"]]
        )
        query_times.append(query_elapsed)

        retrieved = retrieve_documents(
            query_vector[0], document_vectors, documents, effective_k
        )
        retrieved_ids = [item["id"] for item in retrieved]
        relevant_ids = query["relevant_docs"]

        recall = calculate_recall(retrieved_ids, relevant_ids)
        precision = calculate_precision(retrieved_ids, relevant_ids, effective_k)
        mrr = calculate_mrr(retrieved_ids, relevant_ids)
        ndcg = calculate_ndcg(retrieved_ids, relevant_ids)

        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

        print(
            f"Query {index}/{len(queries)} | "
            f"Recall@{effective_k}: {recall:.3f} | "
            f"Precision@{effective_k}: {precision:.3f} | "
            f"MRR: {mrr:.3f} | NDCG@{effective_k}: {ndcg:.3f}"
        )

    return {
        "total_documents": len(documents),
        "total_queries": len(queries),
        "k": effective_k,
        "document_embedding_time_seconds": document_elapsed,
        "document_ollama_total_seconds": document_metadata["ollama_total_seconds"],
        "avg_query_embedding_time_seconds": float(np.mean(query_times)),
        "recall_at_k": float(np.mean(recall_scores)),
        "precision_at_k": float(np.mean(precision_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "ndcg_at_k": float(np.mean(ndcg_scores)),
    }, document_vectors


# ---------------------------------------------------------------------
# Embedding export and CSV results
# ---------------------------------------------------------------------

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def export_embeddings(model, documents, vectors):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / f"embeddings_{safe_filename(model)}.json"

    payload = {
        "model": model,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": "finance_documents.json",
        "text_count": len(documents),
        "embedding_dimensions": int(vectors.shape[1]),
        "items": [
            {
                "id": document["id"],
                "text": document["text"],
                "embedding": vector.tolist(),
            }
            for document, vector in zip(documents, vectors)
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Document embeddings exported to: {output_path}")


def save_results(results, output_path):
    if not results:
        print("No successful model results to save.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    # Write a fresh comparison file for this benchmark execution.
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCombined benchmark results saved to: {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined Ollama embedding speed and retrieval benchmark."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama embedding model names.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repeated speed benchmark runs (default: 3).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--speed-input",
        default=str(DEFAULT_SPEED_INPUT),
        help="Text file used for the speed benchmark.",
    )
    parser.add_argument(
        "--documents",
        default=str(DEFAULT_DOCUMENTS),
        help="JSON file containing retrieval documents.",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES),
        help="JSON file containing labeled test queries.",
    )
    parser.add_argument(
        "--no-export-embeddings",
        action="store_true",
        help="Do not export document embedding vectors as JSON.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1.")
    if args.k < 1:
        parser.error("--k must be at least 1.")
    if not args.models:
        parser.error("At least one model must be supplied.")

    try:
        speed_texts = load_texts(args.speed_input)
        documents, queries = load_retrieval_dataset(
            args.documents, args.queries
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Dataset error: {error}")
        sys.exit(1)

    print("\n========== EMBEDDING BENCHMARK ==========")
    print(f"Models: {', '.join(args.models)}")
    print(f"Speed texts: {len(speed_texts)}")
    print(f"Retrieval documents: {len(documents)}")
    print(f"Test queries: {len(queries)}")
    print(f"Speed runs per model: {args.runs}")
    print(f"Requested retrieval K: {args.k}")

    all_results = []

    for model in args.models:
        print(f"\n{'=' * 65}")
        print(f"MODEL: {model}")
        print(f"{'=' * 65}")

        try:
            warmup_seconds, warmup_metadata = warm_up_model(model)

            speed_results, _ = benchmark_speed(
                model, speed_texts, args.runs
            )

            retrieval_results, document_vectors = benchmark_retrieval(
                model, documents, queries, args.k
            )

            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "warmup_request_seconds": round(warmup_seconds, 4),
                "warmup_ollama_load_seconds": round(
                    warmup_metadata["ollama_load_seconds"], 4
                ),
                **speed_results,
                **retrieval_results,
            }

            all_results.append(result)

            if not args.no_export_embeddings:
                export_embeddings(model, documents, document_vectors)

            print(f"\nCompleted model: {model}")

        except requests.exceptions.RequestException as error:
            print(f"\nAPI error for {model}: {error}")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            print(f"\nBenchmark failed for {model}: {error}")

    if not all_results:
        print("\nNo models completed successfully. Check Ollama and model names.")
        sys.exit(1)

    save_results(all_results, RESULTS_CSV)

    print("\n========== FINAL COMPARISON ==========")
    for result in all_results:
        k_value = result["k"]
        print(f"\nModel: {result['model']}")
        print(
            f"Dimensions: {result['embedding_dimensions']} | "
            f"Speed: {result['avg_texts_per_second']:.2f} texts/sec"
        )
        print(
            f"Avg speed latency: {result['avg_speed_latency_seconds']:.4f}s | "
            f"Warm-up: {result['warmup_request_seconds']:.4f}s"
        )
        print(
            f"Recall@{k_value}: {result['recall_at_k']:.4f} | "
            f"Precision@{k_value}: {result['precision_at_k']:.4f}"
        )
        print(
            f"MRR: {result['mrr']:.4f} | "
            f"NDCG@{k_value}: {result['ndcg_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()
