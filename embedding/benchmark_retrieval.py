
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import requests


# ==========================================
# Configuration
# ==========================================

OLLAMA_URL = "http://localhost:11434/api/embed"

MODELS = [
    "bge-m3",
    "nomic-embed-text",
    "qwen3-embedding:0.6b",
]

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"

DOCUMENTS_FILE = DATA_DIR / "finance_documents.json"
QUERIES_FILE = DATA_DIR / "test_queries.json"

RESULT_FILE = RESULT_DIR / "retrieval_quality_results.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset():
    documents = load_json(DOCUMENTS_FILE)
    queries = load_json(QUERIES_FILE)

    if not documents or not queries:
        raise ValueError("Documents and queries cannot be empty.")

    document_ids = [doc["id"] for doc in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs found.")

    available_ids = set(document_ids)

    for query in queries:
        relevant_ids = query["relevant_docs"]

        if not relevant_ids:
            raise ValueError(
                f"Query {query['id']} has no relevant documents."
            )

        missing_ids = set(relevant_ids) - available_ids

        if missing_ids:
            raise ValueError(
                f"Query {query['id']} references unknown documents: "
                f"{missing_ids}"
            )

    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(queries)} test queries")

    return documents, queries


# ==========================================
# Generate Embeddings with Ollama
# ==========================================

def generate_embeddings(model, texts):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data.get("embeddings")

    if not embeddings:
        raise ValueError(
            f"No embeddings returned by model: {model}"
        )

    return np.array(embeddings, dtype=np.float32)


# ==========================================
# Cosine Similarity
# ==========================================

def calculate_similarity(query_embedding, document_embeddings):

    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Encountered a zero-length embedding.")

    similarities = np.dot(
        document_embeddings,
        query_embedding
    ) / (document_norms * query_norm)

    return similarities


# ==========================================
# Retrieval
# ==========================================

def retrieve_documents(
    query_embedding,
    document_embeddings,
    documents,
    k
):
    similarities = calculate_similarity(
        query_embedding,
        document_embeddings
    )

    # Sort documents from highest to lowest similarity
    ranked_indices = np.argsort(similarities)[::-1]

    top_indices = ranked_indices[:k]

    results = []

    for index in top_indices:
        results.append({
            "id": documents[index]["id"],
            "score": float(similarities[index])
        })

    return results


# ==========================================
# Evaluation Metrics
# ==========================================

def calculate_recall(retrieved_ids, relevant_ids):

    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    retrieved_relevant = set(retrieved_ids) & relevant_ids

    return len(retrieved_relevant) / len(relevant_ids)


def calculate_precision(retrieved_ids, relevant_ids, k):

    relevant_ids = set(relevant_ids)

    if k <= 0:
        return 0.0

    retrieved_relevant = sum(
        1 for doc_id in retrieved_ids
        if doc_id in relevant_ids
    )

    return retrieved_relevant / k


def calculate_mrr(retrieved_ids, relevant_ids):

    relevant_ids = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def calculate_dcg(relevance_scores):

    dcg = 0.0

    for rank, relevance in enumerate(
        relevance_scores,
        start=1
    ):
        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)

        dcg += gain / discount

    return dcg


def calculate_ndcg(retrieved_ids, relevance):

    # relevance: {document_id: relevance_score}
    relevance_scores = [
        relevance.get(doc_id, 0)
        for doc_id in retrieved_ids
    ]

    dcg = calculate_dcg(relevance_scores)

    # Ideal ranking: highest relevance first
    ideal_scores = sorted(
        relevance.values(),
        reverse=True
    )[:len(retrieved_ids)]

    idcg = calculate_dcg(ideal_scores)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ==========================================
# Evaluate One Model
# ==========================================

def evaluate_model(model, documents, queries, k):

    print(f"\n{'=' * 55}")
    print(f"Evaluating model: {model}")
    print(f"{'=' * 55}")

    document_texts = [
        doc["text"] for doc in documents
    ]

    # Embed all documents once
    print("Generating document embeddings...")

    start_time = time.perf_counter()

    document_embeddings = generate_embeddings(
        model,
        document_texts
    )

    document_embedding_time = (
        time.perf_counter() - start_time
    )

    if len(document_embeddings) != len(documents):
        raise ValueError("Document embedding count mismatch.")

    print(
        f"Document embedding time: "
        f"{document_embedding_time:.3f}s"
    )

    # Metric results
    recall_scores = []
    precision_scores = []
    mrr_scores = []
    ndcg_scores = []

    query_embedding_times = []

    print("\nRetrieving documents...")

    for index, query in enumerate(queries, start=1):

        query_text = query["text"]
        relevant_ids = query["relevant_docs"]

        # Generate query embedding
        start_time = time.perf_counter()

        query_embedding = generate_embeddings(
            model,
            [query_text]
        )[0]

        query_time = time.perf_counter() - start_time

        query_embedding_times.append(query_time)

        # Retrieve top K documents
        retrieved = retrieve_documents(
            query_embedding,
            document_embeddings,
            documents,
            k
        )

        retrieved_ids = [
            item["id"] for item in retrieved
        ]

        # Binary relevance for Recall, Precision and MRR
        recall = calculate_recall(
            retrieved_ids,
            relevant_ids
        )

        precision = calculate_precision(
            retrieved_ids,
            relevant_ids,
            k
        )

        mrr = calculate_mrr(
            retrieved_ids,
            relevant_ids
        )

        # Binary relevance grades for NDCG
        relevance = {
            doc_id: 1 for doc_id in relevant_ids
        }

        ndcg = calculate_ndcg(
            retrieved_ids,
            relevance
        )

        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

        print(
            f"Query {index}/{len(queries)} | "
            f"Recall@{k}: {recall:.3f} | "
            f"Precision@{k}: {precision:.3f} | "
            f"MRR: {mrr:.3f} | "
            f"NDCG@{k}: {ndcg:.3f}"
        )

    # Calculate average metrics
    results = {
        "model": model,
        "total_queries": len(queries),
        "k": k,

        "recall_at_k": float(np.mean(recall_scores)),
        "precision_at_k": float(np.mean(precision_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "ndcg_at_k": float(np.mean(ndcg_scores)),

        "document_embedding_time": document_embedding_time,

        "avg_query_embedding_time": float(
            np.mean(query_embedding_times)
        ),
    }

    return results


# ==========================================
# Save CSV Results
# ==========================================

def save_results(all_results):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(all_results[0].keys())

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {RESULT_FILE}")


# ==========================================
# Main
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Embedding Retrieval Quality Benchmark"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Models to benchmark"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than zero")

    documents, queries = load_dataset()

    all_results = []

    for model in args.models:

        try:
            result = evaluate_model(
                model,
                documents,
                queries,
                args.k
            )

            all_results.append(result)

        except requests.exceptions.RequestException as error:
            print(f"\nModel {model} API error: {error}")

        except Exception as error:
            print(f"\nModel {model} failed: {error}")

    if not all_results:
        print("\nNo models completed successfully.")
        return

    save_results(all_results)

    print("\n========== FINAL RESULTS ==========")

    for result in all_results:

        print(f"\nModel: {result['model']}")
        print(
            f"Recall@{args.k}: "
            f"{result['recall_at_k']:.4f}"
        )
        print(
            f"Precision@{args.k}: "
            f"{result['precision_at_k']:.4f}"
        )
        print(f"MRR: {result['mrr']:.4f}")
        print(
            f"NDCG@{args.k}: "
            f"{result['ndcg_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()
