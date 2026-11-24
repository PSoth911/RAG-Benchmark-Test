# Embedding Model Benchmarking Documentation

## 1. Objective

The purpose of this benchmark is to evaluate and compare embedding models for the Retrieval-Augmented Generation (RAG) system.

Embedding models convert text into numerical vectors that represent semantic meaning. These vectors are used to retrieve relevant documents based on similarity.

The benchmark focuses on:

- Embedding generation latency.
- Embedding dimensions.
- Ollama processing time.
- Throughput (texts per second).
- Model loading time.

## 2. Models to Evaluate

| Model | Description |
|---|---|
| BGE-M3 | Multilingual embedding model |
| Nomic Embed Text | Text embedding model designed for semantic search |
| Qwen3-Embedding:0.6b | Embedding model from the Qwen family |

## 3. Prerequisites

### Software
- Python 3.10+
- Ollama
- VS Code or another Python IDE

### Python dependencies

```bash
pip install requests
```

The following modules are included in Python's standard library:

- csv
- time
- argparse
- datetime
- pathlib

## 4. Ollama Preparation

Download the embedding models:

```bash
ollama pull bge-m3
ollama pull nomic-embed-text
ollama pull qwen3-embedding:0.6b
```

Check available models:

```bash
ollama list
```

The benchmark uses the Ollama embedding API:

```text
http://localhost:11434/api/embed
```

## 5. Test Dataset

The benchmark script uses a predefined collection of test sentences.

For example:

```python
TEST_TEXTS = [
    "What is artificial intelligence?",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks.",
    "What is Retrieval-Augmented Generation?",
    "RAG combines retrieval and text generation.",
]
```

The actual benchmark should use the complete fixed test collection defined in the Python script.

Every embedding model receives the same text inputs.

For a more representative RAG evaluation, consider including:

- Short sentences.
- Long paragraphs.
- Technical descriptions.
- Questions.
- English documents.
- Khmer documents, if multilingual retrieval is required.

## 6. Benchmark Methodology

The script performs the following steps:

1. Load the selected embedding model.
2. Send a warm-up request.
3. Send the predefined test texts to the embedding API.
4. Measure the processing time.
5. Record the embedding dimensions.
6. Calculate throughput.
7. Export the results to CSV.

Example execution:

```bash
python benchmark_embedding.py --model bge-m3 --runs 3
```

Run the remaining models:

```bash
python benchmark_embedding.py --model nomic-embed-text --runs 3
python benchmark_embedding.py --model qwen3-embedding:0.6b --runs 3
```

## 7. Evaluation Metrics

| Metric | Description |
|---|---|
| Latency (seconds) | Time required to generate embeddings |
| Embedding dimensions | Number of numerical values in each vector |
| Ollama total time | Processing duration reported by Ollama |
| Model loading time | Model loading duration reported by Ollama |
| Text count | Number of texts processed |
| Texts per second | Number of texts embedded per second |

**Note:** Embedding dimensions can differ between models. This matters when selecting a vector database and configuring its vector column.

## 8. Output

The benchmark script saves results to:

```text
results/embedding_results.csv
```

Example CSV structure:

| Model | Run | Text Count | Dimensions | Latency |
|---|---:|---:|---:|---:|
| bge-m3 | 1 | 5 | Example | 0.85 |
| bge-m3 | 2 | 5 | Example | 0.79 |
| nomic-embed-text | 1 | 5 | Example | 0.63 |

*The values are illustrative. Actual dimensions and performance must be obtained from your benchmark.*

## 9. Execution Checklist

- [ ] Install Ollama.
- [ ] Download all embedding models.
- [ ] Install Python dependencies.
- [ ] Prepare a fixed test dataset.
- [ ] Ensure Ollama is running.
- [ ] Execute the benchmark for every model.
- [ ] Collect CSV results.
- [ ] Compare latency and throughput.
- [ ] Record embedding dimensions.

## 10. Notes

For a fair comparison:

1. Use exactly the same test texts for each model.
2. Use the same number of runs.
3. Perform warm-up requests before measured runs.
4. Keep hardware and software configurations consistent.
5. Record the model versions.

**Important:** A fast embedding model is not automatically the best model for RAG. Retrieval quality should also be evaluated using metrics such as Recall@K, Precision@K, and MRR (Mean Reciprocal Rank).

This benchmark measures embedding performance. It does not independently measure retrieval accuracy.