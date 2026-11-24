# LLM Benchmarking Documentation

## 1. Objective

The purpose of this benchmark is to evaluate and compare different local Large Language Models (LLMs) using Ollama.

The benchmark focuses on:
- Response generation latency.
- Model loading time.
- Prompt token count.
- Generated token count.
- Generation speed (tokens per second).
- Response consistency under the same prompt.

## 2. Models to Evaluate

| Model | Parameters | Description |
|---|---:|---|
| Qwen3:8b | 8B | General-purpose language model |
| Llama3.1:8b | 8B | Meta's general-purpose language model |
| Gemma3:4b | 4B | Lightweight language model |
| Qwen3:4b | 4B | Smaller Qwen model |

## 3. Prerequisites

### Software
- Python 3.10+
- Ollama
- VS Code or another Python IDE

### Python dependencies

Install the required package using PowerShell:

```powershell
py -m pip install requests
```

The following modules are included in Python's standard library:

- csv
- time
- argparse
- datetime
- pathlib

## 4. Ollama Preparation

First, make sure Ollama is installed and running.

Download the models:

```powershell
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull gemma3:4b
ollama pull qwen3:4b
```

Check downloaded models:

```powershell
ollama list
```

Ollama provides a local API endpoint:

```text
http://localhost:11434/api/generate
```

## 5. Benchmark Methodology

The Python benchmark script sends the same predefined prompt to each model.

Example prompt:

> Explain what Retrieval-Augmented Generation (RAG) is. Describe its main components and how they work together. Keep your explanation simple and clear.

Each model receives the same prompt and is tested multiple times.

### Example execution

Run the Qwen3 8B benchmark:

```powershell
py benchmark_llm.py --model qwen3:8b --runs 3
```

Repeat the process for each model:

```powershell
py benchmark_llm.py --model llama3.1:8b --runs 3

py benchmark_llm.py --model gemma3:4b --runs 3

py benchmark_llm.py --model qwen3:4b --runs 3
```

**Note:** Run these commands from the directory containing `benchmark_llm.py`, or provide the correct relative path to the script.

## 6. Evaluation Metrics

| Metric | Description |
|---|---|
| Latency (seconds) | Total time required to generate a response |
| Model loading time | Time spent loading the model, as reported by Ollama |
| Prompt tokens | Number of input tokens |
| Generated tokens | Number of output tokens |
| Tokens per second | Generation speed |
| Run count | Number of benchmark executions |

**Important:** If the script does not perform a warm-up request, the first run may include model loading time. Record this when interpreting results.

## 7. Output

The script exports benchmark results to a CSV file.

Expected output:

```text
results.csv
```

The file is saved beside the Python script in the current implementation.

Example CSV structure:

| Model | Run | Latency | Generated Tokens | Tokens/sec |
|---|---:|---:|---:|---:|
| qwen3:8b | 1 | 5.21 | 150 | 28.79 |
| qwen3:8b | 2 | 4.89 | 148 | 30.27 |
| qwen3:8b | 3 | 4.95 | 151 | 30.51 |

*The values above are illustrative, not actual benchmark results.*

## 8. Execution Checklist

- [ ] Install Ollama.
- [ ] Download all required models.
- [ ] Install Python dependencies.
- [ ] Ensure Ollama is running.
- [ ] Execute the benchmark for each model.
- [ ] Collect CSV results.
- [ ] Calculate average latency and generation speed.
- [ ] Compare model performance.

## 9. Notes

For a fair comparison:

1. Use the same prompt for every model.
2. Use the same number of runs.
3. Avoid running other demanding applications during testing.
4. Keep hardware and Ollama settings consistent.
5. Record the model names and versions.
6. Distinguish the first run from subsequent runs when model loading affects the results.

The fastest model is not necessarily the best model. Response quality should also be evaluated because this benchmark primarily measures performance, not factual accuracy.