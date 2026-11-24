# OCR and Document Processing Benchmarking Documentation

## 1. Objective

The purpose of this benchmark is to investigate and compare OCR technologies for extracting text from documents in the RAG system.

OCR (Optical Character Recognition) converts text in images or scanned documents into machine-readable text.

The benchmark focuses on:

- Text extraction time.
- Character Error Rate (CER).
- Word Error Rate (WER).
- Extracted character count.
- English and Khmer language support.

## 2. OCR Technologies to Evaluate

The current Python benchmark implementation evaluates two OCR engines.

| Technology | Type | Description |
|---|---|---|
| Tesseract OCR | Traditional OCR engine | Open-source OCR engine supporting multiple languages |
| EasyOCR | Deep-learning OCR library | Python-based OCR library supporting multiple languages |

### Additional document-processing technologies for research

These technologies are part of the broader research scope, but they are not automatically tested by the current Tesseract/EasyOCR Python script.

| Technology | Description |
|---|---|
| Qwen2.5-VL-7B | Vision-language model for understanding document images |
| Qwen3-VL-8B | Vision-language model for visual understanding |
| Docling | Document processing and conversion framework |
| MinerU | Document parsing framework |
| Qwen3-VL-Embedding-2B | Multimodal embedding model |
| ColPali / ColQwen2 | Visual document retrieval approaches |

A separate integration and evaluation process is needed to benchmark these technologies fairly.

## 3. Prerequisites

### Software

- Python 3.10+
- Tesseract OCR
- VS Code or another Python IDE

### Python dependencies

Install the required libraries using PowerShell:

```powershell
py -m pip install pytesseract easyocr pillow
```

### Python standard-library modules

- csv
- time
- datetime
- pathlib
- argparse

## 4. Tesseract OCR Preparation

Tesseract must be installed separately. Installing `pytesseract` alone is not enough.

### Windows

Install Tesseract OCR and configure its executable path in the Python script.

Example:

```python
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

Change the path if Tesseract is installed somewhere else.

### Khmer language support

Tesseract needs the appropriate language data files.

The benchmark uses:

```python
TESSERACT_LANG = "eng+khm"
```

This requires both English and Khmer trained data to be installed.

### Linux / WSL

For Ubuntu or Kali Linux, install Tesseract and its language packages:

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-eng
sudo apt install tesseract-ocr-khm
```

Check installation:

```bash
tesseract --version
```

Check available languages:

```bash
tesseract --list-langs
```

## 5. EasyOCR Preparation

EasyOCR is installed through pip:

```powershell
py -m pip install easyocr
```

The current benchmark initializes EasyOCR with English:

```python
reader = easyocr.Reader(["en"], gpu=False)
```

**Important:** This configuration supports English only. Khmer OCR support should not be assumed. If Khmer is a requirement, investigate and validate an appropriate OCR solution separately.

The current script also skips images with `khmer` in their filenames when using EasyOCR.

## 6. Test Dataset Preparation

Create a directory named:

```text
test_documents/
```

Place your test images and their corresponding ground-truth text files inside it.

Example:

```text
test_documents/
├── english_document.png
├── english_document.txt
├── english_receipt.jpg
├── english_receipt.txt
├── khmer_document.png
├── khmer_document.txt
└── scanned_form.jpg
```

### Ground-truth text

Each `.txt` file contains the correct text extracted or transcribed manually from its corresponding image.

For example:

**english_document.png**

An image containing:

> This is a sample document for OCR testing.

**english_document.txt**

```text
This is a sample document for OCR testing.
```

The matching filename allows the script to compare OCR output against the reference text.

Supported image formats in the current script:

- PNG
- JPG
- JPEG
- TIF
- TIFF

## 7. Benchmark Methodology

The script supports selecting an OCR engine through a command-line argument.

### Run Tesseract

```powershell
py benchmark_ocr.py --engine tesseract --runs 3
```

### Run EasyOCR

```powershell
py benchmark_ocr.py --engine easyocr --runs 3
```

The script performs these steps:

1. Read the images from `test_documents/`.
2. Select the requested OCR engine.
3. Extract text from each image.
4. Measure processing time.
5. Compare extracted text against the corresponding ground-truth `.txt` file when available.
6. Calculate CER and WER.
7. Export results to CSV.

**Note:** Run these commands from the directory containing `benchmark_ocr.py`, or provide the correct relative path to the script.

## 8. Evaluation Metrics

| Metric | Description |
|---|---|
| Processing time | Time required to extract text from an image |
| Extracted characters | Number of characters returned by OCR |
| CER | Character Error Rate |
| WER | Word Error Rate |

### Character Error Rate (CER)

CER measures the character-level difference between the OCR output and the ground-truth text.

\[
CER = \frac{S + D + I}{N}
\]

Where:

- S = Number of character substitutions.
- D = Number of character deletions.
- I = Number of character insertions.
- N = Number of characters in the reference text.

### Word Error Rate (WER)

WER measures the word-level difference between the OCR output and the ground-truth text.

\[
WER = \frac{S + D + I}{N}
\]

In this formula, S, D, and I represent word-level errors, and N is the number of words in the reference text.

**Interpretation:** Lower CER and WER indicate better OCR accuracy.

## 9. Output

The script saves the benchmark results to:

```text
results/ocr_results.csv
```

Example CSV structure:

| Engine | Image | Run | Processing Time | CER | WER |
|---|---|---:|---:|---:|---:|
| Tesseract | english_document.png | 1 | 0.85 | 0.05 | 0.10 |
| EasyOCR | english_document.png | 1 | 1.20 | 0.03 | 0.08 |

*These values are illustrative, not actual results.*

## 10. Execution Checklist

- [ ] Install Python dependencies.
- [ ] Install Tesseract OCR.
- [ ] Configure the Tesseract executable path.
- [ ] Install English and Khmer Tesseract language data.
- [ ] Prepare test images.
- [ ] Prepare corresponding ground-truth text files.
- [ ] Run the Tesseract benchmark.
- [ ] Run the EasyOCR benchmark.
- [ ] Collect CSV results.
- [ ] Compare processing time, CER, and WER.

## 11. Notes and Limitations

1. OCR accuracy depends heavily on image quality, resolution, fonts, and document layout.
2. Use the same images and ground-truth text for each engine.
3. EasyOCR's current configuration uses English only.
4. Tesseract supports Khmer only when the appropriate language data is installed.
5. CER and WER require corresponding ground-truth text files.
6. The current script does not benchmark Qwen vision-language models, Docling, MinerU, or multimodal embedding models.
7. Model initialization and warm-up should be handled consistently when comparing processing times.

For a more comprehensive document-processing evaluation, consider measuring:

- Table extraction accuracy.
- Reading-order preservation.
- Layout detection.
- Heading and paragraph extraction.
- Khmer and English mixed-document accuracy.
- Structured output quality.

These additional metrics will help determine which technology is most suitable for the RAG document-ingestion pipeline.