# End-to-End Abstractive NLP Pipeline (BART)

[![Hugging Face Model](https://img.shields.io/badge/HF%20Model-ashbix23%2FBART--Summarizer-blue)](https://huggingface.co/ashbix23/bart-summariser-model)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Demo-Live%20Gradio%20App-green)](https://huggingface.co/spaces/ashbix23/text-summarisation-hf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](#environment)

**Complete MLOps pipeline for Abstractive Text Summarization using Hugging Face's BART-base model, fine-tuned on the `scientific_papers/arxiv` dataset. Features include structured training, ROUGE/BERTScore evaluation, and deployment to the Hugging Face Hub and Spaces.**

---

## Project Highlights

This project goes beyond a simple fine-tuning task, showcasing expertise in building reproducible, production-ready NLP systems:

* **Domain Specialization**: Model is robustly fine-tuned on the **scientific\_papers/arxiv** corpus (article $\rightarrow$ abstract), ensuring high-quality, domain-appropriate summarization.
* **Structured MLOps Pipeline**: Features a clean `src/` package structure with explicit stages (data ingestion, tokenization, training, evaluation, artifact management).
* **Reproducibility & Stability**: Uses deterministic seeds, implements **disk-cached datasets** (for efficiency), pins dependency requirements, and provides specific stability flags for multi-OS support.
* **Comprehensive Evaluation**: Utilizes both **ROUGE-1/2/L** (lexical overlap) and the advanced **BERTScore** (semantic similarity) to provide a complete view of model performance.
* **Deployment Ready**: The fine-tuned model is versioned and hosted on the **Hugging Face Hub** (`ashbix23/bart-summariser-model`) and consumed by a public **Gradio Space** demo.

---

## Quick Links (Deployment Artifacts)

| Resource | Value | Link (Username: `ashbix23`) |
| :--- | :--- | :--- |
| **Hosted Model** | Model Checkpoint & Versioning | [`ashbix23/bart-summariser-model`](https://huggingface.co/ashbix23/bart-summariser-model) |
| **Interactive Demo** | Live Gradio Deployment | [`ashbix23/text-summarisation-hf`](https://huggingface.co/spaces/ashbix23/text-summarisation-hf) |

---

## Evaluation Results (Validation Subset)

Evaluation was performed on a validation subset after fine-tuning on a **40k train / 4k val** slice, a common practice for rapid iteration and establishing a strong baseline.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| ROUGE-1 | 36.80 | High lexical overlap with human abstracts. |
| ROUGE-2 | 13.29 | Good performance on capturing key bi-grams/phrases. |
| ROUGE-L | 22.33 | Strong longest common subsequence overlap. |
| **BERTScore-F1** | 85.17 | Confirms high **semantic similarity** between prediction and reference. |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── app.py                      
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py         # download → tokenize → cache (project-root paths)
    ├── model.py               # tokenizer/model getters
    ├── train.py               # CLI training entry (optional; notebooks preferred)
    ├── evaluate.py            # scriptable evaluation
    ├── utils.py               # shared helpers
    └── seed_utils.py          # deterministic seeds
```

**Outputs** (created at runtime):
```
outputs/
├── model/                     # final exported checkpoint (for local eval)
└── eval/
    ├── metrics.json
    └── examples.json
```

**Cache**:
```
data/cache/
└── scientific_papers_arxiv_bartbase_1024_256_{raw,tok}/
```

> Important: All paths resolve **relative to the project root**, not the notebook working directory. This prevents duplicate caches under `notebooks/`.

---

## 🛠️ Environment and Setup

Targeted for **Python 3.10+**. Supports standard CPU and Apple Silicon (MPS).

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Stability Flags (For macOS/Jupyter)**:
    If encountering import stalls or parallelism issues, set these environment variables **before** importing `transformers` or `torch`:

    ```python
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
    # ... (other stability flags)
    ```

---

## Data

- **Dataset**: [`scientific_papers`](https://huggingface.co/datasets/scientific_papers) → `arxiv` split
- **Fields**: `article` (input), `abstract` (target)
- **Default lengths**: `max_input_len=1024`, `max_target_len=256`

The loader will download and cache both **raw** and **tokenized** datasets to `data/cache/` on first run.

---

## Notebooks (Preferred Workflow)

The notebooks are the authoritative way to run the project end-to-end.

### 1) 01_dataset_exploration.ipynb
- Loads raw + tokenized datasets.
- Prints samples and plots token length distributions (articles vs abstracts).
- Verifies cache integrity.

### 2) 02_model_training.ipynb
- Loads BART-base and tokenized data.
- Uses `Seq2SeqTrainer` with MPS/CPU-friendly defaults.
- Example fast path: subset training (**40k train / 4k val**) with 1–2 epochs for a strong baseline.
- Saves the final model to `outputs/model/`.

Key training args (representative):
```
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=4
learning_rate=5e-5
num_train_epochs=2
predict_with_generate=True
save_strategy="epoch"
load_best_model_at_end=True
```

### 3) 03_model_evaluation.ipynb
- Loads the saved checkpoint from `outputs/model/`.
- Generates summaries for a validation subset (configurable).
- Computes **ROUGE** and **BERTScore**; writes JSON artifacts.
- Plots a simple ROUGE bar chart.

---

## Scripts (Optional CLI)

Notebooks are recommended. If you prefer CLI, you can run:

```bash
python -m src.train
```

If you encounter `libc++abi` or mutex issues on macOS with CLI runs, prefer the notebooks or add the environment flags shown in the **Environment** section.

Evaluation from CLI:
```bash
python -m src.evaluate
```

---

## Model & Inference Notes

- **Backbone**: `facebook/bart-base`
- **Tokenizer**: BPE (uses `vocab.json` + `merges.txt`)
- **Generation**: Beam search by default (`num_beams=4`, `early_stopping=True`)
- **Input limits**: Truncated to 1024 tokens; summaries targeted at ~128 tokens (min 30, max 256 configurable)

**Beam Search Width (num_beams)**
- 1: greedy, fastest, can be dull
- 4–6: balanced quality (default)
- 8+: thorough but slower; may bias toward generic phrasing

---

## Reproducibility

- **Seeding**: `seed_utils.set_seed(42)` applied across NumPy/PyTorch.
- **Caching**: Raw and tokenized datasets saved under `data/cache/`, keyed by model + length settings.
- **Artifacts**: Metrics and examples saved under `outputs/eval/`.

To reset and rebuild caches cleanly:
```python
import shutil, os
shutil.rmtree("data/cache", ignore_errors=True)
os.makedirs("data/cache", exist_ok=True)
```

---

## Known Pitfalls & Fixes

- **Jupyter import hang on macOS**: Set the environment flags under *Environment → Stability flags* before imports.
- **Duplicate caches under `notebooks/`**: Caused by relative paths; this repo resolves paths from project root to prevent it.
- **Transformers/Tokenizers version conflicts**: Ensure `transformers>=4.40.1` and remove overly pinned `tokenizers` versions.
- **Long training times on CPU**: Start with a subset (e.g., 40k/4k) to validate pipeline; scale up on GPU later.

---

## Future Work and Technical Roadmap

This section outlines planned enhancements to further industrialize the summarization pipeline and explore advanced GenAI techniques.

### 1. Algorithmic Extensions (Deepening Mastery)

| Feature | Technical Goal |
| :--- | :--- |
| **Full Parameter Fine-Tuning (LoRA/QLoRA)** | Implement Parameter-Efficient Fine-Tuning (PEFT) using LoRA or QLoRA to reduce memory footprint and training time. |
| **Integrate T5 or Pegasus** | Benchmark an alternative model architecture (e.g., T5 or Pegasus) on the same task/dataset. |


### 2. Robustness and Software Engineering

| Feature | Technical Goal |
| :--- | :--- |
| **MLflow Experiment Tracking** | Integrate MLflow to log training runs, hyperparameters, and ROUGE/BERTScore metrics for systematic experiment comparison. |
| **Custom Data Preprocessing** | Add PDF parsing or LaTeX stripping to allow direct ingestion of raw scientific papers (not just pre-cleaned Hugging Face data). |

---

## Citation

If you use this code/model in academic work, please cite BART and the Hugging Face ecosystem:

- Lewis et al., 2020. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.*
- Hugging Face Transformers, Datasets, and Evaluate libraries.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

