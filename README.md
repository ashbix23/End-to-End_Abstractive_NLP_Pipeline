# Scientific Paper Summarizer (BART)

[![Hugging Face Model](https://img.shields.io/badge/HF%20Model-ashbeexd%2Fbart--summariser--model-blue)](https://huggingface.co/ashbeexd/bart-summariser-model)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Space-ashbeexd%2Ftext--summarisation--hf-green)](https://huggingface.co/spaces/ashbeexd/text-summarisation-hf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](#environment)


Abstractive summarization pipeline for scientific text using **BART-base** fine-tuned on Hugging Face’s **`scientific_papers/arxiv`** dataset. The project covers the complete workflow: data ingestion and caching, training (CPU/MPS-friendly), evaluation (ROUGE + BERTScore), and deployment to a public demo on Hugging Face (Space + hosted model repo). Notebooks are included for end-to-end reproducibility.

---

## Highlights

- **Domain-appropriate data**: Trained on the `scientific_papers/arxiv` split (articles → abstracts), not news data.
- **Solid engineering**: Clean `src/` package with explicit stages (load/tokenize/cache, train, evaluate). Safe defaults for macOS/MPS and CPU.
- **Reproducible**: Deterministic seeds, disk-cached datasets, pinned requirements, and self-contained notebooks.
- **Meaningful evals**: ROUGE-1/2/L and BERTScore, plus qualitative example dumps.
- **Demo-ready**: Model hosted on the Hugging Face Hub; Gradio Space consumes the hosted model. No local app hosting required.

---

## Quick Links

- **Hosted Model (Hugging Face Hub)**: `ashbeexd/bart-summariser-model`
- **Interactive Demo (Space)**: `ashbeexd/text-summarisation-hf`

> Note: The app is deployed on Hugging Face; this repository does not include app configuration code beyond a simple `app.py` reference for local sanity checks.

---

## Results (Validation)

Evaluation was performed on a validation **subset** with BART-base fine-tuned on a **40k train / 4k val** slice for faster iteration. The following scores are representative and reproducible with the provided notebooks.

| Metric | Score |
|---|---:|
| ROUGE-1 | 36.80 |
| ROUGE-2 | 13.29 |
| ROUGE-L | 22.33 |
| BERTScore-F1 | 85.17 |

Artifacts:
- `outputs/eval/metrics.json` — numeric results
- `outputs/eval/examples.json` — qualitative samples (input excerpt, prediction, reference)

> Interpreting the numbers: On scientific long-form text, these scores are in the expected band for BART-base with moderate training. ROUGE emphasizes lexical overlap; BERTScore captures semantic similarity (contextual embeddings).

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

## Environment

Targeted for **Python 3.10+**. macOS/CPU and Apple Silicon (MPS) supported.

```
pip install -r requirements.txt
```

`requirements.txt` (key deps):
```
transformers>=4.40.1
torch>=2.2.0
evaluate>=0.4.1
bert-score>=0.3.13
rouge-score>=0.1.2
gradio>=5.49.1    # for quick local sanity checks; Space hosts the app remotely
```

### Stability flags (Jupyter/macOS)
If you see import stalls in notebooks, set the following **before** importing Transformers/Torch:
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["DATASETS_DISABLE_PARALLELISM"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
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

## Citation

If you use this code/model in academic work, please cite BART and the Hugging Face ecosystem:

- Lewis et al., 2020. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.*
- Hugging Face Transformers, Datasets, and Evaluate libraries.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

