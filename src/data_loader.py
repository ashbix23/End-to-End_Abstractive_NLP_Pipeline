import os
from typing import Tuple, Optional
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_ROOT = os.path.join(PROJECT_ROOT, "data", "cache")

def _strip_columns(ds: DatasetDict, keep=("article", "abstract")) -> DatasetDict:
    for split in ds.keys():
        cols = ds[split].column_names
        drop = [c for c in cols if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)
    return ds

def _preprocess_fn(tokenizer, max_input_len: int, max_target_len: int):
    pad_id = tokenizer.pad_token_id

    def fn(examples):
        inputs = examples["article"]
        targets = examples["abstract"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_len,
            truncation=True,
            padding=False
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_len,
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = [
            [(tid if tid != pad_id else -100) for tid in seq]
            for seq in labels["input_ids"]
        ]
        return model_inputs

    return fn

def _cache_path(tag: str) -> str:
    return os.path.join(CACHE_ROOT, tag)

def load_raw_dataset(
    dataset_name: str = "scientific_papers",
    subset: str = "arxiv"
) -> DatasetDict:
    if dataset_name == "scientific_papers" and subset not in {"arxiv", "pubmed"}:
        raise ValueError("subset must be 'arxiv' or 'pubmed' for scientific_papers.")
    ds = load_dataset(dataset_name, subset)
    ds = _strip_columns(ds, keep=("article", "abstract"))
    return ds

def load_or_build_tokenized(
    dataset_name: str = "scientific_papers",
    subset: str = "arxiv",
    model_name: str = "facebook/bart-base",
    max_input_len: int = 1024,
    max_target_len: int = 256,
    tag: Optional[str] = None
) -> Tuple[DatasetDict, DatasetDict]:
    """
    Returns (raw_dataset, tokenized_dataset).
    Caches both to disk under data/cache/<tag>_raw and <tag>_tok.
    """
    os.makedirs(CACHE_ROOT, exist_ok=True)

    if tag is None:
        tag = f"{dataset_name}_{subset}_bartbase_{max_input_len}_{max_target_len}"

    raw_cache = _cache_path(f"{tag}_raw")
    tok_cache = _cache_path(f"{tag}_tok")

    if os.path.isdir(raw_cache) and os.path.isdir(tok_cache):
        raw = load_from_disk(raw_cache)
        tok = load_from_disk(tok_cache)
        return raw, tok

    raw = load_raw_dataset(dataset_name=dataset_name, subset=subset)
    raw.save_to_disk(raw_cache)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    preprocess = _preprocess_fn(tokenizer, max_input_len, max_target_len)

    tok = raw.map(
        preprocess,
        batched=True,
        desc=f"Tokenizing {dataset_name}/{subset}",
        remove_columns=[]
    )
    tok.save_to_disk(tok_cache)

    return raw, tok
