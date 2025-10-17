import os
import json
from typing import Dict, List
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

from src.seed_utils import set_seed

# Load metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def decode_batch(tokenizer, sequences: List[List[int]]) -> List[str]:
    """
    Decodes token IDs back into text and cleans spacing.
    """
    return tokenizer.batch_decode(sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def prepare_labels_for_decode(dataset_split, tokenizer) -> List[List[int]]:
    """
    Converts -100s in label tensors back to pad_token_id for decoding.
    """
    pad_id = tokenizer.pad_token_id
    return [[(tid if tid != -100 else pad_id) for tid in row["labels"]] for row in dataset_split]

def evaluate_model(
    model_path: str = "outputs/model",
    tokenized_cache: str = "data/cache/scientific_papers_arxiv_bartbase_1024_256_tok",
    split: str = "validation",
    max_new_tokens: int = 128,
    num_beams: int = 4,
    save_dir: str = "outputs/eval",
    num_samples_to_save: int = 5
) -> Dict[str, float]:
    """
    Evaluates the model on a given dataset split.
    Outputs ROUGE, BERTScore, and qualitative examples.
    """

    os.makedirs(save_dir, exist_ok=True)
    set_seed(42)

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    print(f"Loading tokenized dataset from {tokenized_cache}")
    ds = load_from_disk(tokenized_cache)[split]

    # Decode references
    print("Preparing reference summaries...")
    label_ids = prepare_labels_for_decode(ds, tokenizer)
    references = decode_batch(tokenizer, label_ids)

    # Generate predictions
    print("Generating summaries...")
    predictions = []
    for row in tqdm(ds.select(range(len(references))), total=len(references)):
        input_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )[0].tolist()
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)
        predictions.append(pred)

    print("Computing ROUGE and BERTScore...")
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

    result = {
        "rouge1": round(rouge_scores["rouge1"] * 100, 2),
        "rouge2": round(rouge_scores["rouge2"] * 100, 2),
        "rougeL": round(rouge_scores["rougeL"] * 100, 2),
        "bertscore_precision": round(sum(bert_scores["precision"]) / len(bert_scores["precision"]) * 100, 2),
        "bertscore_recall": round(sum(bert_scores["recall"]) / len(bert_scores["recall"]) * 100, 2),
        "bertscore_f1": round(sum(bert_scores["f1"]) / len(bert_scores["f1"]) * 100, 2)
    }

    # Save results
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Save sample predictions
    examples = []
    for i in range(min(num_samples_to_save, len(predictions))):
        examples.append({
            "input_excerpt": tokenizer.decode(ds[i]["input_ids"][:300], skip_special_tokens=True),
            "pred_summary": predictions[i],
            "ref_summary": references[i],
        })
    with open(os.path.join(save_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print(f"\nSaved metrics to {save_dir}/metrics.json")
    print(f"Saved qualitative examples to {save_dir}/examples.json")

    return result

if __name__ == "__main__":
    evaluate_model()
