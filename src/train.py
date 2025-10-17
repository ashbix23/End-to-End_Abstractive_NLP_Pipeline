import os, multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["DATASETS_DISABLE_PARALLELISM"] = "1"
os.environ["ARROW_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
multiprocessing.set_start_method("spawn", force=True)


from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from src.data_loader import load_or_build_tokenized
from src.model import get_model, get_tokenizer
from src.seed_utils import set_seed
import torch
torch.set_num_threads(1)


def get_training_args(output_dir: str = "outputs/model"):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="outputs/logs",
        logging_strategy="epoch",
        predict_with_generate=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="none",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
    )


def train_model(
    model_name: str = "facebook/bart-base",
    dataset_name: str = "scientific_papers",
    subset: str = "arxiv",
    max_input_len: int = 1024,
    max_target_len: int = 256,
    tag: str = "arxiv_bartbase"
):
    """
    Full training pipeline:
    - loads cached dataset
    - loads model/tokenizer
    - sets seed for reproducibility
    - trains with early stopping
    """
    set_seed(42)

    # 1. Load datasets
    raw, tok = load_or_build_tokenized(
        dataset_name=dataset_name,
        subset=subset,
        model_name=model_name,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
        tag=tag
    )

    # 2. Model + Tokenizer
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    # 3. Data Collator (handles dynamic padding)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    # 4. Training Arguments
    args = get_training_args()

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tok["train"].select(range(1000)),  # subset for faster dev run
        eval_dataset=tok["validation"].select(range(300)),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Saving model...")

    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    train_model()
