from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_tokenizer(model_name: str = "facebook/bart-base"):
    """
    Loads a tokenizer for the given model name.
    Uses fast tokenizer when available.
    """
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def get_model(model_name: str = "facebook/bart-base"):
    """
    Loads a pretrained seq2seq model (BART or T5 style).
    """
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)
