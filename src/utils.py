import re

def clean_summary(text):
    """
    Cleans up the generated summary text.
    Removes repeated punctuation, excessive whitespace, and token artifacts.
    """
    text = text.replace("<n>", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?])\1+", r"\1", text)  # remove repeated punctuation
    return text.strip()


def print_sample_predictions(inputs, predictions, references, num_samples=3):
    """
    Pretty-prints sample inputs, generated summaries, and reference abstracts.
    """
    for i in range(min(num_samples, len(inputs))):
        print(f"\n=== Sample {i + 1} ===")
        print(f"\nInput Article:\n{inputs[i][:500]}...")
        print(f"\nGenerated Summary:\n{predictions[i]}")
        print(f"\nReference Summary:\n{references[i]}")
        print("=" * 50)


def chunk_text_list(texts, chunk_size=4):
    """
    Break a list of texts into chunks (used for batch inference).
    """
    for i in range(0, len(texts), chunk_size):
        yield texts[i:i + chunk_size]

