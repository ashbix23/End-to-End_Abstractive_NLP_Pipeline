import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------
# Load model and tokenizer
# ------------------------------------------------------------

MODEL_PATH = "ashbeexd/bart-summariser-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# ------------------------------------------------------------
# Define summarization function
# ------------------------------------------------------------
def summarize(text, max_length=128, min_length=30, num_beams=4):
    text = text.strip()
    if not text:
        return "Please enter text to summarize."

    # Limit overly long inputs
    if len(text.split()) > 1000:
        return "Input too long (max 1000 words). Please shorten the text."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# ------------------------------------------------------------
# Define Gradio interface
# ------------------------------------------------------------
title = "Scientific Paper Summarizer"
description = """
A fine-tuned BART-base model trained on the scientific_papers/arxiv dataset.
It generates concise summaries of long scientific articles.

Instructions:
- Paste or type a research abstract or section below.
- Adjust summary length if desired.
- Click 'Summarize' to generate your result.
"""

examples = [
    ["Quantum entanglement is a phenomenon where particles become interconnected..."],
    ["This paper introduces a novel transformer-based model for protein structure prediction..."]
]

demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Input Text", lines=12, placeholder="Paste your scientific article or abstract here..."),
        gr.Slider(30, 256, value=128, step=1, label="Maximum Summary Length"),
        gr.Slider(10, 100, value=30, step=1, label="Minimum Summary Length"),
        gr.Slider(1, 8, value=4, step=1, label="Beam Search Width")
    ],
    outputs=gr.Textbox(label="Generated Summary", lines=10),
    title=title,
    description=description,
    examples=examples,
    allow_flagging="never",
    theme="default"
)

# ------------------------------------------------------------
# Launch app
# ------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
