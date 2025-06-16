from transformers import AutoTokenizer
import os
from src.config import MODEL_NAME


def save_model(model, output_dir="models/sentiment_roberta"):
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    model.model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to: {output_dir}")
