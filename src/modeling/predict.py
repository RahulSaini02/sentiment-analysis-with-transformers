import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.features import preprocess
from src.config import MODEL_NAME, LABEL_MAP, MAX_LENGTH


def load_model(model_dir: str = "models/sentiment_roberta"):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def predict_sentiment(text: str, model_path: str = "models/sentiment_roberta"):
    model, tokenizer = load_model(model_path)

    text = preprocess(text)
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        output = model(**encoding)
        probs = torch.softmax(output.logits, dim=1).squeeze()  # probabilities
        pred = torch.argmax(probs).item()  # predicted class index

    # Build class prob dictionary
    prob_dict = {LABEL_MAP[i]: float(probs[i]) for i in range(len(LABEL_MAP))}

    predicted_label = LABEL_MAP[pred]
    return predicted_label, prob_dict
