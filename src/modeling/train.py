import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from src.config import MODEL_NAME, NUM_LABELS, LEARNING_RATE, BATCH_SIZE, EPOCHS
from src.modeling.utils import save_model


class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


def evaluate(model, val_loader, device):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")
    return acc, f1


def train_model(train_dataset, val_dataset):
    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    device = "cpu"

    model = SentimentClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc, f1 = evaluate(model, val_loader, device)
        print(f"Val Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    save_model(model)
    return model
