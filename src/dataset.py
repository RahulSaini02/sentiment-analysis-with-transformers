from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.features import preprocess
from src.config import MODEL_NAME, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess(self.texts[idx])
        label = self.labels[idx]
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": label,
        }