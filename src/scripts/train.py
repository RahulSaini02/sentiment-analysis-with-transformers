import pandas as pd
from datasets import load_dataset
from utils import resolve_root

resolve_root()

from src.dataset import TweetDataset
from src.modeling.train import train_model
from sklearn.model_selection import train_test_split

# 1. Load dataset
dataset_name = "cardiffnlp/tweet_eval"
dataset_subset = "sentiment"

dataset = load_dataset(dataset_name, dataset_subset)
df = pd.DataFrame(dataset["train"])

# 2. Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# 3. Create datasets
train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)

# 4. Train
train_model(train_dataset, val_dataset)
